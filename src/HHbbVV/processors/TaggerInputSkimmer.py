import numpy as np
import awkward as ak
import pandas as pd
import uproot

from coffea.processor import ProcessorABC, dict_accumulator
from coffea.analysis_tools import PackedSelection

from .utils import add_selection_no_cutflow
from .TaggerInference import get_pfcands_features, get_svs_features
from .GenSelection import tagger_gen_matching

from typing import Dict

import os
import pathlib
import json


P4 = {
    "eta": "eta",
    "phi": "phi",
    "mass": "mass",
    "pt": "pt",
}


"""
TODOs:
    - More signal labels
    - Is saving as parquet the way to go? Could also potentially save to root files?
        - If yes to parquet, need to update ak_to_pandas function
"""


class TaggerInputSkimmer(ProcessorABC):
    """
    Produces a flat training ntuple from PFNano.
    """

    def __init__(self, label="AK15_H_VV", num_jets=2):
        self.skim_vars = {
            "FatJet": {
                **P4,
                "msoftdrop": "msoftdrop",
            },
            "GenPart": [
                "fj_genjetmsd",
                "fj_genjetmass",
                "fj_nprongs",
                "fj_H_VV_4q",
                "fj_H_VV_elenuqq",
                "fj_H_VV_munuqq",
                "fj_H_VV_taunuqq",
                "fj_H_VV_unmatched",
                "fj_dR_V",
                "fj_genV_pt",
                "fj_genV_eta",
                "fj_genV_phi",
                "fj_genV_mass",
                "fj_dR_Vstar",
                "fj_genVstar_pt",
                "fj_genVstar_eta",
                "fj_genVstar_phi",
                "fj_genVstar_mass",
                "fj_isQCDb",
                "fj_isQCDbb",
                "fj_isQCDc",
                "fj_isQCDcc",
                "fj_isQCDlep",
                "fj_isQCDothers",
            ],
            # formatted to match weaver's preprocess.json
            "PFSV": {
                "pf_features": {
                    "var_names": [
                        "pfcand_pt_log_nopuppi",
                        "pfcand_e_log_nopuppi",
                        "pfcand_etarel",
                        "pfcand_phirel",
                        "pfcand_isEl",
                        "pfcand_isMu",
                        "pfcand_isGamma",
                        "pfcand_isChargedHad",
                        "pfcand_isNeutralHad",
                        "pfcand_abseta",
                        "pfcand_charge",
                        "pfcand_VTX_ass",
                        "pfcand_lostInnerHits",
                        "pfcand_normchi2",
                        "pfcand_quality",
                        "pfcand_dz",
                        "pfcand_dzsig",
                        "pfcand_dxy",
                        "pfcand_dxysig",
                    ],
                },
                "pf_points": {"var_length": 130},  # number of pf cands to select or pad up to
                "sv_features": {
                    "var_names": [
                        "sv_pt_log",
                        "sv_mass",
                        "sv_etarel",
                        "sv_phirel",
                        "sv_abseta",
                        "sv_ntracks",
                        "sv_normchi2",
                        "sv_dxy",
                        "sv_dxysig",
                        "sv_d3d",
                        "sv_d3dsig",
                        "sv_costhetasvpv",
                    ],
                },
                "sv_points": {"var_length": 7},  # number of svs to select or pad up to
            },
        }
        self.fatjet_label = "FatJetAK15"
        self.pfcands_label = "FatJetAK15PFCands"
        self.svs_label = "JetSVsAK15"
        self.num_jets = num_jets
        self.label = label

        self.tagger_resources_path = (
            str(pathlib.Path(__file__).parent.resolve()) + "/tagger_resources/"
        )

        with open(f"{self.tagger_resources_path}/pyg_ef_ul_cw_8_2_preprocess.json") as f:
            self.tagger_vars = json.load(f)

        self._accumulator = dict_accumulator({})

    @property
    def accumulator(self):
        return self._accumulator

    def dump_table(self, pddf: pd.DataFrame, fname: str) -> None:
        """
        Saves pandas dataframe events to './outparquet'
        """
        import pyarrow.parquet as pq
        import pyarrow as pa

        local_dir = os.path.abspath(os.path.join(".", "outparquet"))
        os.system(f"mkdir -p {local_dir}")

        # need to write with pyarrow as pd.to_parquet doesn't support different types in
        # multi-index column names
        table = pa.Table.from_pandas(pddf)
        pq.write_table(table, f"{local_dir}/{fname}")

    def dump_root(self, jet_vars: Dict[str, np.array], fname: str) -> None:
        """
        Saves ``jet_vars`` dict as a rootfile to './outroot'
        """
        local_dir = os.path.abspath(os.path.join(".", "outroot"))
        os.system(f"mkdir -p {local_dir}")

        with uproot.recreate(f"{local_dir}/{fname}", compression=uproot.LZ4(4)) as rfile:
            rfile["Events"] = ak.Array(jet_vars)
            # rfile["Events"].show()

    def to_pandas_lists(self, events: Dict[str, np.array]) -> pd.DataFrame:
        """
        Convert our dictionary of numpy arrays into a pandas data frame.
        Uses lists for numpy arrays with >1 dimension
        (e.g. FatJet arrays with two columns)
        """
        output = pd.DataFrame()
        for field in ak.fields(events):
            if "sv_" in field or "pfcand_" in field:
                output[field] = events[field].tolist()
            else:
                output[field] = ak.to_numpy(ak.flatten(events[field], axis=None))

        return output

    def to_pandas(self, events: Dict[str, np.array]) -> pd.DataFrame:
        """
        Convert our dictionary of numpy arrays into a pandas data frame.
        Uses multi-index columns for numpy arrays with >1 dimension
        (e.g. FatJet arrays with two columns)
        """
        return pd.concat(
            [pd.DataFrame(v.reshape(v.shape[0], -1)) for k, v in events.items()],
            axis=1,
            keys=list(events.keys()),
        )

    def process(self, events: ak.Array):
        import time

        start = time.time()

        jet_vars = []

        for jet_idx in range(self.num_jets):
            # objects
            fatjets = ak.pad_none(events[self.fatjet_label], 2, axis=1)[:, jet_idx]
            genparts = events.GenPart

            # selection
            selection = PackedSelection()
            preselection_cut = (fatjets.pt > 250) * (fatjets.pt < 1500)
            add_selection_no_cutflow("preselection", preselection_cut, selection)

            print(f"preselection: {time.time() - start}s")

            # variables
            FatJetVars = {
                f"fj_{key}": ak.fill_none(fatjets[var], -99999)
                for (var, key) in self.skim_vars["FatJet"].items()
            }

            FatJetVars["fj_PN_XbbvsQCD"] = fatjets.ParticleNetMD_probXbb / (
                fatjets.ParticleNetMD_probQCD + fatjets.ParticleNetMD_probXbb
            )

            if "ParticleNet_probHqqqq" in fatjets:
                FatJetVars["fj_PN_H4qvsQCD"] = fatjets.ParticleNet_probHqqqq / (
                    fatjets.ParticleNet_probHqqqq
                    + fatjets.ParticleNet_probQCDb
                    + fatjets.ParticleNet_probQCDbb
                    + fatjets.ParticleNet_probQCDc
                    + fatjets.ParticleNet_probQCDcc
                    + fatjets.ParticleNet_probQCDothers
                )

            print(f"fat jet vars: {time.time() - start}s")

            PFSVVars = {
                **get_pfcands_features(
                    self.skim_vars["PFSV"],
                    events,
                    jet_idx,
                    self.fatjet_label,
                    self.pfcands_label,
                    normalize=False,
                ),
                **get_svs_features(
                    self.skim_vars["PFSV"],
                    events,
                    jet_idx,
                    self.fatjet_label,
                    self.svs_label,
                    normalize=False,
                ),
            }

            print(f"PFSV vars: {time.time() - start}s")

            matched_mask, genVars = tagger_gen_matching(
                events, genparts, fatjets, self.skim_vars["GenPart"], label=self.label, match_dR=1.0
            )

            add_selection_no_cutflow("gen_match", matched_mask, selection)

            print(f"Gen vars: {time.time() - start}s")

            skimmed_vars = {**FatJetVars, **genVars, **PFSVVars}
            # apply selections
            skimmed_vars = {
                key: np.array(value[selection.all(*selection.names)])
                for (key, value) in skimmed_vars.items()
            }

            jet_vars.append(skimmed_vars)

            print(f"Jet {jet_idx}: {time.time() - start}s")

        if self.num_jets > 1:
            # stack each set of jets
            jet_vars = {
                var: np.concatenate([jet_var[var] for jet_var in jet_vars], axis=0)
                for var in jet_vars[0]
            }
        else:
            jet_vars = jet_vars[0]

        print(f"Stack: {time.time() - start}s")

        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_")

        # convert output to pandas
        df = self.to_pandas(jet_vars)

        print(f"convert: {time.time() - start}s")

        # save to parquet
        self.dump_table(df, fname + ".parquet")
        # save to root (now
        # self.dump_root(jet_vars, fname + ".root")

        print(f"dumped: {time.time() - start}s")

        return {}

    def postprocess(self, accumulator):
        pass
