import numpy as np
import awkward as ak
import pandas as pd

from coffea.processor import ProcessorABC, dict_accumulator
from coffea.analysis_tools import PackedSelection

from .utils import add_selection_no_cutflow
from .TaggerInference import get_pfcands_features, get_svs_features
from .GenSelection import tagger_gen_matching

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

    def ak_to_pandas(self, output_collection: ak.Array) -> pd.DataFrame:
        output = pd.DataFrame()
        for field in ak.fields(output_collection):
            output[field] = ak.to_numpy(ak.flatten(output_collection[field], axis=None))
        return output

    def process(self, events: ak.Array):
        jet_vars = []

        for jet_idx in range(self.num_jets):
            # objects
            fatjets = ak.pad_none(events[self.fatjet_label], 2, axis=1)[:, jet_idx]
            genparts = events.GenPart

            # selection
            selection = PackedSelection()
            preselection_cut = (fatjets.pt > 200) * (fatjets.pt < 1500)
            add_selection_no_cutflow("preselection", preselection_cut, selection)

            # variables
            FatJetVars = {
                f"fj_{key}": ak.fill_none(fatjets[var], -99999)
                for (var, key) in self.skim_vars["FatJet"].items()
            }

            FatJetVars["fj_PN_XbbvsQCD"] = fatjets.ParticleNetMD_probXbb / (
                fatjets.ParticleNetMD_probQCD + fatjets.ParticleNetMD_probXbb
            )

            FatJetVars["fj_PN_H4qvsQCD"] = fatjets.ParticleNet_probHqqqq / (
                fatjets.ParticleNet_probHqqqq
                + fatjets.ParticleNet_probQCDb
                + fatjets.ParticleNet_probQCDbb
                + fatjets.ParticleNet_probQCDc
                + fatjets.ParticleNet_probQCDcc
                + fatjets.ParticleNet_probQCDothers
            )

            PFSVVars = {
                **get_pfcands_features(
                    self.tagger_vars,
                    events,
                    jet_idx,
                    self.fatjet_label,
                    self.pfcands_label,
                    normalize=False,
                ),
                **get_svs_features(
                    self.tagger_vars,
                    events,
                    jet_idx,
                    self.fatjet_label,
                    self.svs_label,
                    normalize=False,
                ),
            }

            matched_mask, genVars = tagger_gen_matching(
                events, genparts, fatjets, self.skim_vars["GenPart"], label=self.label
            )
            add_selection_no_cutflow("gen_match", matched_mask, selection)

            skimmed_vars = {**FatJetVars, **genVars, **PFSVVars}
            # apply selections
            skimmed_vars = {
                key: value[selection.all(*selection.names)] for (key, value) in skimmed_vars.items()
            }

            jet_vars.append(skimmed_vars)

        if self.num_jets > 1:
            # stack each set of jets
            jet_vars = {
                var: np.concatenate([jet_var[var] for jet_var in jet_vars], axis=0)
                for var in jet_vars[0]
            }
        else:
            jet_vars = jet_vars[0]

        # output
        df = self.ak_to_pandas(jet_vars)
        print(df)
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".root"
        self.dump_table(df, fname)

        return {}

    def postprocess(self, accumulator):
        pass
