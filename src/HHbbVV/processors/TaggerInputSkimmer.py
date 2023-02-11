"""
Skimmer for ParticleNet tagger inputs.

Author(s): Cristina Mantilla Suarez, Raghav Kansal
"""

import numpy as np
import awkward as ak
import pandas as pd
import uproot

from coffea.processor import ProcessorABC, dict_accumulator
from coffea.analysis_tools import PackedSelection

from .utils import add_selection_no_cutflow
from .TaggerInference import (
    get_pfcands_features,
    get_svs_features,
    get_lep_features,
    get_met_features,
)
from .GenSelection import tagger_gen_matching
from .TaggerInference import runInferenceTriton

from typing import Dict

import os
import pathlib
import json
import itertools


P4 = {
    "eta": "eta",
    "phi": "phi",
    "mass": "mass",
    "pt": "pt",
}


class TaggerInputSkimmer(ProcessorABC):
    """
    Produces a flat training ntuple from PFNano.
    """

    def __init__(self, label, num_jets=2):
        """
        :label: should be jet_(type of decay)
        e.g. AK15_H_VV for HVV AK15 decays
             AK8_H_qq for Hqq AK8 decays
        :type: str
        """

        self.label = label

        """
        Skimming variables

        Equivalence to other labels present in training datasets:
        ====
        (ignoring if V is W or Z)
        label_H_WqqWqq_0c: ( (fj_H_VV_4q==1) & (fj_nprongs==4) & (fj_ncquarks==0) )
        label_H_WqqWqq_1c: ( (fj_H_VV_4q==1) & (fj_nprongs==4) & (fj_ncquarks==1) )
        label_H_WqqWqq_2c: ( (fj_H_VV_4q==1) & (fj_nprongs==4) & (fj_ncquarks==2) )
        label_H_WqqWq_0c: ( (fj_H_VV_4q==1) & (fj_nprongs==3) & (fj_ncquarks==0) )
        label_H_WqqWq_1c: ( (fj_H_VV_4q==1) & (fj_nprongs==3) & (fj_ncquarks==1) )
        label_H_WqqWq_2c: ( (fj_H_VV_4q==1) & (fj_nprongs==3) & (fj_ncquarks==2) )
        label_H_WqqWev_0c: ( (fj_H_VV_elenuqq==1) & (fj_nprongs==2) & (fj_ncquarks==0) )
        label_H_WqqWev_1c: ( (fj_H_VV_elenuqq==1) & (fj_nprongs==2) & (fj_ncquarks==1) )
        label_H_WqqWmv_0c: ( (fj_H_VV_munuqq==1) & (fj_nprongs==2) & (fj_ncquarks==0) )
        label_H_WqqWmv_1c: ( (fj_H_VV_munuqq==1) & (fj_nprongs==2) & (fj_ncquarks==1) )
        label_H_WqqWtauev_0c: ( (fj_H_VV_leptauelvqq==1) & (fj_nprongs==2) & (fj_ncquarks==0) )
        label_H_WqqWtauev_1c: ( (fj_H_VV_leptauelvqq==1) & (fj_nprongs==2) & (fj_ncquarks==1) )
        label_H_WqqWtaumv_0c: ( (fj_H_VV_leptaumuvqq==1) & (fj_nprongs==2) & (fj_ncquarks==0) )
        label_H_WqqWtaumv_1c: ( (fj_H_VV_leptaumuvqq==1) & (fj_nprongs==2) & (fj_ncquarks==1) )
        label_H_WqqWtauhv_0c: ( (fj_H_VV_hadtauvqq==1) & (fj_nprongs==2) & (fj_ncquarks==0) )
        label_H_WqqWtauhv_1c: ( (fj_H_VV_hadtauvqq==1) & (fj_nprongs==2) & (fj_ncquarks==1) )

        rough equivalence of UCSD labels in terms of PKU labels:
        https://github.com/colizz/DNNTuples/blob/fa381ba419a8814390911dc980d0142c271ec166/FatJetHelpers/src/FatJetMatching.cc#L419-L441

        fj_H_VV_4q: ((label_H_WqqWqq_0c == 1) | (label_H_WqqWqq_1c == 1) | (label_H_WqqWqq_2c == 1) )
        fj_H_VV_3q: ((label_H_WqqWq_0c == 1) | (label_H_WqqWq_1c == 1) | (label_H_WqqWq_2c == 1) )
        fj_H_VV_elenuqq: ((label_H_WqqWev_0c == 1) | (label_H_WqqWev_1c == 1) )
        fj_H_VV_munuqq: ((label_H_WqqWmv_0c == 1) | (label_H_WqqWmv_1c == 1) )
        fj_H_VV_leptauelvqq: ((label_H_WqqWtauev_0c == 1) | (label_H_WqqWtauev_1c == 1) )
        fj_H_VV_leptaumuvqq: ((label_H_WqqWtaumv_0c == 1) | (label_H_WqqWtaumv_1c == 1) )
        fj_H_VV_hadtauvqq: ( (label_H_WqqWtauhv_0c == 0) | (label_H_WqqWtauhv_1c == 1) )

        fj_H_VV_taunuqq: ( (fj_H_VV_leptauelvqq == 1) | (fj_H_VV_leptaumuvqq == 1) | (fj_H_VV_hadtauvqq == 1) )

        label_Top_bWqq_0c: ( (fj_Top_2q==1) & (fj_nprongs == 2)  & (fj_Top_bmerged==1) & (fj_ncquarks==0) ) 
        label_Top_bWqq_1c: ( (fj_Top_2q==1) & (fj_nprongs == 2) & (fj_Top_bmerged==1) & (fj_ncquarks==1) ) 
        label_Top_bWq_0c: ( (fj_Top_2q==1) & (fj_nprongs == 1) & (fj_Top_bmerged==1) & (fj_ncquarks==0) )
        label_Top_bWq_1c: ( (fj_Top_2q==1) & fj_nprongs == 1) & (fj_Top_bmerged==1) & (fj_ncquarks==1) )
        label_Top_bWev: ( (fj_Top_elenu==1) & (fj_Top_bmerged==1) )
        label_Top_bWmv: ( (fj_Top_munu==1) & (fj_Top_bmerged==1) )
        label_Top_bWtauhv: ( (fj_Top_hadtauvqq==1) & (fj_Top_bmerged==1) )
        label_Top_bWtauev: ( (fj_Top_leptauelvnu==1) & (fj_Top_bmerged==1) )
        label_Top_bWtaumv: ( (fj_Top_leptaumuvnu==1) & (fj_Top_bmerged==1) )
        
        fj_Top_taunu: ( (fj_Top_leptauelvnu == 1) | (fj_Top_leptaumuvnu == 1) | (fj_Top_hadtauvqq == 1) )
        fj_Top_2q_1q: ( (fj_Top_2q == 1) & (fj_nprongs == 1) ) # with or without b merged
        fj_Top_2q_2q: ( (fj_Top_2q == 1) & (fj_nprongs == 2) ) # with or without b merged
        fj_ttbar_label: ( (fj_Top_2q==1) | (fj_Top_elenu==1) | (fj_Top_munu==1) | (fj_Top_taunu==1) )

        fj_Vqq_1q: ((fj_V_2q==1) & (fj_nprongs==1))
        fj_Vqq_2q: ((fj_V_2q==1) & (fj_nprongs==2))
        label_Wqq_jets_1c: ((fj_V_2q==1) & (fj_ncquarks==0) )
        label_Wqq_jets_0c: ((fj_V_2q==1) & (fj_ncquarks==1) ) 
        fj_wjets_label: ((fj_V_2q==1) | (fj_V_elenu==1) | (fj_V_munu==1) | (fj_V_taunu==1))

        Note: for two-prong decays make sure you require two prongs, e.g.:
        fj_H_gg_2p: ((fj_H_gg==1) & (fj_nprongs==2))
        """

        self.skim_vars = {
            "Event": {
                "event": "event",
            },
            "FatJet": {
                **P4,
                "msoftdrop": "msoftdrop",
            },
            "SubJet": {
                **P4,
            },
            "GenPart": [
                "fj_genjetmsd",
                "fj_genjetmass",
                "fj_genRes_pt",
                "fj_genRes_eta",
                "fj_genRes_phi",
                "fj_genRes_mass",
                "fj_genX_pt",
                "fj_genX_eta",
                "fj_genX_phi",
                "fj_genX_mass",
                "fj_nprongs",
                "fj_ncquarks",
                "fj_lepinprongs",
                "fj_H_VV_4q",
                "fj_H_VV_elenuqq",
                "fj_H_VV_munuqq",
                "fj_H_VV_leptauelvqq",
                "fj_H_VV_leptaumuvqq",
                "fj_H_VV_hadtauvqq",
                "fj_H_VV_unmatched",
                "fj_genV_pt",
                "fj_genV_eta",
                "fj_genV_phi",
                "fj_genV_mass",
                "fj_dR_V",
                "fj_dR_Vstar",
                "fj_dR_V_Vstar",
                "fj_genVstar_pt",
                "fj_genVstar_eta",
                "fj_genVstar_phi",
                "fj_genVstar_mass",
                "fj_H_gg",
                "fj_H_qq",
                "fj_H_bb",
                "fj_H_cc",
                "fj_QCDb",
                "fj_QCDbb",
                "fj_QCDc",
                "fj_QCDcc",
                "fj_QCDothers",
                "fj_V_2q",
                "fj_V_elenu",
                "fj_V_munu",
                "fj_V_taunu",
                "fj_Top_bmerged",
                "fj_Top_2q",
                "fj_Top_elenu",
                "fj_Top_munu",
                "fj_Top_hadtauvqq",
                "fj_Top_leptauelvnu",
                "fj_Top_leptaumuvnu",
            ],
            # formatted to match weaver's preprocess.json
            "MET": {
                "met_features": {
                    "var_names": [
                        "met_relpt",
                        "met_relphi",
                    ],
                },
                "met_points": {"var_length": 1},
            },
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
                        "pfcand_btagEtaRel",
                        "pfcand_btagPtRatio",
                        "pfcand_btagPParRatio",
                        "pfcand_btagSip3dVal",
                        "pfcand_btagSip3dSig",
                        "pfcand_btagJetDistVal",
                    ],
                    # number of pf cands to select or pad up to
                    "var_length": 128,
                },
                "pf_vectors": {
                    "var_names": [
                        "pfcand_px",
                        "pfcand_py",
                        "pfcand_pz",
                        "pfcand_energy",
                    ],
                },
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
                    # number of svs to select or pad up to
                    "var_length": 10,
                },
                "sv_vectors": {
                    "var_names": [
                        "sv_px",
                        "sv_py",
                        "sv_pz",
                        "sv_energy",
                    ],
                },
            },
            "Lep": {
                "fj_features": {
                    "fj_lep_dR",
                    "fj_lep_pt",
                    "fj_lep_iso",
                    "fj_lep_miniiso",
                },
            },
        }

        self.tagger_resources_path = (
            str(pathlib.Path(__file__).parent.resolve()) + "/tagger_resources/"
        )

        self.ak15 = "AK15" in self.label
        self.fatjet_label = "FatJetAK15" if self.ak15 else "FatJet"
        self.subjet_label = "FatJetAK15SubJet" if self.ak15 else "SubJet"
        self.pfcands_label = "FatJetAK15PFCands" if self.ak15 else "FatJetPFCands"
        self.svs_label = "JetSVsAK15" if self.ak15 else "FatJetSVs"

        # self.met_label = "MET"

        self.num_jets = num_jets
        self.num_subjets = 2
        self.match_dR = 1.0  # max dR for object-jet-matching

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

        isMC = hasattr(events, "genWeight")

        jet_vars = []

        for jet_idx in range(self.num_jets):
            # objects
            fatjets = ak.pad_none(events[self.fatjet_label], self.num_jets, axis=1)[:, jet_idx]
            subjets = events[self.subjet_label]

            # selection
            selection = PackedSelection()
            # preselection_cut = (fatjets.pt > 200) * (fatjets.pt < 1500)
            # preselection_cut = fatjets.pt > 200
            # add_selection_no_cutflow("preselection", preselection_cut, selection)

            # variables
            FatJetVars = {
                f"fj_{key}": ak.fill_none(fatjets[var], -99999)
                for (var, key) in self.skim_vars["FatJet"].items()
            }

            # select subjets within self.match_dR of fatjet
            matched_subjets = ak.pad_none(
                subjets[fatjets.delta_r(subjets) < self.match_dR],
                self.num_subjets,
                axis=1,
                clip=True,
            )

            SubJetVars = {
                f"fj_subjet{i + 1}_{key}": ak.fill_none(matched_subjets[:, i][var], -99999)
                for i, (var, key) in itertools.product(
                    range(self.num_subjets), self.skim_vars["SubJet"].items()
                )
            }

            # standard PN tagger scores
            if self.ak15:
                FatJetVars["fj_PN_XbbvsQCD"] = fatjets.ParticleNetMD_probXbb / (
                    fatjets.ParticleNetMD_probQCD + fatjets.ParticleNetMD_probXbb
                )

                if "ParticleNet_probHqqqq" in fatjets.fields:
                    FatJetVars["fj_PN_H4qvsQCD"] = fatjets.ParticleNet_probHqqqq / (
                        fatjets.ParticleNet_probHqqqq
                        + fatjets.ParticleNet_probQCDb
                        + fatjets.ParticleNet_probQCDbb
                        + fatjets.ParticleNet_probQCDc
                        + fatjets.ParticleNet_probQCDcc
                        + fatjets.ParticleNet_probQCDothers
                    )
            else:
                FatJetVars["fj_PN_XbbvsQCD"] = fatjets.particleNetMD_Xbb / (
                    fatjets.particleNetMD_QCD + fatjets.particleNetMD_Xbb
                )

                if "particleNet_H4qvsQCD" in fatjets.fields:
                    FatJetVars["fj_PN_H4qvsQCD"] = fatjets.particleNet_H4qvsQCD

            PFSVVars = {
                **get_pfcands_features(
                    self.skim_vars["PFSV"],
                    events,
                    jet_idx=jet_idx,
                    fatjet_label=self.fatjet_label,
                    pfcands_label=self.pfcands_label,
                    normalize=False,
                ),
                **get_svs_features(
                    self.skim_vars["PFSV"],
                    events,
                    jet_idx=jet_idx,
                    fatjet_label=self.fatjet_label,
                    svs_label=self.svs_label,
                    normalize=False,
                ),
            }

            LepVars = {
                **get_lep_features(
                    self.skim_vars["Lep"],
                    events,
                    jet_idx,
                    self.fatjet_label,
                    "Muon",
                    "Electron",
                    normalize=False,
                ),
            }

            METVars = {
                **get_met_features(
                    self.skim_vars["MET"],
                    events,
                    jet_idx,
                    self.fatjet_label,
                    "MET",
                    normalize=False,
                ),
            }

            if isMC:
                genparts = events.GenPart
                matched_mask, genVars = tagger_gen_matching(
                    events,
                    genparts,
                    fatjets,
                    self.skim_vars["GenPart"],
                    label=self.label,
                    match_dR=self.match_dR,
                )
                add_selection_no_cutflow("gen_match", matched_mask, selection)
            else:
                genVars = {"fj_isData": ak.values_astype((fatjets.pt > 200), np.int32)}

            if np.sum(selection.all(*selection.names)) == 0:
                continue

            skimmed_vars = {**FatJetVars, **SubJetVars, **genVars, **PFSVVars, **METVars}

            # apply selections
            skimmed_vars = {
                key: np.squeeze(np.array(value[selection.all(*selection.names)]))
                for (key, value) in skimmed_vars.items()
            }

            jet_vars.append(skimmed_vars)

            print(f"Jet {jet_idx + 1}: {time.time() - start:.1f}s")

        pnet_vars = runInferenceTriton(
            self.tagger_resources_path,
            events[selection.all(*selection.names)],
            num_jets=self.num_jets,
            ak15=False,
        )

        for jet_idx in range(self.num_jets):
            pnet_vars_jet = {**{key: value[:, jet_idx] for (key, value) in pnet_vars.items()}}
            # print(jet_idx,jet_vars[jet_idx]["fj_pt"],pnet_vars_jet)
            jet_vars[jet_idx] = {**jet_vars[jet_idx], **pnet_vars_jet}

        if len(jet_vars) > 1:
            # for debugging
            # for var in ["pfcand_energy","pfcand_px"]:
            #    for jet_var in jet_vars:
            #        print(var,jet_var[var])
            # print(len(jet_var[var]))

            # stack each set of jets
            jet_vars = {
                var: np.concatenate([jet_var[var] for jet_var in jet_vars], axis=0)
                for var in jet_vars[0]
            }

            # for var in jet_vars:
            #    if "FatJetParTMD_" in var or "fj_pt" in var:
            #        print(var, jet_vars[var])

            # some of the pfcand_dz/pfcand_dxy values are missing in v2.3 PFNano..
            # some of the SV info is missing in v2.3 PFNano..
            # test_vars = [
            #     "pfcand_dz",
            #     "pfcand_dzsig",
            #     "pfcand_dxy",
            #     "pfcand_dxysig",
            #     "pfcand_normchi2",
            #     "pfcand_btagEtaRel",
            #     "pfcand_btagPtRatio",
            #     "pfcand_btagPParRatio",
            #     "pfcand_btagSip3dVal",
            #     "pfcand_btagSip3dSig",
            #     "pfcand_btagJetDistVal"
            #     "sv_pt_log",
            #     "sv_mass",
            #     "sv_etarel",
            #     "sv_phirel",
            #     "sv_abseta",
            #     "sv_ntracks",
            #     "sv_normchi2",
            #     "sv_dxy",
            #     "sv_dxysig",
            #     "sv_d3d",
            #     "sv_d3dsig",
            #     "sv_costhetasvpv",
            #     "sv_px",
            #     "sv_py",
            #     "sv_pz",
            #     "sv_energy",
            # ]
            # for var in test_vars:
            #     print(var)
            #     print(jet_vars[var])
            #     # print(jet_vars[var][1])
            #     # print(jet_vars[var][3])
            #     print("\n")

        elif len(jet_vars) == 1:
            print("One jet passed selection")
            jet_vars = jet_vars[0]
        else:
            print("No jets passed selection")
            return {}

        print(f"Stack: {time.time() - start:.1f}s")

        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_")

        # convert output to pandas
        df = self.to_pandas(jet_vars)

        print(f"convert: {time.time() - start:.1f}s")

        # save to parquet
        self.dump_table(df, fname + ".parquet")

        print(f"dumped: {time.time() - start:.1f}s")

        return {}

    def postprocess(self, accumulator):
        pass
