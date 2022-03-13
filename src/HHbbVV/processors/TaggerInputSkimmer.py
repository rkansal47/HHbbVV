import numpy as np
import awkward as ak
import pandas as pd

from coffea.processor import ProcessorABC, dict_accumulator
from coffea.analysis_tools import PackedSelection

from .utils import pad_val, add_selection_no_cutflow
from .TaggerInference import get_pfcands_features, get_svs_features

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

def gen_matching(genparts,fatjet,genlabels,label="H_VV",match_dR=1.0,jet_dR=1.5):
    B_PDGID = 5
    Z_PDGID = 23
    W_PDGID = 24
    HIGGS_PDGID = 25
    ELE_PDGID = 11
    MU_PDGID = 13
    TAU_PDGID = 15
    B_PDGID = 5

    matched_mask = np.ones(len(genparts), dtype='bool')
    genVars = {}

    def get_pid_mask(genpart,pdgids,ax=2,byall=True):
        pdgid = abs(genpart.pdgId)
        mask = (pdgid == pdgids[0])
        for i,pid in enumerate(pdgids):
            if i==0: continue
            mask = mask | (pdgid == pid)
        if byall:
            return ak.all(mask,axis=ax)
        else:
            return mask

    def to_label(array):
        return ak.values_astype(array, np.int32)
            
    if "H_" in label:
        higgs = genparts[
            (abs(genparts.pdgId) == HIGGS_PDGID) * genparts.hasFlags(["fromHardProcess", "isLastCopy"])
        ]
        matched_higgs = higgs[ak.argmin(fatjet.delta_r(higgs), axis=1,keepdims=True)]
        matched_mask = ak.any(fatjet.delta_r(matched_higgs) < match_dR, axis=1)

        genVars = {
            f"fj_genRes_{key}": ak.fill_none(matched_higgs[var], -99999)
            for (var, key) in P4.items()
        }

        if "VV" in label:
            is_decay = get_pid_mask(matched_higgs.children, [W_PDGID,Z_PDGID])
            sort_by_mass = matched_higgs[is_decay].children.mass
            v_star = ak.firsts(matched_higgs[is_decay].children[ak.argmin(sort_by_mass, axis=2, keepdims=True)])
            v = ak.firsts(matched_higgs[is_decay].children[ak.argmax(sort_by_mass, axis=2, keepdims=True)])

            daughter_mask = get_pid_mask(genparts.distinctParent, [W_PDGID,Z_PDGID], 1, byall=False)
            daughters = genparts[daughter_mask & genparts.hasFlags(['fromHardProcess', 'isLastCopy'])]
            nprongs = ak.sum(fatjet.delta_r(daughters) < jet_dR, axis=1)
            decay = (
                (ak.sum(daughters[abs(daughters.pdgId) <= B_PDGID].pt>0,axis=1)==2)*1 # 2quarks*1
                + (ak.sum(daughters[abs(daughters.pdgId) == ELE_PDGID].pt>0,axis=1)==1)*3 # 1electron*3
                + (ak.sum(daughters[abs(daughters.pdgId) == MU_PDGID].pt>0,axis=1)==1)*5 # 1muon*5
                + (ak.sum(daughters[abs(daughters.pdgId) == TAU_PDGID].pt>0,axis=1)==1)*7 # 1tau*7
                + (ak.sum(daughters[abs(daughters.pdgId) <= B_PDGID].pt>0,axis=1)==4)*11 # 4quarks*1
            )
            is_hVV_4q = (decay == 11)

            matched_vs = ak.any(fatjet.delta_r(v) < jet_dR, axis=1) & ak.any(fatjet.delta_r(v_star) < jet_dR, axis=1)
            matched_mask = matched_mask & matched_vs

            genVVars = {
                f"fj_genV_{key}": ak.fill_none(v[var], -99999)
                for (var, key) in P4.items()
            }
            genLabelVars = {
                "fj_nprongs": nprongs,
                "fj_H_VV_4q": to_label(decay == 11),
                "fj_H_VV_elenuqq": to_label(decay == 4),
                "fj_H_VV_munuqq": to_label(decay == 6),
                "fj_H_VV_taunuqq": to_label(decay == 8),
                "fj_H_VV_unmatched": to_label(~matched_mask),
            }
            genVars = {**genVVars,**genLabelVars,**genVars}

    elif "QCD" in label:
        partons = genparts[get_pid_mask(genparts,[21,1,2,3,4,5])]
        matched_mask = ak.any(fatjet.delta_r(partons) < match_dR, axis=1)
        btoleptons = genparts[get_pid_mask(genparts,[511,521,523]) & get_pid_mask(genparts.children, [11,13])]
        matched_b = ak.any(fatjet.delta_r(btoleptons) < 0.5, axis=1)
        genLabelVars = {
            "fj_isQCDb": to_label(matched_mask & (fatjet.nBHadrons == 1) & ~matched_b),
            "fj_isQCDbb": to_label(matched_mask & (fatjet.nBHadrons > 1) & ~matched_b),
            "fj_isQCDc": to_label(matched_mask & (fatjet.nCHadrons == 1)),
            "fj_isQCDcc": to_label(matched_mask & (fatjet.nCHadrons > 1)),
            "fj_isQCDlep": to_label(matched_mask & matched_b),
        }
        genLabelVars["fj_isQCDothers"] = matched_mask & (fatjet.nBHadrons == 0) & (fatjet.nBHadrons == 0) & ~matched_b

        genVars = {**genLabelVars}

    gen_vars = {
        key: genVars[key] if key in genVars.keys() else np.zeros(len(genparts))
        for key in genlabels
    }

    return matched_mask,gen_vars

class TaggerInputSkimmer(ProcessorABC):
    """
    Produces a flat training ntuple from PFNano.
    """

    def __init__(self, label="H_VV", num_jets=2):
        self.skim_vars = {
            "FatJet": {
                **P4,
                "msoftdrop": "msoftdrop",
            },
            "GenPart": {
                "fj_nprongs"
                "fj_H_VV_4q",
                "fj_H_VV_elenuqq",
                "fj_H_VV_munuqq",
                "fj_H_VV_taunuqq",
                "fj_H_VV_unmatched",
                "fj_genV_pt",
                "fj_genV_eta",
                "fj_genV_phi",
                "fj_genV_mass",
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

    def ak_to_pandas(self, output_collection: ak.Array) -> pd.DataFrame:
        output = pd.DataFrame()
        for field in ak.fields(output_collection):
            output[field] = ak.to_numpy(ak.flatten(output_collection[field],axis=None))
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

            matched_mask,genVars = gen_matching(genparts,fatjets,self.skim_vars["GenPart"],label="H_VV")
            add_selection_no_cutflow(
                "gen_match", matched_mask, selection
            )

            skimmed_vars = {**FatJetVars, **genVars, **PFSVVars}
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
