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
    - Finish signal gen vars
    - Background gen labels
    - Is saving as parquet the way to go? Could also potentially save to root files?
        - If yes to parquet, need to update ak_to_pandas function
"""


class TaggerInputSkimmer(ProcessorABC):
    """
    Produces a flat training ntuple from PFNano.
    """

    def __init__(self, num_jets=2):
        self.skim_vars = {
            "FatJet": {
                **P4,
                "msoftdrop": "msoftdrop",
            }
        }
        self.fatjet_label = "FatJetAK15"
        self.pfcands_label = "FatJetAK15PFCands"
        self.svs_label = "JetSVsAK15"
        self.num_jets = num_jets

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
            output[field] = ak.to_numpy(ak.flatten(output_collection[field]))
        return output

    def process(self, events: ak.Array):
        jet_vars = []

        for jet_idx in range(self.num_jets):
            # objects
            fatjets = ak.pad_none(events[self.fatjet_label], 2, axis=1)[:, jet_idx]
            # pfcands = events[self.pfcands_label]

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

            ############ Gen Matching ##############

            B_PDGID = 5
            Z_PDGID = 23
            W_PDGID = 24
            HIGGS_PDGID = 25

            match_dR = 1.0
            jet_dR = 1.5

            flags = ["fromHardProcess", "isLastCopy"]
            higgs = events.GenPart[
                (abs(events.GenPart.pdgId) == HIGGS_PDGID) * events.GenPart.hasFlags(flags)
            ]

            jet = ak.pad_none(fatjets, 2, axis=1)[:, jet_idx]
            jet_dr_higgs = jet.delta_r(higgs)
            # TODO: this indexing is not working!
            matched_higgs = higgs[ak.argmin(jet_dr_higgs, axis=1)]
            add_selection_no_cutflow(
                "higgs_match", ak.any(jet_dr_higgs < match_dR, axis=1), selection
            )

            genResVars = {
                f"fj_genRes_{key}": ak.fill_none(matched_higgs[var], -99999)
                for (var, key) in P4.items()
            }

            higgs_children = matched_higgs.children
            higgs_children_pdgId = abs(higgs_children.pdgId[:, :, 0])
            is_VV = (higgs_children_pdgId == W_PDGID) + (higgs_children_pdgId == Z_PDGID)
            FatJetVars["fj_H_WW"] = is_VV

            FatJetVars = {**FatJetVars, **genResVars}

            ############ Gen Matching ##############

            skimmed_vars = {**FatJetVars, **PFSVVars}
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
        df = self.ak_to_pandas(jet_vars)  # need to modify to deal with pfcands
        print(df)
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
        self.dump_table(df, fname)

        return {}

    def postprocess(self, accumulator):
        pass
