import os
import numpy as np
import awkward as ak
import pandas as pd

from coffea.processor import ProcessorABC, dict_accumulator
from coffea.analysis_tools import PackedSelection

from .utils import pad_val, add_selection

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
    def __init__(self):
        self.skim_vars = {
            "FatJet": {
                **P4,
                "msoftdrop": "msoftdrop",
            }
        }
        self.fatjet = "FatJetAK15"
        self.pfcands = "FatJetAK15PFCands"
        self.num_jets = 2

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
        # objects
        fatjets = events[self.fatjet]
        pfcands = events[self.pfcands]

        # selection
        selection = PackedSelection()
        preselection_cut =  np.prod(
            pad_val(
                (fatjets.pt > 200)
                * (fatjets.pt < 1500),
                self.num_jets,
                False,
                axis=1,
            ),
            axis=1,
        ) > 0
        add_selection("preselection", preselection_cut, selection, {}, True, None)

        # variables
        FatJetVars = {
            f"fj_{key}": pad_val(fatjets[var], self.num_jets, -99999, axis=1)
            for (var, key) in self.skim_vars["FatJet"].items()
        }

        for jet_idx in range(0,2):
            jet_col = ak.pad_none(fatjets, self.num_jets, axis=1)[:, jet_idx : jet_idx+1]
            idx_sel = pfcands.jetIdx == 0
            jet_pfcands = events.PFCands[pfcands.pFCandsIdx[idx_sel]]
            fjets = ak.pad_none(jet_col, 1, axis=1)
            eta_sign = ak.values_astype(jet_pfcands.eta > 0, int) * 2 - 1
            PFVars = {
                "pfcand_etarel": (eta_sign * (jet_pfcands.eta - ak.flatten(fjets.eta))),
            }

        #TaggerVars = {
        #    **get_pfcands_features(tagger_vars, events, fatjets, pfcands),
        #    **get_svs_features(tagger_vars, events, fatjets, pfcands),
        #}


        # output
        skimmed_events = {}
        skimmed_events = {**skimmed_events, **FatJetVars}
        skimmed_events = {key: value[selection.all(*selection.names)] for (key, value) in skimmed_events.items()}
        df = self.ak_to_pandas(skimmed_events)
        print(df)
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
        self.dump_table(df, fname)

        return {}

    def postprocess(self,accumulator):
        pass
