"""
Skimmer for H(bb) analysis.
Author(s): Raghav Kansal, Cristina Suarez
"""

import numpy as np
import awkward as ak
import pandas as pd

from coffea import processor
from coffea.analysis_tools import Weights, PackedSelection
import vector

import pathlib
import pickle
import gzip
import os

from typing import Dict
from collections import OrderedDict

from .GenSelection import gen_selection_Hqq

# from .TaggerInference import runInferenceTriton

from .utils import pad_val, add_selection, concatenate_dicts, select_dicts, P4
from .common import LUMI
from . import common

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

gen_selection_dict = {
    "GluGluHToBB_M-125_TuneCP5_13p6TeV_powheg-pythia8": gen_selection_Hqq,
}

class HbbSkimmer(processor.ProcessorABC):
    """
    Skims nanoaod files, saving selected branches and events passing preselection cuts.
    """

    def __init__(self):

        self.skim_vars = {
            "FatJet": {
                **P4,
                "msoftdrop": "Msd",
                "particleNetMD_QCD": "ParticleNetMD_QCD",
                "particleNetMD_Xbb": "ParticleNetMD_Xbb",
                "particleNet_mass": "ParticleNetMass",
                "Txbb": "ParticleNetMD_Txbb",
            },
            "GenHiggs": P4,
        }
        
        self.preselection = {
            "pt": 250.0,
        }
        
        # run inference
        #self._inference = inference

        # for tagger model and preprocessing dict
        self.tagger_resources_path = (
            str(pathlib.Path(__file__).parent.resolve()) + "/tagger_resources/"
        )

        self._accumulator = processor.dict_accumulator({})


    def to_pandas(self, events: Dict[str, np.array]):
        """
        Convert our dictionary of numpy arrays into a pandas data frame
        Uses multi-index columns for numpy arrays with >1 dimension
        (e.g. FatJet arrays with two columns)
        """
        return pd.concat(
            [pd.DataFrame(v) for k, v in events.items()],
            axis=1,
            keys=list(events.keys()),
        )

    def dump_table(self, pddf: pd.DataFrame, fname: str, odir_str: str = None) -> None:
        """
        Saves pandas dataframe events to './outparquet'
        """
        import pyarrow.parquet as pq
        import pyarrow as pa

        local_dir = os.path.abspath(os.path.join(".", "outparquet"))
        if odir_str:
            local_dir += odir_str
        os.system(f"mkdir -p {local_dir}")

        # need to write with pyarrow as pd.to_parquet doesn't support different types in
        # multi-index column names
        table = pa.Table.from_pandas(pddf)
        pq.write_table(table, f"{local_dir}/{fname}")

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events: ak.Array):
        """Runs event processor for different types of jets"""

        year = events.metadata["dataset"].split("_")[0]
        year_nosuffix = year.replace("APV", "")
        dataset = "_".join(events.metadata["dataset"].split("_")[1:])

        isData = "JetHT" in dataset
        isQCD = "QCD" in dataset
        isSignal = (
            "HTobb" in dataset
        )

        if not isData:
            gen_weights = events["genWeight"].to_numpy()
        else:
            gen_weights = None

        n_events = len(events) if isData else np.sum(gen_weights)
        selection = PackedSelection()
        weights = Weights(len(events), storeIndividual=True)

        cutflow = OrderedDict()
        cutflow["all"] = n_events

        selection_args = (selection, cutflow, isData, gen_weights)

        skimmed_events = {}

        #########################
        # Save / derive variables
        #########################
        
        # FatJet variables
        fatjets = events.FatJet

        # note this changed in later versions of nano
        # particleNet_XbbVsQCD 
        # https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/jetsAK8_cff.py#L123C9-L123C29
        fatjets["Txbb"] = fatjets.particleNetMD_Xbb / (
            fatjets.particleNetMD_QCD + fatjets.particleNetMD_Xbb
        )

        goodfatjets = fatjets[
            (fatjets.pt > 200)
            & (abs(fatjets.eta) < 2.4)
            & fatjets.isTight
        ]

        candidatefatjet = ak.firsts(goodfatjets[ak.argmax(goodfatjets.Txbb, axis=1, keepdims=True)])

        fatjetVars = {
            f"ak8FatJet{key}": candidatefatjet[var]
            for (var, key) in self.skim_vars["FatJet"].items()
        }

        skimmed_events = {**skimmed_events, **fatjetVars}

        # gen variables
        for d in gen_selection_dict:
            if d in dataset:
                vars_dict = gen_selection_dict[d](
                    events, candidatefatjet, selection, cutflow, gen_weights, P4
                )
                skimmed_events = {**skimmed_events, **vars_dict}

        ######################
        # Selection
        ######################
        # OR-ing HLT triggers
        if isData:
            for trigger in HLTs[year_nosuffix]:
                if trigger not in events.HLT.fields:
                    logger.warning(f"Missing HLT {trigger}!")

            HLT_triggered = np.any(
                np.array(
                    [
                        events.HLT[trigger]
                        for trigger in HLTs[year_nosuffix]
                        if trigger in events.HLT.fields
                    ]
                ),
                axis=0,
            )
            add_selection("trigger", HLT_triggered, *selection_args)
            
        add_selection("ak8_pt", candidatefatjet.pt > self.preselection["pt"], *selection_args)

        ######################
        # Weights
        ######################

        if isData:
            skimmed_events["weight"] = np.ones(n_events)
        else:
            weights.add("genweight", gen_weights)
            skimmed_events["weight"] = weights.weight()

        # reshape and apply selections
        sel_all = selection.all(*selection.names)

        skimmed_events = {
            key: value[sel_all]
            for (key, value) in skimmed_events.items()
        }

        # convert to pandas
        df = self.to_pandas(skimmed_events)
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
        self.dump_table(df, fname)

        return {year: {dataset: {"nevents": n_events, "cutflow": cutflow}}}

    def postprocess(self, accumulator):
        return accumulator

