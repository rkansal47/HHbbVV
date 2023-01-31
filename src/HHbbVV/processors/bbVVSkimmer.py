"""
Skimmer for bbVV analysis.
Author(s): Raghav Kansal
"""

import numpy as np
import awkward as ak
import pandas as pd

from coffea.processor import ProcessorABC, dict_accumulator
from coffea.analysis_tools import Weights, PackedSelection

import pathlib
import pickle
import gzip
import os

from typing import Dict
from collections import OrderedDict

from .GenSelection import gen_selection_HHbbVV, gen_selection_HH4V, gen_selection_HYbbVV
from .TaggerInference import runInferenceTriton
from .utils import pad_val, add_selection
from .corrections import add_pileup_weight, add_VJets_kFactors, get_jec_key, get_jec_jets


P4 = {
    "eta": "Eta",
    "phi": "Phi",
    "mass": "Mass",
    "pt": "Pt",
}

# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {
    "HTo2bYTo2W": gen_selection_HYbbVV,
    "GluGluToHHTobbVV_node_cHHH1": gen_selection_HHbbVV,
    "jhu_HHbbWW": gen_selection_HHbbVV,
    "GluGluToBulkGravitonToHHTo4W_JHUGen": gen_selection_HH4V,
    "GluGluToHHTo4V_node_cHHH1": gen_selection_HH4V,
    "GluGluHToWWTo4q_M-125": gen_selection_HH4V,
}


class bbVVSkimmer(ProcessorABC):
    """
    Skims nanoaod files, saving selected branches and events passing preselection cuts
    (and triggers for data), for preliminary cut-based analysis and BDT studies.

    Args:
        xsecs (dict, optional): sample cross sections,
          if sample not included no lumi and xsec will not be applied to weights
        save_ak15 (bool, optional): save ak15 jets as well, for HVV candidate
    """

    # TODO: Check if this is correct for JetHT
    LUMI = {  # in pb^-1
        "2016": 16830.0,
        "2016APV": 19500.0,
        "2017": 41480.0,
        "2018": 59830.0,
    }

    HLTs = {
        "2016": [
            "AK8DiPFJet250_200_TrimMass30_BTagCSV_p20",
            "AK8DiPFJet280_200_TrimMass30_BTagCSV_p20",
            #
            "AK8PFHT600_TrimR0p1PT0p03Mass50_BTagCSV_p20",
            "AK8PFHT700_TrimR0p1PT0p03Mass50",
            #
            "AK8PFJet360_TrimMass30",
            "AK8PFJet450",
            "PFJet450",
            #
            "PFHT800",
            "PFHT900",
            "PFHT1050",
            #
            "PFHT750_4JetPt50",
            "PFHT750_4JetPt70",
            "PFHT800_4JetPt50",
        ],
        "2017": [
            "PFJet450",
            "PFJet500",
            #
            "AK8PFJet400",
            "AK8PFJet450",
            "AK8PFJet500",
            #
            "AK8PFJet360_TrimMass30",
            "AK8PFJet380_TrimMass30",
            "AK8PFJet400_TrimMass30",
            #
            "AK8PFHT750_TrimMass50",
            "AK8PFHT800_TrimMass50",
            #
            "PFHT1050",
            #
            "AK8PFJet330_PFAK8BTagCSV_p17",
        ],
        "2018": [
            "PFJet500",
            #
            "AK8PFJet500",
            #
            "AK8PFJet360_TrimMass30",
            "AK8PFJet380_TrimMass30",
            "AK8PFJet400_TrimMass30",
            "AK8PFHT750_TrimMass50",
            "AK8PFHT800_TrimMass50",
            #
            "PFHT1050",
            #
            "HLT_AK8PFJet330_TrimMass30_PFAK8BTagCSV_p17_v",
        ],
    }

    # key is name in nano files, value will be the name in the skimmed output
    skim_vars = {
        "FatJet": {
            **P4,
            "msoftdrop": "Msd",
            "particleNetMD_QCD": "ParticleNetMD_QCD",
            "particleNetMD_Xbb": "ParticleNetMD_Xbb",
            # "particleNetMD_Xcc": "ParticleNetMD_Xcc",
            # "particleNetMD_Xqq": "ParticleNetMD_Xqq",
            "particleNet_H4qvsQCD": "ParticleNet_Th4q",
        },
        "GenHiggs": P4,
        "other": {"MET_pt": "MET_pt"},
    }

    preselection_cut_vals = {"pt": 250, "msd": 20}

    def __init__(self, xsecs={}, save_ak15=False):
        super(bbVVSkimmer, self).__init__()

        self.XSECS = xsecs  # in pb
        self.save_ak15 = save_ak15

        # for tagger model and preprocessing dict
        self.tagger_resources_path = (
            str(pathlib.Path(__file__).parent.resolve()) + "/tagger_resources/"
        )

        self._accumulator = dict_accumulator({})

    def to_pandas(self, events: Dict[str, np.array]):
        """
        Convert our dictionary of numpy arrays into a pandas data frame
        Uses multi-index columns for numpy arrays with >1 dimension
        (e.g. FatJet arrays with two columns)
        """
        return pd.concat(
            [pd.DataFrame(v.reshape(v.shape[0], -1)) for k, v in events.items()],
            axis=1,
            keys=list(events.keys()),
        )

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

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events: ak.Array):
        """Returns skimmed events which pass preselection cuts (and triggers if data) with the branches listed in ``self.skim_vars``"""

        print("processing")

        year = events.metadata["dataset"][:4]
        dataset = events.metadata["dataset"][5:]

        isData = "JetHT" in dataset
        signGenWeights = None if isData else np.sign(events["genWeight"])
        n_events = len(events) if isData else int(np.sum(signGenWeights))
        selection = PackedSelection()
        weights = Weights(len(events), storeIndividual=True)

        cutflow = OrderedDict()
        cutflow["all"] = n_events

        skimmed_events = {}

        # gen vars - saving HH, bb, VV, and 4q 4-vectors + Higgs children information
        for d in gen_selection_dict:
            if d in dataset:
                skimmed_events = {
                    **skimmed_events,
                    **gen_selection_dict[d](events, selection, cutflow, signGenWeights, P4),
                }

        # triggers
        # OR-ing HLT triggers
        if isData:
            HLT_triggered = np.any(
                np.array(
                    [
                        events.HLT[trigger]
                        for trigger in self.HLTs[year]
                        if trigger in events.HLT.fields
                    ]
                ),
                axis=0,
            )
            add_selection("trigger", HLT_triggered, selection, cutflow, isData, signGenWeights)

        # TODO: save variations (?)
        try:
            fatjets = get_jec_jets(events, year) if not isData else events.FatJet
        except:
            print("Couldn't load JECs")
            fatjets = events.FatJet

        num_jets = 2 if not dataset == "GluGluHToWWTo4q_M-125" else 1

        # pre-selection cuts
        preselection_cut = np.prod(
            pad_val(
                (events.FatJet.pt > self.preselection_cut_vals["pt"])
                * (events.FatJet.msoftdrop > self.preselection_cut_vals["msd"]),
                num_jets,
                False,
                axis=1,
            ),
            axis=1,
        )

        add_selection(
            "preselection",
            preselection_cut.astype(bool),
            selection,
            cutflow,
            isData,
            signGenWeights,
        )

        # TODO: trigger SFs

        # select vars

        ak8FatJetVars = {
            f"ak8FatJet{key}": pad_val(events.FatJet[var], 2, -99999, axis=1)
            for (var, key) in self.skim_vars["FatJet"].items()
        }
        otherVars = {
            key: events[var.split("_")[0]]["_".join(var.split("_")[1:])].to_numpy()
            for (var, key) in self.skim_vars["other"].items()
        }

        skimmed_events = {**skimmed_events, **ak8FatJetVars, **otherVars}

        # particlenet xbb vs qcd

        skimmed_events["ak8FatJetParticleNetMD_Txbb"] = pad_val(
            events.FatJet.particleNetMD_Xbb
            / (events.FatJet.particleNetMD_QCD + events.FatJet.particleNetMD_Xbb),
            2,
            -1,
            axis=1,
        )

        # calc weights
        if isData:
            skimmed_events["weight"] = np.ones(n_events)
        else:
            skimmed_events["genWeight"] = events.genWeight.to_numpy()
            add_pileup_weight(weights, year, events.Pileup.nPU.to_numpy())
            # add_VJets_kFactors(weights, events.GenPart, dataset)

            # TODO: theory uncertainties
            # TODO: trigger SFs here once calculated properly

            # this still needs to be normalized with the acceptance of the pre-selection (done in post processing)
            if dataset in self.XSECS:
                skimmed_events["weight"] = (
                    np.sign(skimmed_events["genWeight"])
                    * self.XSECS[dataset]
                    * self.LUMI[year]
                    * weights.weight()
                )
            else:
                skimmed_events["weight"] = np.sign(skimmed_events["genWeight"])

            # TODO: can add uncertainties with weights._modifiers?

        # apply selections

        skimmed_events = {
            key: value[selection.all(*selection.names)] for (key, value) in skimmed_events.items()
        }

        # apply HWW4q tagger
        pnet_vars = runInferenceTriton(
            self.tagger_resources_path,
            events[selection.all(*selection.names)],
            ak15=False,
            all_outputs=False,
        )

        # pnet_vars = {}

        skimmed_events = {
            **skimmed_events,
            **{key: value for (key, value) in pnet_vars.items()},
        }

        if len(skimmed_events["weight"]):
            df = self.to_pandas(skimmed_events)
            fname = (
                events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
            )
            self.dump_table(df, fname)

        return {year: {dataset: {"nevents": n_events, "cutflow": cutflow}}}

    def postprocess(self, accumulator):
        return accumulator
