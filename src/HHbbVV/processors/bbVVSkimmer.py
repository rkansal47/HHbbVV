"""
Skimmer for bbVV analysis.
Author(s): Raghav Kansal
"""

import numpy as np
import awkward as ak
import pandas as pd

from coffea.processor import ProcessorABC, dict_accumulator
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.analysis_tools import PackedSelection

import pathlib
import pickle
import gzip
import os

from typing import Dict

from .TaggerInference import runInferenceTriton
from .utils import pad_val


class bbVVSkimmer(ProcessorABC):
    """
    Skims nanoaod files, saving selected branches and events passing preselection cuts
    (and triggers for data), for preliminary cut-based analysis and BDT studies.

    Args:
        xsecs (dict, optional): sample cross sections, if sample not included no lumi and xsec will not be applied to weights
        condor (bool, optional): using normal condor or not - if not, post processing will not divide by original total events
    """

    # TODO: do ak8, ak15 sorting for hybrid case

    def __init__(self, xsecs={}, output_location=None):
        super(bbVVSkimmer, self).__init__()

        self.LUMI = {"2017": 40000}  # in pb^-1
        self.XSECS = xsecs  # in pb
        self.output_location = output_location

        self.HLTs = {
            "2017": [
                "PFJet500",
                "AK8PFJet400",
                "AK8PFJet500",
                "AK8PFJet360_TrimMass30",
                "AK8PFJet380_TrimMass30",
                "AK8PFJet400_TrimMass30",
                "AK8PFHT750_TrimMass50",
                "AK8PFHT800_TrimMass50",
                "PFHT1050",
                # 'AK8PFJet330_PFAK8BTagCSV_p17'
            ]
        }

        # key is name in nano files, value will be the name in the skimmed output
        self.skim_vars = {
            "FatJet": {
                "eta": "Eta",
                "phi": "Phi",
                "mass": "Mass",
                "msoftdrop": "Msd",
                "pt": "Pt",
                "particleNetMD_QCD": "ParticleNetMD_QCD",
                "particleNetMD_Xbb": "ParticleNetMD_Xbb",
                "particleNetMD_Xcc": "ParticleNetMD_Xcc",
                "particleNetMD_Xqq": "ParticleNetMD_Xqq",
                "particleNet_H4qvsQCD": "ParticleNet_Th4q",
            },
            "FatJetAK15": {
                "eta": "Eta",
                "phi": "Phi",
                "mass": "Mass",
                "msoftdrop": "Msd",
                "pt": "Pt",
                "ParticleNetMD_probQCD": "ParticleNetMD_probQCD",
                "ParticleNetMD_probQCDb": "ParticleNetMD_probQCDb",
                "ParticleNetMD_probQCDbb": "ParticleNetMD_probQCDbb",
                "ParticleNetMD_probQCDc": "ParticleNetMD_probQCDc",
                "ParticleNetMD_probQCDcc": "ParticleNetMD_probQCDcc",
                "ParticleNetMD_probXbb": "ParticleNetMD_probXbb",
                "ParticleNetMD_probXcc": "ParticleNetMD_probXcc",
                "ParticleNetMD_probXqq": "ParticleNetMD_probXqq",
                "ParticleNet_probHbb": "ParticleNet_probHbb",
                "ParticleNet_probHcc": "ParticleNet_probHcc",
                "ParticleNet_probHqqqq": "ParticleNet_probHqqqq",
                "ParticleNet_probQCDb": "ParticleNet_probQCDb",
                "ParticleNet_probQCDbb": "ParticleNet_probQCDbb",
                "ParticleNet_probQCDc": "ParticleNet_probQCDc",
                "ParticleNet_probQCDcc": "ParticleNet_probQCDcc",
                "ParticleNet_probQCDothers": "ParticleNet_probQCDothers",
            },
            "GenHiggs": {
                "eta": "Eta",
                "phi": "Phi",
                "mass": "Mass",
                "pt": "Pt",
            },
            "other": {"MET_pt": "MET_pt"},
        }

        self.preselection_cut_vals = {"pt": 250, "msd": 20}

        # find corrections path using this file's path
        with gzip.open(
            str(pathlib.Path(__file__).parent.parent.resolve()) + "/corrections/corrections.pkl.gz",
            "rb",
        ) as filehandler:
            self.corrections = pickle.load(filehandler)

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

        # need to write with pyarrow as pd.to_parquet doesn't support different types in
        # multi-index column names
        table = pa.Table.from_pandas(pddf)
        pq.write_table(table, "test.parquet")
        local_dir = os.path.abspath(os.path.join(".", "outparquet"))
        os.system(f"mkdir -p {local_dir}")
        pddf.to_parquet(f"{local_dir}/{fname}")

    @property
    def accumulator(self):
        return self._accumulator

    def add_selection(self, name: str, sel: np.ndarray, selection: PackedSelection, cutflow: dict):
        """adds selection to PackedSelection object and the cutflow dictionary"""
        selection.add(name, sel)
        cutflow[name] = np.sum(selection.all(*selection.names))

    def process(self, events: ak.Array):
        """Returns skimmed events which pass preselection cuts (and triggers if data) and with the branches listed in self.skim_vars"""

        print("processing")

        year = events.metadata["dataset"][:4]
        dataset = events.metadata["dataset"][5:]

        n_events = len(events)
        isData = "JetHT" in dataset
        selection = PackedSelection()

        cutflow = {}
        cutflow["all"] = len(events)

        skimmed_events = {}

        # gen vars - saving HH, bb, VV, and 4q 4-vectors + Higgs children information
        if "HHToBBVVToBBQQQQ" in dataset:
            skimmed_events = {**skimmed_events, **self.gen_matching(events, selection, cutflow)}

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
            self.add_selection("trigger", HLT_triggered, selection, cutflow)

        # pre-selection cuts
        # ORing ak8 and ak15 cuts

        preselection_cut = np.logical_or(
            np.prod(
                pad_val(
                    (events.FatJet.pt > self.preselection_cut_vals["pt"])
                    * (events.FatJet.msoftdrop > self.preselection_cut_vals["msd"]),
                    2,
                    False,
                    axis=1,
                ),
                axis=1,
            ),
            np.prod(
                pad_val(
                    (events.FatJetAK15.pt > self.preselection_cut_vals["pt"])
                    * (events.FatJetAK15.msoftdrop > self.preselection_cut_vals["msd"]),
                    2,
                    False,
                    axis=1,
                ),
                axis=1,
            ),
        )
        self.add_selection("preselection", preselection_cut, selection, cutflow)

        # TODO: trigger SFs

        # select vars

        ak8FatJetVars = {
            f"ak8FatJet{key}": pad_val(events.FatJet[var], 2, -99999, axis=1)
            for (var, key) in self.skim_vars["FatJet"].items()
        }
        ak15FatJetVars = {
            f"ak15FatJet{key}": pad_val(events.FatJetAK15[var], 2, -99999, axis=1)
            for (var, key) in self.skim_vars["FatJetAK15"].items()
        }
        otherVars = {
            key: events[var.split("_")[0]]["_".join(var.split("_")[1:])].to_numpy()
            for (var, key) in self.skim_vars["other"].items()
        }

        skimmed_events = {**skimmed_events, **ak8FatJetVars, **ak15FatJetVars, **otherVars}

        # particlenet h4q vs qcd, xbb vs qcd

        skimmed_events["ak8FatJetParticleNetMD_Txbb"] = pad_val(
            events.FatJet.particleNetMD_Xbb
            / (events.FatJet.particleNetMD_QCD + events.FatJet.particleNetMD_Xbb),
            2,
            -1,
            axis=1,
        )
        skimmed_events["ak15FatJetParticleNetMD_Txbb"] = pad_val(
            events.FatJetAK15.ParticleNetMD_probXbb
            / (events.FatJetAK15.ParticleNetMD_probQCD + events.FatJetAK15.ParticleNetMD_probXbb),
            2,
            -1,
            axis=1,
        )
        skimmed_events["ak15FatJetParticleNet_Th4q"] = pad_val(
            events.FatJetAK15.ParticleNet_probHqqqq
            / (
                events.FatJetAK15.ParticleNet_probHqqqq
                + events.FatJetAK15.ParticleNet_probQCDb
                + events.FatJetAK15.ParticleNet_probQCDbb
                + events.FatJetAK15.ParticleNet_probQCDc
                + events.FatJetAK15.ParticleNet_probQCDcc
                + events.FatJetAK15.ParticleNet_probQCDothers
            ),
            2,
            -1,
            axis=1,
        )

        # calc weights
        skimmed_events["weight"] = np.ones(n_events)
        if not isData:
            skimmed_events["genWeight"] = events.genWeight.to_numpy()
            skimmed_events["pileupWeight"] = self.corrections[f"{year}_pileupweight"](
                events.Pileup.nPU
            ).to_numpy()

        # apply selections

        skimmed_events = {
            key: value[selection.all(*selection.names)] for (key, value) in skimmed_events.items()
        }

        # apply HWW4q tagger
        print("pre-inference")

        # pnet_vars = runInferenceTriton(
        #    self.tagger_resources_path, events[selection.all(*selection.names)]
        # )

        pnet_vars = {}

        print("post-inference")
        skimmed_events = {
            **skimmed_events,
            **{key: value for (key, value) in pnet_vars.items()},
        }

        df = self.to_pandas(skimmed_events)
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
        self.dump_table(df, fname)

        return {year: {dataset: {"nevents": n_events, "cutflow": cutflow}}}

    def gen_matching(self, events: NanoEventsArray, selection: PackedSelection, cutflow: dict):
        """Gets HH, bb, VV, and 4q 4-vectors + Higgs children information"""
        B_PDGID = 5
        Z_PDGID = 23
        W_PDGID = 24
        HIGGS_PDGID = 25

        # finding the two gen higgs
        flags = ["fromHardProcess", "isLastCopy"]
        higgs = events.GenPart[
            (abs(events.GenPart.pdgId) == HIGGS_PDGID) * events.GenPart.hasFlags(flags)
        ]
        GenHiggsVars = {
            f"GenHiggs{key}": higgs[var].to_numpy()
            for (var, key) in self.skim_vars["GenHiggs"].items()
        }  # saving 4-vector info

        higgs_children = higgs.children
        GenHiggsVars["GenHiggsChildren"] = abs(
            higgs_children.pdgId[:, :, 0]
        ).to_numpy()  # saving whether H->bb or H->VV

        # finding bb and VV children
        is_bb = abs(higgs_children.pdgId) == B_PDGID
        is_VV = (abs(higgs_children.pdgId) == W_PDGID) + (abs(higgs_children.pdgId) == Z_PDGID)

        # checking that there are 2 b's and 2 V's
        has_bb = ak.sum(ak.flatten(is_bb, axis=2), axis=1) == 2
        has_VV = ak.sum(ak.flatten(is_VV, axis=2), axis=1) == 2
        self.add_selection(
            "has_bbVV", has_bb * has_VV, selection, cutflow
        )  # only select events with 2 b's and 2 V's

        # saving bb and VV 4-vector info
        bb = ak.flatten(higgs_children[is_bb], axis=2)
        VV = ak.flatten(higgs_children[is_VV], axis=2)

        # have to pad to 2 because of some 4V events
        GenbbVars = {
            f"Genbb{key}": pad_val(bb[var], 2, -99999, axis=1)
            for (var, key) in self.skim_vars["GenHiggs"].items()
        }

        # selecting only up to the 2nd index because of some 4V events
        # (doesn't matter which two are selected since these events will be excluded anyway)
        GenVVVars = {
            f"GenVV{key}": VV[var][:, :2].to_numpy()
            for (var, key) in self.skim_vars["GenHiggs"].items()
        }

        # checking that each V has 2 q children
        VV_children = VV.children
        V_has_2q = ak.count(VV_children.pdgId, axis=2) == 2
        has_4q = ak.values_astype(ak.prod(V_has_2q, axis=1), np.bool)
        self.add_selection("has_4q", has_4q, selection, cutflow)

        # saving 4q 4-vector info
        Gen4qVars = {
            f"Gen4q{key}": pad_val(ak.fill_none(VV_children[var][:, :2], []), 2, -99999, axis=2)
            for (var, key) in self.skim_vars["GenHiggs"].items()
        }

        return {**GenHiggsVars, **GenbbVars, **GenVVVars, **Gen4qVars}

    def postprocess(self, accumulator):
        return accumulator
