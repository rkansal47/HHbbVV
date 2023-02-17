"""
Skimmer for bbVV analysis.
Author(s): Raghav Kansal
"""

import numpy as np
import awkward as ak
import pandas as pd

from coffea import processor
from coffea.analysis_tools import Weights, PackedSelection

import pathlib
import pickle
import gzip
import os

from typing import Dict
from collections import OrderedDict

from .GenSelection import gen_selection_HHbbVV, gen_selection_HH4V, gen_selection_HYbbVV
from .TaggerInference import runInferenceTriton
from .utils import pad_val, add_selection, concatenate_dicts
from .corrections import (
    add_pileup_weight,
    add_VJets_kFactors,
    add_ps_weight,
    add_pdf_weight,
    add_scalevar_7pt,
    get_jec_key,
    get_jec_jets,
    get_lund_SFs,
)


P4 = {
    "eta": "Eta",
    "phi": "Phi",
    "mass": "Mass",
    "pt": "Pt",
}

# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {
    "HTo2bYTo2W": gen_selection_HYbbVV,
    "GluGluToHHTobbVV_node_cHHH": gen_selection_HHbbVV,
    "jhu_HHbbWW": gen_selection_HHbbVV,
    "GluGluToBulkGravitonToHHTo4W_JHUGen": gen_selection_HH4V,
    "GluGluToHHTo4V_node_cHHH1": gen_selection_HH4V,
    "GluGluHToWWTo4q_M-125": gen_selection_HH4V,
}

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def update(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out


class bbVVSkimmer(processor.ProcessorABC):
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
            "particleNet_H4qvsQCD": "ParticleNet_Th4q",
        },
        "GenHiggs": P4,
        "other": {"MET_pt": "MET_pt"},
    }

    preselection_cut_vals = {"pt": 300, "msd": 50, "txbb": 0.8}

    def __init__(self, xsecs={}, save_ak15=False, save_systematics=True, inference=True):
        super(bbVVSkimmer, self).__init__()

        self.XSECS = xsecs  # in pb
        self.save_ak15 = save_ak15

        # save systematic variations
        self._systematics = save_systematics

        # run inference
        self._inference = inference

        # for tagger model and preprocessing dict
        self.tagger_resources_path = (
            str(pathlib.Path(__file__).parent.resolve()) + "/tagger_resources/"
        )

        self._accumulator = processor.dict_accumulator({})

        logger.info(
            f"Running skimmer with inference {self._inference} and systematics {self._systematics}"
        )

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

        year = events.metadata["dataset"][:4]
        year_nosuffix = year.replace("APV", "")
        dataset = events.metadata["dataset"][5:]
        isData = "JetHT" in dataset
        isQCD = "QCD" in dataset

        if isData or isQCD or (np.sum(ak.num(events.FatJet, axis=1)) < 1):
            return self.process_shift(events, None)
        else:
            try:
                fatjets = get_jec_jets(events, year)
            except:
                logger.warning("Couldn't load JECs - will proceed without variations")
                fatjets = events.FatJet
                self._systematics = False

            if not self._systematics:
                shifts = [({"FatJet": fatjets}, None)]
            else:
                # logger.debug(fatjets.fields)

                # naming conventions: https://gitlab.cern.ch/hh/naming-conventions
                # reduced set of uncertainties: https://docs.google.com/spreadsheets/d/1Feuj1n0MdotcPq19Mht7SUIgvkXkA4hiB0BxEuBShLw/edit#gid=1345121349
                shifts = [
                    ({"FatJet": fatjets}, None),
                    ({"FatJet": fatjets.JES_jes.up}, "JES_up"),
                    ({"FatJet": fatjets.JES_jes.down}, "JES_down"),
                    ({"FatJet": fatjets.JER.up}, "JER_up"),
                    ({"FatJet": fatjets.JER.down}, "JER_down"),
                ]

                # commenting these out until we derive the uncertainties from the regrouped files
                """
                shifts = [
                ({"FatJet": fatjets}, None),
                ({"FatJet": fatjets.JES_AbsoluteScale / JES_AbsoluteMPFBias / JES_Fragmentation / JES_PileUpDataMC / JES_PileUpPtRef / JES_RelativeFSR / JES_SinglePionECAL / JESSinglePionHCAL }, "JESUp_Abs"),
                ({"FatJet": }, "JESDown_Abs"),
                ({"FatJet": fatjets.JES_AbsoluteStat / JES_RelativeStatFSR / JES_TimePtEta }, f"JESUp_Abs_{year_nosuffix}"),
                ({"FatJet": }, f"JESDown_Abs_{year_nosuffix}"),
                ({"FatJet": fatjets.JES_PileUpPtBB / PileUpPtEC1 / RelativePtBB }, f"JESUp_BBEC1"),
                ({"FatJet": },  f"JESDown_BBEC1"),
                ({"FatJet": fatjets.JES_RelativeJEREC1 / JES_RelativePtEC1 / JES_RelativeStatEC }, f"JESUp_BBEC1_{year_nosuffix}"),
                ({"FatJet": }, f"JESDown_BBEC1_{year_nosuffix}"),
                ({"FatJet": fatjets.PileUpPtEC2.up}, "JESUp_EC2")
                ({"FatJet": fatjets.PileUpPtEC2.down}, "JESDown_EC2")
                ({"FatJet": RelativeJEREC2 / RelativePtEC2 }, f"JESUp_EC2_{year_nosuffix}"),
                ({"FatJet": }, f"JESDown_EC2_{year_nosuffix}"),
                ({"FatJet": fatjets.JES_FlavQCD.up}, "JESUp_FlavQCD"),
                ({"FatJet": fatjets.JES_FlavQCD.down}, "JESDown_FlavQCD"),
                ({"FatJet": PileUpPtHF /RelativeJERHF / RelativePtHF }, "JESUp_HF"),
                ({"FatJet": PileUpPtHF /RelativeJERHF / RelativePtHF }, "JESDown_HF"),
                ({"FatJet": fatjets.JES_RelativeStatHF.up}, f"JESUp_HF_{year_nosuffix}"),
                ({"FatJet": fatjets.JES_RelativeStatHF.down}, f"JESDown_HF_{year_nosuffix}"),
                ({"FatJet": fatjers.JES_RelativeBal.up}, f"JESUp_RelBal"),
                ({"FatJet": fatjers.JES_RelativeBal.down}, f"JESDown_RelBal"),
                ({"FatJet": fatjers.JES_RelativeSample.up}, f"JESUp_RelSample_{year_nosuffix}")
                ({"FatJet": fatjers.JES_RelativeSample.down}, f"JESDown_RelSample_{year_nosuffix}")
                ]
                """
        return processor.accumulate(
            self.process_shift(update(events, collections), name) for collections, name in shifts
        )

    def process_shift(self, events: ak.Array, shift_name: str = None):
        """Returns skimmed events which pass preselection cuts (and triggers if data) with the branches listed in ``self.skim_vars``"""
        logger.info(f"Procesing events with shift {shift_name}")

        year = events.metadata["dataset"][:4]
        dataset = events.metadata["dataset"][5:]

        isData = "JetHT" in dataset
        gen_weights = None
        if not isData:
            if "GluGluToHHTobbVV":
                gen_weights = np.sign(events["genWeight"])
            else:
                gen_weights = events["genWeight"].to_numpy()
        n_events = len(events) if isData else np.sum(gen_weights)
        selection = PackedSelection()
        weights = Weights(len(events), storeIndividual=True)

        cutflow = OrderedDict()
        cutflow["all"] = n_events

        num_jets = 2 if not dataset == "GluGluHToWWTo4q_M-125" else 1

        skimmed_events = {}

        if shift_name is None:
            # gen vars - saving HH, bb, VV, and 4q 4-vectors + Higgs children information
            for d in gen_selection_dict:
                if d in dataset:
                    vars_dict, (genbb, genq) = gen_selection_dict[d](
                        events, events.FatJet, selection, cutflow, gen_weights, P4
                    )
                    skimmed_events = {**skimmed_events, **vars_dict}

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
            add_selection("trigger", HLT_triggered, selection, cutflow, isData, gen_weights)

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
            gen_weights,
        )

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

        # deep-WH to compare SF with resonant WWW
        # https://indico.cern.ch/event/1023231/contributions/4296222/attachments/2217081/3756197/VVV_0lep_PreApp.pdf

        if "GluGluToHHTobbVV_node_cHHH1" in dataset:
            skimmed_events["ak8FatJetdeepTagMD_H4qvsQCD"] = pad_val(
                events.FatJet.deepTagMD_H4qvsQCD,
                2,
                -1,
                axis=1,
            )

        # calculate weights
        if isData:
            skimmed_events["weight"] = np.ones(n_events)
        else:
            weights.add("genweight", gen_weights)

            add_pileup_weight(weights, year, events.Pileup.nPU.to_numpy())
            add_VJets_kFactors(weights, events.GenPart, dataset)

            if "GluGluToHHTobbVV" in dataset or "WJets" in dataset or "ZJets" in dataset:
                add_ps_weight(weights, events.PSWeight)

            if "GluGluToHHTobbVV" in dataset:
                if "LHEPdfWeight" in events.fields:
                    add_pdf_weight(weights, events.LHEPdfWeight)
                else:
                    add_pdf_weight(weights, [])
                if "LHEScaleWeight" in events.fields:
                    add_scalevar_7pt(weights, events.LHEScaleWeight)
                else:
                    add_scalevar_7pt(weights, [])

            if year in ("2016APV", "2016", "2017"):
                weights.add(
                    "L1EcalPrefiring",
                    events.L1PreFiringWeight.Nom,
                    events.L1PreFiringWeight.Up,
                    events.L1PreFiringWeight.Dn,
                )

            # TODO: trigger SFs here once calculated properly

            if self._systematics and shift_name is None:
                systematics = [None] + list(weights.variations)
            else:
                systematics = [shift_name]

            # TODO: need to be careful about the sum of gen weights used for the LHE/QCDScale uncertainties
            logger.debug("weights ", weights._weights.keys())
            for systematic in systematics:
                if systematic in weights.variations:
                    weight = weights.weight(modifier=systematic)
                    weight_name = f"weight_{systematic}"
                else:
                    weight = weights.weight()
                    weight_name = "weight"

                # this still needs to be normalized with the acceptance of the pre-selection (done in post processing)
                if dataset in self.XSECS:
                    skimmed_events[weight_name] = (
                        self.XSECS[dataset]
                        * self.LUMI[year]
                        * weight  # includes genWeight (or signed genWeight)
                    )
                else:
                    logger.warning("Weight not normalized to cross section")
                    skimmed_events[weight_name] = weight

        # apply selections
        sel_all = selection.all(*selection.names)

        skimmed_events = {key: value[sel_all] for (key, value) in skimmed_events.items()}

        ################
        # Lund plane SFs
        ################

        if "GluGluToHHTobbVV_node_cHHH" in dataset and shift_name is None:
            genbb = genbb[sel_all]
            genq = genq[sel_all]

            sf_dicts = []

            for i in range(num_jets):
                bb_select = skimmed_events["ak8FatJetHbb"][:, i].astype(bool)
                VV_select = skimmed_events["ak8FatJetHVV"][:, i].astype(bool)

                # selectors for Hbb jets and HVV jets with 2, 3, or 4 prongs separately
                selectors = {
                    # name: (selector, gen quarks, num prongs)
                    "bb": (bb_select, genbb, 2),
                    **{
                        f"VV{k}q": (
                            VV_select * (skimmed_events["ak8FatJetHVVNumProngs"] == k),
                            genq,
                            k,
                        )
                        for k in range(2, 4 + 1)
                    },
                }

                selected_sfs = {}

                for key, (selector, gen_quarks, num_prongs) in selectors.items():
                    if np.sum(selector) > 0:
                        selected_sfs[key] = get_lund_SFs(
                            events[sel_all][selector],
                            i,
                            num_prongs,
                            gen_quarks[selector],
                            trunc_gauss=False,
                            lnN=True,
                        )

                sf_dict = {}

                # collect all the scale factors, fill in 0s for unmatched jets
                for key, val in selected_sfs["bb"].items():
                    arr = np.ones((np.sum(sel_all), val.shape[1]))

                    for select_key, (selector, _, _) in selectors.items():
                        if np.sum(selector) > 0:
                            arr[selector] = selected_sfs[select_key][key]

                    sf_dict[key] = arr

                sf_dicts.append(sf_dict)

            sf_dicts = concatenate_dicts(sf_dicts)

            skimmed_events = {**skimmed_events, **sf_dicts}

        if self._inference:
            # apply HWW4q tagger
            pnet_vars = {}
            pnet_vars = runInferenceTriton(
                self.tagger_resources_path,
                events[sel_all],
                ak15=False,
                all_outputs=False,
            )
            skimmed_events = {
                **skimmed_events,
                **{key: value for (key, value) in pnet_vars.items()},
            }

        if len(skimmed_events["weight"]):
            fname = (
                events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
            )
            odir_str = None

            # change keys if shift name is not None
            if shift_name is not None:
                # logger.debug("renaming keys")
                # skimmed_events = {f"{key}_{shift_name}":val for key,val in skimmed_events.items()}

                odir_str = f"_{shift_name}"

            df = self.to_pandas(skimmed_events)
            self.dump_table(df, fname, odir_str)

        if shift_name is not None:
            out_dict = {shift_name: {year: {dataset: {"nevents": n_events, "cutflow": cutflow}}}}
        else:
            out_dict = {"all": {year: {dataset: {"nevents": n_events, "cutflow": cutflow}}}}

        return out_dict

    def postprocess(self, accumulator):
        return accumulator
