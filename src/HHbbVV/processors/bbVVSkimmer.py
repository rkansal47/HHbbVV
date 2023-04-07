"""
Skimmer for bbVV analysis.
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

from .GenSelection import gen_selection_HHbbVV, gen_selection_HH4V, gen_selection_HYbbVV
from .TaggerInference import runInferenceTriton
from .utils import pad_val, add_selection, concatenate_dicts, P4
from .corrections import (
    add_pileup_weight,
    add_VJets_kFactors,
    add_ps_weight,
    add_pdf_weight,
    add_scalevar_7pt,
    get_jec_key,
    get_jec_jets,
    get_jmsr,
    get_lund_SFs,
)
from .common import LUMI, HLTs, btagWPs


# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {
    "HTo2bYTo2W": gen_selection_HYbbVV,
    "XToYHTo2W2BTo4Q2B": gen_selection_HYbbVV,
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

    # key is name in nano files, value will be the name in the skimmed output
    skim_vars = {
        "FatJet": {
            **P4,
            "msoftdrop": "Msd",
            "particleNetMD_QCD": "ParticleNetMD_QCD",
            "particleNetMD_Xbb": "ParticleNetMD_Xbb",
            "particleNet_H4qvsQCD": "ParticleNet_Th4q",
            "particleNet_mass": "ParticleNetMass",
        },
        "GenHiggs": P4,
        "other": {"MET_pt": "MET_pt"},
    }

    ak8_jet_selection = {
        "pt": 300.0,
        "eta": 2.4,
        "VVmsd": 50,
        "VVparticleNet_mass": 50,
        "bbparticleNet_mass": 50,
        "bbFatJetParticleNetMD_Txbb": 0.8,
    }

    jecs = {
        "JES": "JES_jes",
        "JER": "JER",
    }

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
            # [pd.DataFrame(v.reshape(v.shape[0], -1)) for k, v in events.items()],
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
        isSignal = "GluGluToHHTobbVV" in dataset or "XToYHTo2W2BTo4Q2B" in dataset

        if isSignal:
            # only signs for HH
            # TODO: check if this is also the case for HY
            gen_weights = np.sign(events["genWeight"])
        elif not isData:
            gen_weights = events["genWeight"].to_numpy()
        else:
            gen_weights = None

        n_events = len(events) if isData else np.sum(gen_weights)
        selection = PackedSelection()
        weights = Weights(len(events), storeIndividual=True)

        cutflow = OrderedDict()
        cutflow["all"] = n_events

        selection_args = (selection, cutflow, isData, gen_weights)

        num_jets = 2 if not dataset == "GluGluHToWWTo4q_M-125" else 1
        fatjets, jec_shifted_vars = get_jec_jets(events, year, isData, self.jecs)

        # change to year with suffix after updated JMS/R values
        jmsr_shifted_vars = get_jmsr(fatjets, num_jets, year_nosuffix, isData)

        skimmed_events = {}

        #########################
        # Save / derive variables
        #########################

        # TODO: resonant selection gets rid of events where Ws decay into Ws first
        # gen vars - saving HH, bb, VV, and 4q 4-vectors + Higgs children information
        for d in gen_selection_dict:
            if d in dataset:
                vars_dict, (genbb, genq) = gen_selection_dict[d](
                    events, fatjets, selection, cutflow, gen_weights, P4
                )
                skimmed_events = {**skimmed_events, **vars_dict}

        # FatJet vars

        ak8FatJetVars = {
            f"ak8FatJet{key}": pad_val(fatjets[var], num_jets, axis=1)
            for (var, key) in self.skim_vars["FatJet"].items()
        }

        # JEC vars

        for var in ["pt"]:
            key = self.skim_vars["FatJet"][var]
            for shift, vals in jec_shifted_vars[var].items():
                if shift != "":
                    ak8FatJetVars[f"ak8FatJet{key}_{shift}"] = pad_val(vals, num_jets, axis=1)

        # JMSR vars

        for var in ["msoftdrop", "particleNet_mass"]:
            key = self.skim_vars["FatJet"][var]
            for shift, vals in jmsr_shifted_vars[var].items():
                # overwrite saved mass vars with corrected ones
                label = "" if shift == "" else "_" + shift
                ak8FatJetVars[f"ak8FatJet{key}{label}"] = vals

        # particlenet xbb vs qcd

        fatjets["Txbb"] = fatjets.particleNetMD_Xbb / (
            fatjets.particleNetMD_QCD + fatjets.particleNetMD_Xbb
        )

        ak8FatJetVars["ak8FatJetParticleNetMD_Txbb"] = pad_val(
            fatjets["Txbb"],
            num_jets,
            axis=1,
        )

        # bb VV assignment

        bb_mask = (
            ak8FatJetVars["ak8FatJetParticleNetMD_Txbb"][:, 0]
            >= ak8FatJetVars["ak8FatJetParticleNetMD_Txbb"][:, 1]
        )
        bb_mask = np.stack((bb_mask, ~bb_mask)).T

        # dijet variables

        dijetVars = {}

        for shift in jec_shifted_vars["pt"]:
            label = "" if shift == "" else "_" + shift
            dijetVars = {**dijetVars, **self.getDijetVars(ak8FatJetVars, bb_mask, pt_shift=label)}

        for shift in jmsr_shifted_vars["msoftdrop"]:
            if shift != "":
                label = "_" + shift
                dijetVars = {
                    **dijetVars,
                    **self.getDijetVars(ak8FatJetVars, bb_mask, mass_shift=label),
                }

        otherVars = {
            key: events[var.split("_")[0]]["_".join(var.split("_")[1:])].to_numpy()
            for (var, key) in self.skim_vars["other"].items()
        }

        skimmed_events = {**skimmed_events, **ak8FatJetVars, **dijetVars, **otherVars}

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

        # pt, eta cuts: check if jet passes pT cut in any of the JEC variations
        cuts = []

        for pts in jec_shifted_vars["pt"].values():
            cut = np.prod(
                pad_val(
                    (pts > self.ak8_jet_selection["pt"])
                    * (np.abs(fatjets.eta) < self.ak8_jet_selection["eta"]),
                    num_jets,
                    False,
                    axis=1,
                ),
                axis=1,
            )
            cuts.append(cut)

        add_selection("ak8_pt_eta", np.any(cuts, axis=0), *selection_args)

        # mass cuts: check if jet passes mass cut in any of the JMS/R variations
        cuts = []

        for shift in jmsr_shifted_vars["msoftdrop"]:
            msds = jmsr_shifted_vars["msoftdrop"][shift]
            pnetms = jmsr_shifted_vars["particleNet_mass"][shift]

            # TODO: change to cut on regressed mass only
            cut = (
                (msds[~bb_mask] >= self.ak8_jet_selection["VVmsd"])
                | (pnetms[~bb_mask] >= self.ak8_jet_selection["VVparticleNet_mass"])
            ) * (pnetms[bb_mask] >= self.ak8_jet_selection["bbparticleNet_mass"])
            cuts.append(cut)

        add_selection("ak8_mass", np.any(cuts, axis=0), *selection_args)

        # Txbb pre-selection cut

        txbb_cut = (
            ak8FatJetVars["ak8FatJetParticleNetMD_Txbb"][bb_mask]
            >= self.ak8_jet_selection["bbFatJetParticleNetMD_Txbb"]
        )
        add_selection("ak8bb_txbb", txbb_cut, *selection_args)

        # 2018 HEM cleaning
        # https://indico.cern.ch/event/1249623/contributions/5250491/attachments/2594272/4477699/HWW_0228_Draft.pdf
        if year == "2018":
            hem_cleaning = (
                ((events.run >= 319077) & isData)  # if data check if in Runs C or D
                # else for MC randomly cut based on lumi fraction of C&D
                | ((np.random.rand(len(events)) < 0.632) & ~isData)
            ) & (
                ak.any(
                    (
                        (events.Jet.pt > 30.0)
                        & (events.Jet.eta > -3.2)
                        & (events.Jet.eta < -1.3)
                        & (events.Jet.phi > -1.57)
                        & (events.Jet.phi < -0.87)
                    ),
                    -1,
                )
                | ((events.MET.phi > -1.62) & (events.MET.pt < 470.0) & (events.MET.phi < -0.62))
            )

            add_selection("hem_cleaning", ~hem_cleaning, *selection_args)

        #########################
        # Veto variables
        #########################

        electrons, muons = events.Electron, events.Muon

        # selection from https://github.com/cmantill/boostedhiggs/blob/e7dc206de17fd108a5e1abcb7d76a52ccb636599/boostedhiggs/hwwprocessor.py#L185-L224

        good_muons = (
            (muons.pt > 30)
            & (np.abs(muons.eta) < 2.4)
            & (np.abs(muons.dz) < 0.1)
            & (np.abs(muons.dxy) < 0.05)
            & (muons.sip3d <= 4.0)
            & muons.mediumId
        )
        n_good_muons = ak.sum(good_muons, axis=1)

        good_electrons = (
            (electrons.pt > 38)
            & (np.abs(electrons.eta) < 2.4)
            & ((np.abs(electrons.eta) < 1.44) | (np.abs(electrons.eta) > 1.57))
            & (np.abs(electrons.dz) < 0.1)
            & (np.abs(electrons.dxy) < 0.05)
            & (electrons.sip3d <= 4.0)
            & (electrons.mvaFall17V2noIso_WP90)
        )
        n_good_electrons = ak.sum(good_electrons, axis=1)

        bbjet = ak.pad_none(fatjets[ak.argsort(fatjets.Txbb, ascending=False)], 1, axis=1)[:, 0]

        goodjets = (
            (events.Jet.pt > 30)
            & (np.abs(events.Jet.eta) < 5.0)
            & events.Jet.isTight
            & (events.Jet.puId > 0)
            & (events.Jet.btagDeepFlavB > btagWPs["deepJet"][year]["M"])
            & (np.abs(events.Jet.delta_r(bbjet)) > 0.8)
        )
        n_good_jets = ak.sum(goodjets, axis=1)

        skimmed_events["nGoodMuons"] = n_good_muons.to_numpy()
        skimmed_events["nGoodElectrons"] = n_good_electrons.to_numpy()
        skimmed_events["nGoodJets"] = n_good_jets.to_numpy()

        ######################
        # Weights
        ######################

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

            systematics = [""]

            if self._systematics:
                systematics += list(weights.variations)

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
                if dataset in self.XSECS or "XToYHTo2W2BTo4Q2B" in dataset:
                    # 1 fb xsec for now for resonant signal
                    xsec = self.XSECS[dataset] if dataset in self.XSECS else 1e-3  # in pb
                    skimmed_events[weight_name] = (
                        xsec * LUMI[year] * weight  # includes genWeight (or signed genWeight)
                    )

                    if systematic == "":
                        # to check in postprocessing for xsec & lumi normalisation
                        skimmed_events["weight_noxsec"] = weight
                else:
                    logger.warning("Weight not normalized to cross section")
                    skimmed_events[weight_name] = weight

        print(cutflow)

        # reshape and apply selections
        sel_all = selection.all(*selection.names)

        skimmed_events = {
            key: value.reshape(len(skimmed_events["weight"]), -1)[sel_all]
            for (key, value) in skimmed_events.items()
        }

        ################
        # Lund plane SFs
        ################

        fatjet_idx = 0
        ak8_pfcands = events.FatJetPFCands
        ak8_pfcands = ak8_pfcands[ak8_pfcands.jetIdx == fatjet_idx]
        pfcands0 = events.PFCands[ak8_pfcands.pFCandsIdx]

        fatjet_idx = 1
        ak8_pfcands = events.FatJetPFCands
        ak8_pfcands = ak8_pfcands[ak8_pfcands.jetIdx == fatjet_idx]
        pfcands1 = events.PFCands[ak8_pfcands.pFCandsIdx]

        print(np.sort(ak.count(pfcands0[sel_all].pt, axis=1))[:5])
        print(np.sort(ak.count(pfcands1[sel_all].pt, axis=1))[:5])

        breakpoint()

        if isSignal and len(skimmed_events["weight"]):
            genbb = genbb[sel_all]
            genq = genq[sel_all]

            sf_dicts = []

            for i in range(num_jets):
                print(i)
                bb_select = skimmed_events["ak8FatJetHbb"][:, i].astype(bool)
                VV_select = skimmed_events["ak8FatJetHVV"][:, i].astype(bool)

                # selectors for Hbb jets and HVV jets with 2, 3, or 4 prongs separately
                selectors = {
                    # name: (selector, gen quarks, num prongs)
                    "bb": (bb_select, genbb, 2),
                    **{
                        f"VV{k}q": (
                            VV_select * (skimmed_events["ak8FatJetHVVNumProngs"].squeeze() == k),
                            genq,
                            k,
                        )
                        for k in range(2, 4 + 1)
                    },
                }

                selected_sfs = {}

                for key, (selector, gen_quarks, num_prongs) in selectors.items():
                    print(key)
                    if np.sum(selector) > 0:
                        sel_events = events[sel_all][selector]
                        selected_sfs[key] = get_lund_SFs(
                            sel_events,
                            i,
                            num_prongs,
                            gen_quarks[selector],
                            trunc_gauss=False,
                            lnN=True,
                        )
                        items = selected_sfs[key].items()

                sf_dict = {}

                # collect all the scale factors, fill in 0s for unmatched jets
                for key, val in items:
                    arr = np.ones((np.sum(sel_all), val.shape[1]))

                    for select_key, (selector, _, _) in selectors.items():
                        if np.sum(selector) > 0:
                            arr[selector] = selected_sfs[select_key][key]

                    sf_dict[key] = arr

                sf_dicts.append(sf_dict)

            sf_dicts = concatenate_dicts(sf_dicts)

            skimmed_events = {**skimmed_events, **sf_dicts}

        ######################
        # HWW Tagger Inference
        ######################

        # TODO: only need to run inference for the WW candidate jet
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

        df = self.to_pandas(skimmed_events)
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
        self.dump_table(df, fname)

        return {year: {dataset: {"nevents": n_events, "cutflow": cutflow}}}

    def postprocess(self, accumulator):
        return accumulator

    def getDijetVars(
        self, ak8FatJetVars: Dict, bb_mask: np.ndarray, pt_shift: str = None, mass_shift: str = None
    ):
        """Calculates Dijet variables for given pt / mass JEC / JMS/R variation"""
        dijetVars = {}

        ptlabel = pt_shift if pt_shift is not None else ""
        mlabel = mass_shift if mass_shift is not None else ""
        bbJet = vector.array(
            {
                "pt": ak8FatJetVars[f"ak8FatJetPt{ptlabel}"][bb_mask],
                "phi": ak8FatJetVars["ak8FatJetPhi"][bb_mask],
                "eta": ak8FatJetVars["ak8FatJetEta"][bb_mask],
                "M": ak8FatJetVars[f"ak8FatJetParticleNetMass{mlabel}"][bb_mask],
            }
        )

        VVJet = vector.array(
            {
                "pt": ak8FatJetVars[f"ak8FatJetPt{ptlabel}"][~bb_mask],
                "phi": ak8FatJetVars["ak8FatJetPhi"][~bb_mask],
                "eta": ak8FatJetVars["ak8FatJetEta"][~bb_mask],
                "M": ak8FatJetVars[f"ak8FatJetMsd{mlabel}"][~bb_mask],
                # TODO: change this to ParticleNetMass for next run
                # "M": ak8FatJetVars[f"ak8FatJetParticleNetMass{mlabel}"][~bb_mask],
            }
        )

        Dijet = bbJet + VVJet

        shift = ptlabel + mlabel

        dijetVars[f"DijetPt{shift}"] = Dijet.pt
        dijetVars[f"DijetMass{shift}"] = Dijet.M
        dijetVars[f"DijetEta{shift}"] = Dijet.eta

        dijetVars[f"bbFatJetPtOverDijetPt{shift}"] = bbJet.pt / Dijet.pt
        dijetVars[f"VVFatJetPtOverDijetPt{shift}"] = VVJet.pt / Dijet.pt
        dijetVars[f"VVFatJetPtOverbbFatJetPt{shift}"] = VVJet.pt / bbJet.pt

        return dijetVars
