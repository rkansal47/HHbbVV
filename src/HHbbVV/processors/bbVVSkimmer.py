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

from .GenSelection import gen_selection_HHbbVV, gen_selection_HH4V, gen_selection_HYbbVV, G_PDGID
from .TaggerInference import runInferenceTriton
from .utils import pad_val, add_selection, concatenate_dicts, select_dicts, P4
from .corrections import (
    add_pileup_weight,
    add_VJets_kFactors,
    add_top_pt_weight,
    add_ps_weight,
    add_pdf_weight,
    add_scalevar_7pt,
    add_trig_effs,
    get_jec_key,
    get_jec_jets,
    get_jmsr,
    get_lund_SFs,
)
from .common import LUMI, HLTs, btagWPs, jec_shifts, jmsr_shifts
from . import common


# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {
    "HTo2bYTo2W": gen_selection_HYbbVV,
    "XToYHTo2W2BTo4Q2B": gen_selection_HYbbVV,
    "GluGluToHHTobbVV_node_cHHH": gen_selection_HHbbVV,
    "VBF_HHTobbVV": gen_selection_HHbbVV,
    "jhu_HHbbWW": gen_selection_HHbbVV,
    "GluGluToBulkGravitonToHHTo4W_JHUGen": gen_selection_HH4V,
    "GluGluToHHTo4V_node_cHHH1": gen_selection_HH4V,
    "GluGluHToWWTo4q_M-125": gen_selection_HH4V,
}


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    preselection = {
        "pt": 300.0,
        "eta": 2.4,
        "VVmsd": 50,
        "VVparticleNet_mass": [50, 250],
        "bbparticleNet_mass": [92.5, 162.5],
        "bbFatJetParticleNetMD_Txbb": 0.8,
        "DijetMass": 800,  # TODO
        # "nGoodElectrons": 0,
    }

    jecs = common.jecs

    # only the branches necessary for templates and post processing
    min_branches = [
        "ak8FatJetPt",
        "ak8FatJetMsd",
        "ak8FatJetParticleNetMass",
        "DijetMass",
        "ak8FatJetHbb",
        "ak8FatJetHVV",
        "ak8FatJetHVVNumProngs",
        "ak8FatJetParticleNetMD_Txbb",
        "VVFatJetParTMD_THWWvsT",
    ]

    for shift in jec_shifts:
        min_branches.append(f"ak8FatJetPt_{shift}")
        min_branches.append(f"DijetMass_{shift}")

    for shift in jmsr_shifts:
        min_branches.append(f"ak8FatJetParticleNetMass_{shift}")
        min_branches.append(f"DijetMass_{shift}")

    def __init__(
        self, xsecs={}, save_ak15=False, save_systematics=True, inference=True, save_all=True
    ):
        super(bbVVSkimmer, self).__init__()

        self.XSECS = xsecs  # in pb
        self.save_ak15 = save_ak15

        # save systematic variations
        self._systematics = save_systematics

        # run inference
        self._inference = inference

        # save all branches or only necessary ones
        self._save_all = save_all

        # for tagger model and preprocessing dict
        self.tagger_resources_path = (
            str(pathlib.Path(__file__).parent.resolve()) + "/tagger_resources/"
        )

        self._accumulator = processor.dict_accumulator({})

        logger.info(
            f"Running skimmer with inference {self._inference} and systematics {self._systematics} and save all {self._save_all}"
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
        isVBFSearch = True
        isSignal = (
            "GluGluToHHTobbVV" in dataset
            or "XToYHTo2W2BTo4Q2B" in dataset
            or "VBF_HHTobbVV" in dataset
        )

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

        #jets_ak4 = '' # deal with these guys later. not sure but I think the type is fatjet afterward which breaks things
        #if isVBFSearch:
        #    jets_ak4,jec_shifted_var_ak4 = get_jec_jets(events, year, isData, self.jecs,fatjets = False)

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

        # VBF ak4 Jet vars (jetId,puId, pt, eta, phi, M)
        #logger.warning(isVBFSearch) # note that this pads it to exactly 20 jets. maybe there is more. in the future I will make it just the best 2. I am doing this because to reshape we need np instead of ak.
        if isVBFSearch:
            ak4JetVars = { **self.getVBFVars(events,ak8FatJetVars,bb_mask) } #pad_val(jets_ak4[var], 20, axis=1) # consider using selection to make cuts for VBF instead of passing back boolean array or somth
                 # i can deal wit jet energy corrections later on. (we will isolate this to 2 vbf jets and then be able to pad). 550 might give issues
        
            skimmed_events = {**skimmed_events, **ak4JetVars}
            logger.warning('did thingy!!!!') # this is also quite goofy looking



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
                    (pts > self.preselection["pt"])
                    * (np.abs(fatjets.eta) < self.preselection["eta"]),
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

            if self._save_all:
                cut = (
                    (pnetms[~bb_mask] >= self.preselection["VVparticleNet_mass"][0])
                    + (msds[~bb_mask] >= self.preselection["VVparticleNet_mass"][0])
                ) * (pnetms[bb_mask] >= 50)
            else:
                cut = (
                    (pnetms[~bb_mask] >= self.preselection["VVparticleNet_mass"][0])
                    * (pnetms[~bb_mask] < self.preselection["VVparticleNet_mass"][1])
                    * (pnetms[bb_mask] >= self.preselection["bbparticleNet_mass"][0])
                    * (pnetms[bb_mask] < self.preselection["bbparticleNet_mass"][1])
                )

            cuts.append(cut)

        add_selection("ak8_mass", np.any(cuts, axis=0), *selection_args)

        # TODO: dijet mass: check if dijet mass cut passes in any of the JEC or JMC variations

        # Txbb pre-selection cut

        txbb_cut = (
            ak8FatJetVars["ak8FatJetParticleNetMD_Txbb"][bb_mask]
            >= self.preselection["bbFatJetParticleNetMD_Txbb"]
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

        # remove weird jets which have <4 particles (due to photon scattering?)
        pfcands_sel = []

        for i in range(num_jets):
            ak8_pfcands = events.FatJetPFCands
            ak8_pfcands = ak8_pfcands[ak8_pfcands.jetIdx == i]
            pfcands = events.PFCands[ak8_pfcands.pFCandsIdx]
            pfcands_sel.append(ak.count(pfcands.pdgId, axis=1) < 4)

        add_selection("photon_jets", ~np.sum(pfcands_sel, axis=0).astype(bool), *selection_args)

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
        # Remove branches
        ######################

        # if not saving all, save only necessary branches
        if not self._save_all:
            skimmed_events = {
                key: val for key, val in skimmed_events.items() if key in self.min_branches
            }

        ######################
        # Weights
        ######################

        if isData:
            skimmed_events["weight"] = np.ones(n_events)
        else:
            weights.add("genweight", gen_weights)

            add_pileup_weight(weights, year, events.Pileup.nPU.to_numpy())
            add_VJets_kFactors(weights, events.GenPart, dataset)

            # if dataset.startswith("TTTo"):
            #     # TODO: need to add uncertainties and rescale yields (?)
            #     add_top_pt_weight(weights, events)

            # TODO: figure out which of these apply to VBF, single Higgs, ttbar etc.

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

            if not self._save_all:
                add_trig_effs(weights, fatjets, year, num_jets)

            # xsec and luminosity and normalization
            # this still needs to be normalized with the acceptance of the pre-selection (done in post processing)
            if dataset in self.XSECS or "XToYHTo2W2BTo4Q2B" in dataset:
                # 1 fb xsec for now for resonant signal
                xsec = self.XSECS[dataset] if dataset in self.XSECS else 1e-3  # in pb

                weight_norm = xsec * LUMI[year]
            else:
                logger.warning("Weight not normalized to cross section")
                weight_norm = 1

            if self._save_all:
                systematics = [""]
            else:
                systematics = ["", "notrigeffs"]

            if self._systematics:
                systematics += list(weights.variations)

            # TODO: need to be careful about the sum of gen weights used for the LHE/QCDScale uncertainties
            logger.debug("weights ", weights._weights.keys())
            for systematic in systematics:
                if systematic in weights.variations:
                    weight = weights.weight(modifier=systematic)
                    weight_name = f"weight_{systematic}"
                elif systematic == "":
                    weight = weights.weight()
                    weight_name = "weight"
                elif systematic == "notrigeffs":
                    weight = weights.partial_weight(exclude=["trig_effs"])
                    weight_name = "weight_noTrigEffs"

                # includes genWeight (or signed genWeight)
                skimmed_events[weight_name] = weight * weight_norm

                if systematic == "":
                    # to check in postprocessing for xsec & lumi normalisation
                    skimmed_events["weight_noxsec"] = weight

        # reshape and apply selections
        sel_all = selection.all(*selection.names)

        skimmed_events = {
            key: value.reshape(len(skimmed_events["weight"]), -1)[sel_all]
            for (key, value) in skimmed_events.items()
        }

        bb_mask = bb_mask[sel_all]

        ################
        # Lund plane SFs
        ################

        if isSignal:
            # TODO: remember to add new LP variables
            items = [
                ("lp_sf_lnN", 101),
                ("lp_sf_sys_down", 1),
                ("lp_sf_sys_up", 1),
                ("lp_sf_double_matched_event", 1),
                ("lp_sf_unmatched_quarks", 1),
                ("lp_sf_num_sjpt_gt350", 1),
            ]

            if len(skimmed_events["weight"]):
                genbb = genbb[sel_all]
                genq = genq[sel_all]

                sf_dicts = []
                lp_num_jets = num_jets if self._save_all else 1

                for i in range(lp_num_jets):
                    if self._save_all:
                        bb_select = skimmed_events["ak8FatJetHbb"][:, i].astype(bool)
                        VV_select = skimmed_events["ak8FatJetHVV"][:, i].astype(bool)
                    else:
                        # only do VV jets
                        bb_select = np.zeros(len(skimmed_events["ak8FatJetHbb"])).astype(bool)
                        # exactly 1 jet is matched (otherwise those SFs are ignored in post-processing anyway)
                        VV_select = np.sum(skimmed_events["ak8FatJetHVV"], axis=1) == 1

                    # selectors for Hbb jets and HVV jets with 2, 3, or 4 prongs separately
                    selectors = {
                        # name: (selector, gen quarks, num prongs)
                        "bb": (bb_select, genbb, 2),
                        **{
                            f"VV{k}q": (
                                VV_select
                                * (skimmed_events["ak8FatJetHVVNumProngs"].squeeze() == k),
                                genq,
                                k,
                            )
                            for k in range(2, 4 + 1)
                        },
                    }

                    selected_sfs = {}

                    for key, (selector, gen_quarks, num_prongs) in selectors.items():
                        if np.sum(selector) > 0:
                            sel_events = events[sel_all][selector]
                            selected_sfs[key] = get_lund_SFs(
                                sel_events,
                                i
                                if self._save_all
                                else skimmed_events["ak8FatJetHVV"][selector][:, 1],
                                num_prongs,
                                gen_quarks[selector],
                                trunc_gauss=False,
                                lnN=True,
                            )

                    sf_dict = {}

                    # collect all the scale factors, fill in 1s for unmatched jets
                    for key, shape in items:
                        arr = np.ones((np.sum(sel_all), shape))

                        for select_key, (selector, _, _) in selectors.items():
                            if np.sum(selector) > 0:
                                arr[selector] = selected_sfs[select_key][key]

                        sf_dict[key] = arr

                    sf_dicts.append(sf_dict)

                sf_dicts = concatenate_dicts(sf_dicts)

            else:
                print("No signal events selected")
                sf_dicts = {}
                for key, shape in items:
                    arr = np.ones((np.sum(sel_all), shape))
                    sf_dicts[key] = arr

            skimmed_events = {**skimmed_events, **sf_dicts}

        ######################
        # HWW Tagger Inference
        ######################

        # TODO: only need to run inference for the WW candidate jet
        if self._inference:
            # apply HWW4q tagger
            pnet_vars = {}
            if self._save_all:
                # run on both jets
                pnet_vars = runInferenceTriton(
                    self.tagger_resources_path,
                    events[sel_all],
                    ak15=False,
                    all_outputs=False,
                )
                skimmed_events = {**skimmed_events, **pnet_vars}
            else:
                # run only on VV candidate jet
                pnet_vars = runInferenceTriton(
                    self.tagger_resources_path,
                    events[sel_all],
                    num_jets=1,
                    in_jet_idx=bb_mask[:, 0].astype(int),
                    ak15=False,
                    all_outputs=False,
                    jet_label="VV",
                )
                skimmed_events = {
                    **skimmed_events,
                    **{key: val for (key, val) in pnet_vars.items() if key in self.min_branches},
                }

        df = self.to_pandas(skimmed_events)
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
        self.dump_table(df, fname)

        return {year: {dataset: {"nevents": n_events, "cutflow": cutflow}}}

    def postprocess(self, accumulator):
        return accumulator
    
    def getVBFVars(self,events: ak.Array,ak8FatJetVars: Dict, bb_mask: np.ndarray,skim_vars: dict = None, pt_shift: str = None, mass_shift: str = None): #  NanoEventsArray not imported yet, may not be necessary?
        """Computes selections on vbf jet candidates. Sorts jets by pt,delta eta,dijet mass. Computes a nGoodVBFJets thing return skim variables for the first two of the filtered guys and the nGoodVBFJets thingy"""
            
        # TODO implement use of pt_shift, mass_shift, skim_vars. Decide on best sorting and remove extras.
            
        vbfVars = {}    
        jets = events.Jet
        
        # AK4 jets definition
        #  The PF candidates associated to pileup vertices are removed from the
        # 430 jet constituents using the charged hadron subtraction (CHS) algorithm [22]. 
        # 431 Jet energy and resolution corrections supplied by the JetMET POG are applied [19, 20], using
        # 432 the tags indicated in Table 11. The L1FastJet, L2Relative, and L3Absolute corrections
        # 433 for CHS jets are applied to data and simulation. The L2L3Residual corrections are applied
        # 434 to the data. The selected jets are required to have pT > 25 GeV and to be within |η| < 4.7. The
        # 435 AK4 jets are also required to satisfy the tight PF Jet identification criteria recommended by the
        # 436 JetMET POG [21].
        # 437 In addition, jets with pT < 50 GeV are required to pass medium or tight working points of
        # 438 the pileup ID discriminator [23]. Corrective scale factors for the pileup ID are also applied as
        # 439 described in Section 9.2.
        
        ak4_jet_mask = (jets.pt > 25) & (np.abs(jets.eta) < 4.7) & (jets.jetId != 1) &  ((jets.pt > 50) | ((jets.puId == 7) | (jets.puId == 7)) )
        
        # was charged hadron subtraction already performed? I think so before to the nano. not sure abt correctons but I think also.
        
        
        # then we need to filter based on the placement of the jets:
        
        #  To identify the two tag jets, a collection of AK4 jets with charged hadron subtraction (CHS) with
        # 743 pT > 25 GeV, |η| < 4.7, and other preselections detailed in Section 5.4 is considered. The VBF
        # 744 tag jet candidates (j) must not overlap with reconstructed electrons, muons or the two Higgs
        # 745 candidate AK8 jets: ∆R(j, e) > 0.4, ∆R(j, µ) > 0.4 and ∆R(j, AK8) > 1.2. For electrons and
        # 746 muons, looser selection criteria than the ones used in lepton veto are applied: any reconstructed
        # 747 electrons (muons) with pT > 5 (7) GeV are vetoed, without requiring additional identification
        # 748 or isolation criteria.

        # reconstructing electrons + muons for delta R calculations. we need to first filter the electrons and muons so that only 
        # the low pt ones remain. Then we must calculate delta r with each of the possible vbfs and remove them based on that.
        
        # compute mask for electron/muon overlap
        electrons, muons = events.Electron[events.Electron.pt<5], events.Muon[events.Muon.pt<7]
        e_pairs = ak.cartesian([jets, electrons],nested= True)
        e_pairs_mask = np.abs(e_pairs.slot0.delta_r(e_pairs.slot1)) < 0.4 # this may not work due to type. just compute as dist in phi eta plane
        m_pairs = ak.cartesian([jets, muons],nested= True)
        m_pairs_mask = np.abs(m_pairs.slot0.delta_r(m_pairs.slot1)) < 0.4
        #counts = [len(event) for event in jets]
        #logger.warning(f"Jets total elements: {counts} len: {len(jets)}")
        #counts = [len(event) for event in ak.any(e_pairs_mask, axis=-1)]
        #logger.warning(f"Jets total elements: {counts} len: {len(ak.any(e_pairs_mask, axis=-1))}")
        electron_muon_overlap_mask = ~(ak.any(e_pairs_mask, axis=-1) | ak.any(m_pairs_mask, axis=-1)) # both should be true initially if electron or muons overlap
        

        # reconstructing fatjets for delta R calculations
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
        
        fatjet_overlap_mask = (np.abs(jets.delta_r(bbJet)) > 1.2) & (np.abs(jets.delta_r(VVJet)) > 1.2) # this might not work due to types
        
        
        
        
        # compute n_good_vbf_jets + incorporate eta_jj > 4.0
        vbfJets_mask = ak4_jet_mask & electron_muon_overlap_mask & fatjet_overlap_mask
        vbfJets = jets[vbfJets_mask]
        
        # Generating the three different sorting methods: (pt, dijet eta, dijet mass). For each we keep the two best
        vbfJets_sorted_pt = vbfJets[ak.argsort(vbfJets.pt,ascending = False)]
        vbfJets_sorted_pt = ak.pad_none(vbfJets_sorted_pt, 2, clip=True) # this is the only which does not guarantee two guys. in the other sorts, the entries are specifically None.
        
        
        # dijet eta sorting
        jet_pairs = ak.combinations(vbfJets, 2, axis=1, fields=["j1", "j2"])
        delta_eta = np.abs(jet_pairs.j1.eta - jet_pairs.j2.eta) 
        eta_sorted_pairs = jet_pairs[ak.argsort(delta_eta, axis=1,ascending= False)] # picks the two furthest jets.
        eta_first_pairs = ak.firsts(eta_sorted_pairs, axis=1)
        eta_sorted_mask = ak.any((vbfJets[:, :, None].eta == eta_first_pairs.j1.eta) | (vbfJets[:, :, None].eta == eta_first_pairs.j2.eta), axis=2)
        vbfJets_sorted_eta = vbfJets[eta_sorted_mask]
        
        # dijet mass sorting
        jj = jet_pairs.j1 + jet_pairs.j2
        mass_sorted_pairs = jet_pairs[ak.argsort(jj.mass, axis=1,ascending= False)] # picks the two furthest jets.
        mass_first_pairs = ak.firsts(mass_sorted_pairs, axis=1)
        mass_sorted_mask = ak.any((vbfJets[:, :, None].mass == mass_first_pairs.j1.mass) | (vbfJets[:, :, None].mass == mass_first_pairs.j2.mass), axis=2)
        vbfJets_sorted_mass = vbfJets[mass_sorted_mask]
        
        # Compute dijet eta and dijet mass cuts
        jj_sorted_mass = mass_first_pairs.j1 + mass_first_pairs.j2 # we update dijet since the previous one had many per event. this should be one number per event.
        mass_jj_cut_sorted_mass = jj_sorted_mass.mass > 500
        eta_jj_cut_sorted_mass = np.abs(mass_first_pairs.j1.eta - mass_first_pairs.j2.eta)  > 4.0
        vbfJets_mask_sorted_mass = vbfJets_mask * mass_jj_cut_sorted_mass * eta_jj_cut_sorted_mass
        
        jj_sorted_eta = eta_first_pairs.j1 + eta_first_pairs.j2
        mass_jj_cut_sorted_eta = jj_sorted_eta.mass  > 500
        eta_jj_cut_sorted_eta = np.abs(eta_first_pairs.j1.eta - eta_first_pairs.j2.eta)  > 4.0
        vbfJets_mask_sorted_eta = vbfJets_mask * mass_jj_cut_sorted_eta * eta_jj_cut_sorted_eta
        
        
        # here is a really slow way to compute pt mass
        # pt_jet_pairs = ak.combinations(vbfJets_sorted_pt, 2, axis=1, fields=["j1", "j2"]) # we have to compute the pairs of pt to calculate the mass.
        # delta_pt = np.abs(pt_jet_pairs.j1.pt + pt_jet_pairs.j2.eta) 
        # pt_sorted_pairs = pt_jet_pairs[ak.argsort(delta_pt, axis=1,ascending= False)] # picks the two furthest jets.
        # pt_first_pairs = ak.firsts(pt_sorted_pairs, axis=1)
        # jj_sorted_pt = pt_first_pairs.j1 + pt_first_pairs.j2
        # print(jj_sorted_pt.mass) 
        
        # pt sorted eta and dijet mass mask
        jj_sorted_pt = vbfJets_sorted_pt[:,0:1] + vbfJets_sorted_pt[:,1:2]
        mass_jj_cut_sorted_pt = jj_sorted_pt.mass  > 500
        eta_jj_cut_sorted_pt = np.abs(vbfJets_sorted_pt[:,0:1].eta - vbfJets_sorted_pt[:,1:2].eta)  > 4.0
        vbfJets_mask_sorted_pt = vbfJets_mask * mass_jj_cut_sorted_pt * eta_jj_cut_sorted_pt
 
        
        
        
        
        
        
        
        n_good_vbf_jets = ak.fill_none(ak.sum(vbfJets_mask, axis=1),0) #* eta_jj_mask * mass_jj_mask # filters out the events where the vbf jets are too close and mass too small., May need to convert to 0, 1 array instead of mask.
        n_good_vbf_jets_sorted_pt = ak.fill_none(ak.sum(vbfJets_mask_sorted_pt, axis=1),0)
        n_good_vbf_jets_sorted_mass = ak.fill_none(ak.sum(vbfJets_mask_sorted_mass, axis=1),0)
        n_good_vbf_jets_sorted_eta = ak.fill_none(ak.sum(vbfJets_mask_sorted_eta, axis=1),0)
        
        
        #logging.warning(f"Jets eta {jets.eta}")
        # logging.warning(f"Jets pt {jets.pt}")
        
        GEN_FLAGS = ["fromHardProcess"]
        vbfGenJets = events.GenPart[events.GenPart.hasFlags(["isHardProcess"])][:, 4:6]
        
        #logging.warning(f"Gen Jets eta {vbfGenJets.eta}")
        #logging.warning(f"Gen Jets mass {vbfGenJets.mass}")
        #logging.warning(f"Gen Jets pt {vbfGenJets.pt}")
        #logging.warning(f"Gen Jets pdgId {vbfGenJets.pdgId}")
        
        #self.getVBFGenMatchCount(events,jets)
        #self.getVBFGenMatchCount(events,vbfJets)
        #self.getVBFGenMatchCount(events,vbfJets[ak.argsort(vbfJets.btagDeepFlavCvL,ascending = False)][:,0:2])
        #self.getVBFGenMatchCount(events,vbfJets_sorted_pt)
        #self.getVBFGenMatchCount(events,vbfJets_sorted_mass)
        #self.getVBFGenMatchCount(events,vbfJets_sorted_eta)
        
        
        #logging.warning(np.sum(ak.sum(ak4_jet_mask, axis=1).to_numpy())) #," ," and final: {",np.sum(ak.sum(vbfJets_mask, axis=1).to_numpy())," compared to initial: {",np.sum(ak.sum(jets, axis=1).to_numpy()))
        #logging.warning(np.sum(ak.sum(electron_muon_overlap_mask, axis=1).to_numpy()))
        #logging.warning(np.sum(ak.sum(fatjet_overlap_mask, axis=1).to_numpy()))
        #logging.warning(np.sum(ak.sum(vbfJets_mask, axis=1).to_numpy()))
        #logging.warning(ak.sum(ak.num(jets, axis=1) ))
                                                                
        # dijet mass must be greater than 500
        #dijet = vbfJets[:,0:1] + vbfJets[:,1:2]
        #mass_jj_mask = dijet.mass > 500                                                        
        #eta_jj_mask = (np.abs(vbfJets[:,0:1].eta -vbfJets[:,1:2].eta) > 4.0)  
        #vbfJet1, vbfJet2 = vbfJets[:,0],vbfJets[:,1]
        
        vbfVars[f"vbfptGen"] = pad_val(vbfGenJets.pt, 2, axis=1)
        vbfVars[f"vbfetaGen"] = pad_val(vbfGenJets.eta, 2, axis=1)
        vbfVars[f"vbfphiGen"] = pad_val(vbfGenJets.phi, 2, axis=1)
        vbfVars[f"vbfMGen"] = pad_val(vbfGenJets.mass, 2, axis=1)
        
        vbfVars[f"vbfptSortedRand"] = pad_val(vbfJets[ak.argsort(vbfJets.btagDeepFlavCvL,ascending = False)][:,0:2].pt, 2, axis=1)
        vbfVars[f"vbfetaSortedRand"] = pad_val(vbfJets[ak.argsort(vbfJets.btagDeepFlavCvL,ascending = False)][:,0:2].eta, 2, axis=1)
        vbfVars[f"vbfphiSortedRand"] = pad_val(vbfJets[ak.argsort(vbfJets.btagDeepFlavCvL,ascending = False)][:,0:2].phi, 2, axis=1)
        vbfVars[f"vbfMSortedRand"] = pad_val(vbfJets[ak.argsort(vbfJets.btagDeepFlavCvL,ascending = False)][:,0:2].mass, 2, axis=1)
        
        vbfVars[f"vbfptSortedpt"] = pad_val(vbfJets_sorted_pt.pt, 2, axis=1)
        vbfVars[f"vbfetaSortedpt"] = pad_val(vbfJets_sorted_pt.eta, 2, axis=1)
        vbfVars[f"vbfphiSortedpt"] = pad_val(vbfJets_sorted_pt.phi, 2, axis=1)
        vbfVars[f"vbfMSortedpt"] = pad_val(vbfJets_sorted_pt.mass, 2, axis=1)
        
        vbfVars[f"vbfptSortedM"] = pad_val(vbfJets_sorted_mass.pt, 2, axis=1)
        vbfVars[f"vbfetaSortedM"] = pad_val(vbfJets_sorted_mass.eta, 2, axis=1)
        vbfVars[f"vbfphiSortedM"] = pad_val(vbfJets_sorted_mass.phi, 2, axis=1)
        vbfVars[f"vbfMSortedM"] = pad_val(vbfJets_sorted_mass.mass, 2, axis=1)
        
        vbfVars[f"vbfptSortedeta"] = pad_val(vbfJets_sorted_eta.pt, 2, axis=1)
        vbfVars[f"vbfetaSortedeta"] = pad_val(vbfJets_sorted_eta.eta, 2, axis=1)
        vbfVars[f"vbfphiSortedeta"] = pad_val(vbfJets_sorted_eta.phi, 2, axis=1)
        vbfVars[f"vbfMSortedeta"] = pad_val(vbfJets_sorted_eta.mass, 2, axis=1)
        
        vbfVars[f"nGoodVBFJetsUnsorted"] = n_good_vbf_jets.to_numpy() # the original one does not have jj cuts since it assumes no sorting.
        vbfVars[f"nGoodVBFJetsSortedpt"] = n_good_vbf_jets_sorted_pt.to_numpy()
        vbfVars[f"nGoodVBFJetsSortedM"] = n_good_vbf_jets_sorted_mass.to_numpy()
        vbfVars[f"nGoodVBFJetsSortedeta"] = n_good_vbf_jets_sorted_eta.to_numpy()
        
        # this is all very ugly but I just want it to work first. later we can decide to pack both jets together. also we can document better. Also we can copy the form of getdijet variables function. also we can implement correctons somehow. unsure of when they apply.
        
        
        return vbfVars
    
    
    def getVBFGenMatchCount(self, events: ak.Array,jets: ak.Array):
        """ Computes number of matching jets per event and returns this list. Was used for debugging"""
        # NOTE THIS IS ONLY FOR GENERATED SAMPLES THAT THIS WILL RUN. Will delete later or perhaps ammend to the GenSelections Script
        
        # for each sorting method, we can now compute delta R with the bb jet candidates and calculate if we got it right. 
        # We will generate a mask that will capture only the vbf jets that match with atleast one of the gen jets
        GEN_FLAGS = ["fromHardProcess", "isLastCopy"]
        vbfGenJets = events.GenPart[
        ((abs(events.GenPart.pdgId) == 24) ) * events.GenPart.hasFlags(GEN_FLAGS)
        ]
        vbfGenJets = events.GenPart[events.GenPart.hasFlags(["isHardProcess"])][:, 4:6]
        
                # Create pairs of jets from vbfJets_sorted_pt and two generator jets
        jet_pairs = ak.cartesian({"reco": jets, "gen": vbfGenJets[:,0:2]})

        # Calculate delta eta and delta phi for each pair
        delta_eta = jet_pairs["reco"].eta - jet_pairs["gen"].eta
        delta_phi = np.pi - np.abs(np.abs(jet_pairs["reco"].phi - jet_pairs["gen"].phi) - np.pi)

        # Calculate delta R for each pair
        delta_R = np.sqrt(delta_eta**2 + delta_phi**2)

        # Apply a mask for a low delta R value 
        mask_low_delta_R = delta_R < 0.4
        
        # Find events where at least one of the pairs of jets has a low delta R
        num_per_event= ak.sum(mask_low_delta_R, axis=-1)
        
        #logging.warning(f'Number of True values: {num_per_event}')
        for i in range(3):
            percentage = ak.sum(num_per_event == i) / len(num_per_event) * 100
            logging.warning(f"Percentage of events with {i} true values: {percentage:.2f}% {len(num_per_event)} and {ak.sum(num_per_event == i)}")
            
            
            
        # print kinematic properties of matches:
        # Apply the low delta R mask to the jet pairs
        matched_jet_pairs = jet_pairs[mask_low_delta_R]
        
        for i in range(3):
            # Select events with the specified number of matches
            event_mask = ak.sum(mask_low_delta_R, axis=-1) == i

            # For these events, get the corresponding reco and gen jets
            selected_pairs = matched_jet_pairs[event_mask]
            reco_jets = selected_pairs['reco']
            gen_jets = selected_pairs['gen']
            

            # Calculate and print the average and standard deviation for mass, pt, phi and eta of both reco and gen jets
            for name, jets in [("reco", reco_jets), ("gen", gen_jets)]:
                avg_mass = ak.mean(jets.mass, axis=None)
                std_mass = ak.std(jets.mass, axis=None)
                avg_pt = ak.mean(jets.pt, axis=None)
                std_pt = ak.std(jets.pt, axis=None)
                avg_phi = ak.mean(jets.phi, axis=None)
                std_phi = ak.std(jets.phi, axis=None)
                avg_eta = ak.mean(jets.eta, axis=None)
                std_eta = ak.std(jets.eta, axis=None)

                logging.warning(f"For {i} matches, {name} jet average mass: {avg_mass:.1f}, std: {std_mass:.1f}, average pt: {avg_pt:.1f}, std: {std_pt:.1f}, average phi: {avg_phi:.1f}, std: {std_phi:.1f}, average eta: {avg_eta:.1f}, std: {std_eta:.1f}")


        

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
