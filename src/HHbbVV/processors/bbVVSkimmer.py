"""
Skimmer for bbVV analysis.
Author(s): Raghav Kansal, Cristina Suarez
"""

import numpy as np
import awkward as ak
import pandas as pd

from coffea import processor
from coffea.analysis_tools import Weights, PackedSelection
from coffea.nanoevents.methods.nanoaod import JetArray
import vector

import pathlib
import pickle, json, gzip
import os

from typing import Dict
from collections import OrderedDict

from .GenSelection import gen_selection_HHbbVV, gen_selection_HH4V, gen_selection_HYbbVV, G_PDGID
from .TaggerInference import runInferenceTriton
from .utils import pad_val, add_selection, concatenate_dicts, select_dicts, P4
from .corrections import (
    add_pileup_weight,
    add_pileupid_weights,
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
        "Jet": P4,
        "GenHiggs": P4,
        "other": {"MET_pt": "MET_pt", "MET_phi": "MET_phi"},
    }

    preselection = {
        "pt": 300.0,
        "eta": 2.4,
        "VVmsd": 50,
        # "VVparticleNet_mass": [50, 250],
        # "bbparticleNet_mass": [92.5, 162.5],
        "bbparticleNet_mass": 50,
        "VVparticleNet_mass": 50,
        "bbFatJetParticleNetMD_Txbb": 0.8,
        "jetId": 2,  # tight ID bit
        "DijetMass": 800,  # TODO
        # "nGoodElectrons": 0,
    }

    ak4_jet_selection = {
        "pt": 25,
        "eta": 2.7,
        "jetId": "tight",
        "puId": "medium",
        "dR_fatjetbb": 1.2,
        "dR_fatjetVV": 0.8,
    }

    jecs = common.jecs

    # only the branches necessary for templates and post processing
    min_branches = [
        "ak8FatJetPhi",
        "ak8FatJetEta",
        "ak8FatJetPt",
        "ak8FatJetMsd",
        "ak8FatJetParticleNetMass",
        "DijetMass",
        "VBFJetEta",
        "VBFJetPt",
        "VBFJetPhi",
        "VBFJetMass",
        "ak8FatJetHbb",
        "ak8FatJetHVV",
        "ak8FatJetHVVNumProngs",
        "ak8FatJetParticleNetMD_Txbb",
        "VVFatJetParTMD_THWWvsT",
        "MET_pt",
        "MET_phi",
        "nGoodElectrons",
        "nGoodMuons",
        "genWW",
        "genZZ",
    ]

    for shift in jec_shifts:
        min_branches.append(f"ak8FatJetPt_{shift}")
        min_branches.append(f"DijetMass_{shift}")
        min_branches.append(f"VBFJetPt_{shift}")

    for shift in jmsr_shifts:
        min_branches.append(f"ak8FatJetParticleNetMass_{shift}")
        min_branches.append(f"DijetMass_{shift}")

    def __init__(
        self,
        xsecs={},
        save_ak15=False,
        save_systematics=True,
        inference=True,
        save_all=False,
        vbf_search=True,
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

        # search for VBF production events
        self._vbf_search = vbf_search

        # for tagger model and preprocessing dict
        self.tagger_resources_path = (
            str(pathlib.Path(__file__).parent.resolve()) + "/tagger_resources/"
        )

        # MET filters
        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        package_path = str(pathlib.Path(__file__).parent.parent.resolve())
        with open(package_path + "/data/metfilters.json", "rb") as filehandler:
            self.metfilters = json.load(filehandler)

        self._accumulator = processor.dict_accumulator({})

        logger.info(
            f"Running skimmer with inference {self._inference} and systematics {self._systematics} and save all {self._save_all} and VBF search {self._vbf_search}"
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
        num_ak4_jets = 2
        fatjets, jec_shifted_vars = get_jec_jets(events, year, isData, self.jecs, fatjets=True)

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
                if (
                    "GenHiggsChildren" in vars_dict.keys()
                ):  # Only HY samples which are WW by default will not have this
                    data = vars_dict["GenHiggsChildren"]
                    skimmed_events["genWW"] = np.any(data == 24, axis=1) # true if WW false if ZZ (It must be one of the two.)
                    skimmed_events["genZZ"] = np.any(data == 23, axis=1) # maybe we can make this one mask since the two are disjoint
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

        # VBF ak4 jet vars

        ak4_jet_vars = {}

        jets, _ = get_jec_jets(events, year, isData, self.jecs, fatjets=False)

        # dR_fatjetVV = 0.8 used from last two cells of VBFgenInfoTests.ipynb with data generated from SM signal vbf
        # https://github.com/rkansal47/HHbbVV/blob/vbf_systematics/src/HHbbVV/VBF_binder/VBFgenInfoTests.ipynb
        # (0-14R1R2study.parquet) has columns of different nGoodVBFJets corresponding to R1 and R2 cuts
        vbf_jet_mask = (
            jets.isTight
            & (jets.pt > self.ak4_jet_selection["pt"])
            & (np.abs(jets.eta) < 4.7)
            # medium puId https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetIDUL
            & ((jets.pt > 50) | ((jets.puId & 2) == 2))
            & (
                ak.all(
                    jets.metric_table(
                        ak.singletons(ak.pad_none(fatjets, num_jets, axis=1, clip=True)[bb_mask])
                    )
                    > self.ak4_jet_selection["dR_fatjetbb"],
                    axis=-1,
                )
            )
            & (
                ak.all(
                    jets.metric_table(
                        ak.singletons(ak.pad_none(fatjets, num_jets, axis=1, clip=True)[~bb_mask])
                    )
                    > self.ak4_jet_selection["dR_fatjetVV"],
                    axis=-1,
                )
            )
        )

        vbf_jets = jets[vbf_jet_mask]

        VBFJetVars = {
            f"VBFJet{key}": pad_val(vbf_jets[var], num_ak4_jets, axis=1)
            for (var, key) in self.skim_vars["Jet"].items()
        }

        # JEC vars
        if not isData:
            for var in ["pt"]:
                key = self.skim_vars["Jet"][var]
                for label, shift in self.jecs.items():
                    for vari in ["up", "down"]:
                        VBFJetVars[f"VBFJet{key}_{label}_{vari}"] = pad_val(
                            vbf_jets[shift][vari][var], num_ak4_jets, axis=1
                        )

        skimmed_events["nGoodVBFJets"] = np.array(ak.sum(vbf_jet_mask, axis=1))

        # VBF ak4 Jet vars (pt, eta, phi, M, nGoodJets)
        # if self._vbf_search:
        #     isGen = "VBF_HHTobbVV" in dataset
        #     ak4JetVars = {
        #         **self.getVBFVars(events, jets, ak8FatJetVars, bb_mask, isGen=isGen)
        #     }
        #     skimmed_events = {**skimmed_events, **ak4JetVars}

        otherVars = {
            key: events[var.split("_")[0]]["_".join(var.split("_")[1:])].to_numpy()
            for (var, key) in self.skim_vars["other"].items()
        }

        skimmed_events = {**skimmed_events, **ak8FatJetVars, **VBFJetVars, **dijetVars, **otherVars}

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
                    * (np.abs(fatjets.eta) < self.preselection["eta"])
                    # tight ID
                    * (fatjets.jetId & self.preselection["jetId"] == self.preselection["jetId"]),
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
                cut = (pnetms[~bb_mask] >= self.preselection["VVparticleNet_mass"]) * (
                    pnetms[bb_mask] >= self.preselection["bbparticleNet_mass"]
                )
                # cut = (
                #     (pnetms[~bb_mask] >= self.preselection["VVparticleNet_mass"][0])
                #     * (pnetms[~bb_mask] < self.preselection["VVparticleNet_mass"][1])
                #     * (pnetms[bb_mask] >= self.preselection["bbparticleNet_mass"][0])
                #     * (pnetms[bb_mask] < self.preselection["bbparticleNet_mass"][1])
                # )

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
            check_fatjets = events.FatJet[:, :2]
            hem_cleaning = (
                ((events.run >= 319077) & isData)  # if data check if in Runs C or D
                # else for MC randomly cut based on lumi fraction of C&D
                | ((np.random.rand(len(events)) < 0.632) & ~isData)
            ) & (
                ak.any(
                    (
                        (check_fatjets.pt > 30.0)
                        & (check_fatjets.eta > -3.2)
                        & (check_fatjets.eta < -1.3)
                        & (check_fatjets.phi > -1.57)
                        & (check_fatjets.phi < -0.87)
                    ),
                    -1,
                )
                | ak.any(
                    (
                        (vbf_jets.eta > -3.2)
                        & (vbf_jets.eta < -1.3)
                        & (vbf_jets.phi > -1.57)
                        & (vbf_jets.phi < -0.87)
                    ),
                    -1,
                )
                | ((events.MET.phi > -1.62) & (events.MET.pt < 470.0) & (events.MET.phi < -0.62))
            )

            add_selection("hem_cleaning", ~np.array(hem_cleaning).astype(bool), *selection_args)

        # MET filters

        metfilters = np.ones(len(events), dtype="bool")
        metfilterkey = "data" if isData else "mc"
        for mf in self.metfilters[year][metfilterkey]:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]

        add_selection("met_filters", metfilters, *selection_args)

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

        # selection from https://github.com/jennetd/hbb-coffea/blob/85bc3692be9e0e0a0c82ae3c78e22cdf5b3e4d68/boostedhiggs/vhbbprocessor.py#L283-L307
        # https://indico.cern.ch/event/1154430/#b-471403-higgs-meeting-special

        goodelectron = (
            (events.Electron.pt > 20)
            & (abs(events.Electron.eta) < 2.5)
            & (events.Electron.miniPFRelIso_all < 0.4)
            & (events.Electron.cutBased >= events.Electron.LOOSE)
        )
        nelectrons = ak.sum(goodelectron, axis=1)

        goodmuon = (
            (events.Muon.pt > 15)
            & (abs(events.Muon.eta) < 2.4)
            & (events.Muon.miniPFRelIso_all < 0.4)
            & events.Muon.looseId
        )
        nmuons = ak.sum(goodmuon, axis=1)

        skimmed_events["nGoodMuons"] = nmuons.to_numpy()
        skimmed_events["nGoodElectrons"] = nelectrons.to_numpy()

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
            add_pileupid_weights(weights, year, vbf_jets, events.GenJet, wp="M")  # this gives error
            add_VJets_kFactors(weights, events.GenPart, dataset)

            # if dataset.startswith("TTTo"):
            #     # TODO: need to add uncertainties and rescale yields (?)
            #     add_top_pt_weight(weights, events)

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

            systematics = ["", "notrigeffs"]

            if self._systematics:
                systematics += list(weights.variations)

            single_weight_pileup = weights.partial_weight(["single_weight_pileup"])
            add_selection("single_weight_pileup", (single_weight_pileup <= 4), *selection_args)

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

        if isSignal and self._systematics:
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

    def getVBFVars(
        self,
        events: ak.Array,
        jets: JetArray,
        ak8FatJetVars: Dict,
        bb_mask: np.ndarray,
        isGen: bool = False,
        skim_vars: dict = None,
        pt_shift: str = None,
        mass_shift: str = None,
    ) -> Dict:
        """
        Computes selections on VBF jet candidates based on B2G-22-003.
        Sorts jets by pt after applying base cuts and fatjet exclusion.
        Stores nGoodVBFJets, an int list encoding cuts, and kinematic variables of VBF jet candidates.

        TODO:
            - Implement the use of pt_shift, mass_shift, skim_vars.
            - Study selection parameters that provide the best sensitivity.
            - Decide best way to compute vbf variables (vector or by hand)

        Args:
            events (ak.Array): Event array.
            jets (JetArray): Array of ak4 jets **with JECs already applied**.
            ak8FatJetVars (Dict): AK8 Fat Jet variables.
            bb_mask (np.ndarray): BB mask array.
            isGen (bool, optional): Flag for generation-level. Defaults to False.
            skim_vars (dict, optional): Skim variables. Defaults to None.
            pt_shift (str, optional): PT shift. Defaults to None.
            mass_shift (str, optional): Mass shift. Defaults to None.

        Returns:
            Dict: VBF variables dictionary.

        """
        vbfVars = {}

        # AK4 jets definition: 5.4 B2G-22-003
        ak4_jet_mask = (
            jets.isTight
            & (jets.pt > self.ak4_jet_selection["pt"])
            & (np.abs(jets.eta) < 4.7)
            # medium puId https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetIDUL
            & ((jets.pt > 50) | ((jets.puId & 2) == 2))
        )

        # VBF selections: 7.1.4 B2G-22-003

        # Mask for electron/muon overlap
        # electrons, muons = events.Electron[events.Electron.pt < 5], events.Muon[events.Muon.pt < 7]
        # e_pairs = ak.cartesian([jets, electrons], nested=True)
        # e_pairs_mask = np.abs(e_pairs.slot0.delta_r(e_pairs.slot1)) < 0.4
        # m_pairs = ak.cartesian([jets, muons], nested=True)
        # m_pairs_mask = np.abs(m_pairs.slot0.delta_r(m_pairs.slot1)) < 0.4

        # electron_muon_overlap_mask = ~(
        #     ak.any(e_pairs_mask, axis=-1) | ak.any(m_pairs_mask, axis=-1)
        # )

        # Fatjet Definition for ∆R calculations: same definition as in getDijetVars() (not included so we can study its effects in the output.)
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

        # this might not work due to types
        fatjet_overlap_mask = (np.abs(jets.delta_r(bbJet)) > 1.2) & (
            np.abs(jets.delta_r(VVJet)) > 0.8
        )

        # Apply base masks, sort, and calculate vbf dijet (jj) cuts
        vbfJets_mask = ak4_jet_mask  # & electron_muon_overlap_mask & fatjet_overlap_mask
        vbfJets = jets[vbfJets_mask]

        vbfJets_sorted_pt = vbfJets[ak.argsort(vbfJets.pt, ascending=False)]
        # this is the only which does not guarantee two guys. in the other sorts, the entries are specifically None.
        vbfJets_sorted_pt = ak.pad_none(vbfJets_sorted_pt, 2, clip=True)

        # pt sorted eta and dijet mass mask
        vbf1 = vector.array(
            {
                "pt": ak.flatten(pad_val(vbfJets_sorted_pt[:, 0:1].pt, 1, axis=1)),
                "phi": ak.flatten(pad_val(vbfJets_sorted_pt[:, 0:1].phi, 1, axis=1)),
                "eta": ak.flatten(pad_val(vbfJets_sorted_pt[:, 0:1].eta, 1, axis=1)),
                "M": ak.flatten(pad_val(vbfJets_sorted_pt[:, 0:1].mass, 1, axis=1)),
            }
        )

        vbf2 = vector.array(
            {
                "pt": ak.flatten(pad_val(vbfJets_sorted_pt[:, 1:2].pt, 1, axis=1)),
                "phi": ak.flatten(pad_val(vbfJets_sorted_pt[:, 1:2].phi, 1, axis=1)),
                "eta": ak.flatten(pad_val(vbfJets_sorted_pt[:, 1:2].eta, 1, axis=1)),
                "M": ak.flatten(pad_val(vbfJets_sorted_pt[:, 1:2].mass, 1, axis=1)),
            }
        )

        jj = vbf1 + vbf2

        mass_jj_cut_sorted_pt = jj.mass > 500
        eta_jj_cut_sorted_pt = np.abs(vbf1.eta - vbf2.eta) > 4.0

        # uncomment these last two to include dijet cuts
        vbfJets_mask_sorted_pt = vbfJets_mask  # * mass_jj_cut_sorted_pt * eta_jj_cut_sorted_pt
        n_good_vbf_jets_sorted_pt = ak.fill_none(ak.sum(vbfJets_mask_sorted_pt, axis=1), 0)

        # add vbf gen quark info
        if isGen:  # add | True when debugging with local files
            vbfGenJets = events.GenPart[events.GenPart.hasFlags(["isHardProcess"])][:, 4:6]

            vbfVars[f"vbfptGen"] = pad_val(vbfGenJets.pt, 2, axis=1)
            vbfVars[f"vbfetaGen"] = pad_val(vbfGenJets.eta, 2, axis=1)
            vbfVars[f"vbfphiGen"] = pad_val(vbfGenJets.phi, 2, axis=1)
            vbfVars[f"vbfMGen"] = pad_val(vbfGenJets.mass, 2, axis=1)

            jet_pairs = ak.cartesian({"reco": vbfJets_sorted_pt[:, 0:2], "gen": vbfGenJets[:, 0:2]})

            # Calculate delta eta and delta phi for each pair
            delta_eta = jet_pairs["reco"].eta - jet_pairs["gen"].eta
            delta_phi = np.pi - np.abs(np.abs(jet_pairs["reco"].phi - jet_pairs["gen"].phi) - np.pi)

            # Calculate delta R for each pair
            delta_R = np.sqrt(delta_eta**2 + delta_phi**2)

            # Apply a mask for a low delta R value
            mask_low_delta_R = delta_R < 0.4
            num_per_event = ak.sum(mask_low_delta_R, axis=-1)  # miscounts 0's since some are empty

            # Combine masks with logical 'and' operation
            total_mask = n_good_vbf_jets_sorted_pt > 1

            # set event that fail to have 0 for num of events.
            num_per_event = ak.where(total_mask, num_per_event, 0)
            vbfVars[f"vbfNumMatchedGen"] = num_per_event.to_numpy()

            # adds data about R1 R2 selection efficiencies.
            graphingR1R2 = False
            if (
                graphingR1R2 == True
            ):  # compute fatjet mask many times, execute dijet selection and associated cuts. add to datafram label with R1 R2 info in it and ngoodjets for this configuration
                for R1 in np.arange(0.3, 2, 0.1):
                    for R2 in np.arange(0.3, 2, 0.1):
                        fatjet_overlap_mask = (np.abs(jets.delta_r(bbJet)) > R1) & (
                            np.abs(jets.delta_r(VVJet)) > R2
                        )

                        # compute n_good_vbf_jets + incorporate eta_jj > 4.0
                        vbfJets_mask = (
                            ak4_jet_mask  # & electron_muon_overlap_mask & fatjet_overlap_mask
                        )

                        # vbfJets_mask = fatjet_overlap_mask # this is for unflitered events
                        vbfJets = jets[vbfJets_mask]
                        vbfJets_sorted_pt = vbfJets[ak.argsort(vbfJets.pt, ascending=False)]
                        vbfJets_sorted_pt = ak.pad_none(vbfJets_sorted_pt, 2, clip=True)
                        jj_sorted_pt = vbfJets_sorted_pt[:, 0:1] + vbfJets_sorted_pt[:, 1:2]
                        mass_jj_cut_sorted_pt = jj_sorted_pt.mass > 500
                        eta_jj_cut_sorted_pt = (
                            np.abs(vbfJets_sorted_pt[:, 0:1].eta - vbfJets_sorted_pt[:, 1:2].eta)
                            > 4.0
                        )
                        vbfJets_mask_sorted_pt = (
                            vbfJets_mask * mass_jj_cut_sorted_pt * eta_jj_cut_sorted_pt
                        )
                        num_sorted_pt = ak.fill_none(ak.sum(vbfJets_mask_sorted_pt, axis=1), 0)

                        # here we can print some information about the jets so that we can study the selections a bit.
                        # jet_pairs = ak.cartesian({"reco": vbfJets, "gen": vbfGenJets[:,0:2]}) # this is only for unfiltered events
                        jet_pairs = ak.cartesian(
                            {"reco": vbfJets_sorted_pt[:, 0:2], "gen": vbfGenJets[:, 0:2]}
                        )

                        # Calculate delta eta and delta phi for each pair
                        delta_eta = jet_pairs["reco"].eta - jet_pairs["gen"].eta
                        delta_phi = np.pi - np.abs(
                            np.abs(jet_pairs["reco"].phi - jet_pairs["gen"].phi) - np.pi
                        )

                        # Calculate delta R for each pair
                        delta_R = np.sqrt(delta_eta**2 + delta_phi**2)

                        # Apply a mask for a low delta R value
                        mask_low_delta_R = delta_R < 0.4
                        num_per_event = ak.sum(
                            mask_low_delta_R, axis=-1
                        )  # miscounts 0's since some are empty

                        # Combine masks with logical 'and' operation
                        total_mask = num_sorted_pt > 1

                        # set event that fail to have 0 for num of events.
                        num_per_event = ak.where(total_mask, num_per_event, 0)
                        vbfVars[f"vbfR1{R1:.1f}R2{R2:.1f}"] = num_per_event.to_numpy()

                        # note later when we have the pd dataframe, we need to apply mask, then turn data into graph.
                        # and df = df[selection_mask & (df[('nGoodMuons', 0)] == 0) & (df[('nGoodElectrons', 0)] == 0) & (df[('nGoodVBFJets', 0)] >= 1)& (df[('nGoodJets', 0)] == 0)]

        vbfVars[f"vbfpt"] = pad_val(vbfJets_sorted_pt.pt, 2, axis=1)
        vbfVars[f"vbfeta"] = pad_val(vbfJets_sorted_pt.eta, 2, axis=1)
        vbfVars[f"vbfphi"] = pad_val(vbfJets_sorted_pt.phi, 2, axis=1)
        vbfVars[f"vbfM"] = pad_val(vbfJets_sorted_pt.mass, 2, axis=1)

        # int list representing the number of passing vbf jets per event
        vbfVars[f"nGoodVBFJets"] = n_good_vbf_jets_sorted_pt.to_numpy()

        adding_bdt_vars = True
        if adding_bdt_vars == True:
            # Adapted from HIG-20-005 ggF_Killer 6.2.2
            # https://coffeateam.github.io/coffea/api/coffea.nanoevents.methods.vector.PtEtaPhiMLorentzVector.html
            # https://coffeateam.github.io/coffea/api/coffea.nanoevents.methods.vector.LorentzVector.html
            # Adding variables defined in HIG-20-005 that show strong differentiation for VBF signal events and background

            # seperation between both ak8 higgs jets
            vbfVars[f"vbf_dR_HH"] = VVJet.deltaR(bbJet)

            vbfVars[f"vbf_dR_j0_HVV"] = vbf1.deltaR(VVJet)
            vbfVars[f"vbf_dR_j1_HVV"] = vbf2.deltaR(VVJet)
            vbfVars[f"vbf_dR_j0_Hbb"] = vbf1.deltaR(bbJet)
            vbfVars[f"vbf_dR_j1_Hbb"] = vbf2.deltaR(bbJet)
            vbfVars[f"vbf_dR_jj"] = vbf1.deltaR(vbf2)
            vbfVars[f"vbf_Mass_jj"] = jj.M
            vbfVars[f"vbf_dEta_jj"] = np.abs(vbf1.eta - vbf2.eta)

            # Subleading VBF-jet cos(θ) in the HH+2j center of mass frame:
            # https://github.com/scikit-hep/vector/blob/main/src/vector/_methods.py#L916
            system_4vec = vbf1 + vbf2 + VVJet + bbJet
            j1_CMF = vbf1.boostCM_of_p4(system_4vec)

            # Leading VBF-jet cos(θ) in the HH+2j center of mass frame:
            thetab1 = 2 * np.arctan(np.exp(-j1_CMF.eta))
            thetab1 = np.cos(thetab1)  # 12
            vbfVars[f"vbf_cos_j1"] = np.abs(thetab1)

            # Subleading VBF-jet cos(θ) in the HH+2j center of mass frame:
            j2_CMF = vbf2.boostCM_of_p4(system_4vec)
            thetab2 = 2 * np.arctan(np.exp(-j2_CMF.eta))
            thetab2 = np.cos(thetab2)
            vbfVars[f"vbf_cos_j2"] = np.abs(thetab2)

            # H1-centrality * H2-centrality:
            delta_eta = vbf1.eta - vbf2.eta
            avg_eta = (vbf1.eta + vbf2.eta) / 2
            prod_centrality = np.exp(
                -np.power((VVJet.eta - avg_eta) / delta_eta, 2)
                - np.power((bbJet.eta - avg_eta) / delta_eta, 2)
            )
            vbfVars[f"vbf_prod_centrality"] = prod_centrality

        return vbfVars

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
                "M": ak8FatJetVars[f"ak8FatJetParticleNetMass{mlabel}"][~bb_mask],
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
