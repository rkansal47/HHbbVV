"""
Skimmer for bbVV analysis.
Author(s): Raghav Kansal, Cristina Suarez
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path

import awkward as ak
import numpy as np
import vector
from coffea import processor
from coffea.analysis_tools import PackedSelection

import HHbbVV
from HHbbVV import hh_vars
from HHbbVV.hh_vars import jec_shifts, jmsr_shifts

from . import utils
from .common import HLTs
from .corrections import (
    add_lepton_id_weights,
    add_pileup_weight,
    add_pileupid_weights,
    add_ps_weight,
    add_trig_effs,
    add_VJets_kFactors,
    get_jec_jets,
    get_jmsr,
    get_lund_SFs,
    get_pdf_weights,
    get_scale_weights,
)
from .GenSelection import (
    gen_selection_HH4V,
    gen_selection_HHbbVV,
    gen_selection_HYbbVV,
)
from .SkimmerABC import SkimmerABC
from .TaggerInference import runInferenceTriton
from .utils import P4, Weights, add_selection, concatenate_dicts, pad_val

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

logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("bbVVSkimmer")
# logger.setLevel(logging.INFO)


class bbVVSkimmer(SkimmerABC):
    """
    Skims nanoaod files, saving selected branches and events passing preselection cuts
    (and triggers for data), for preliminary cut-based analysis and BDT studies.

    Args:
        xsecs (dict, optional): sample cross sections,
          if sample not included no lumi and xsec will not be applied to weights
        save_ak15 (bool, optional): save ak15 jets as well, for HVV candidate
    """

    # key is name in nano files, value will be the name in the skimmed output
    skim_vars = {  # noqa: RUF012
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

    preselection = {  # noqa: RUF012
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

    ak4_jet_selection = {  # noqa: RUF012
        "pt": 25,
        "eta": 2.7,
        "jetId": "tight",
        "puId": "medium",
        "dR_fatjetbb": 1.2,
        "dR_fatjetVV": 0.8,
    }

    jecs = hh_vars.jecs

    # only the branches necessary for templates and post processing
    min_branches = [  # noqa: RUF012
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
        "nGoodVBFJets",
        "ak8FatJetHbb",
        "ak8FatJetHVV",
        "ak8FatJetHVVNumProngs",
        "ak8FatJetParticleNetMD_Txbb",
        "VVFatJetParTMD_THWWvsT",
        "MET_pt",
        "MET_phi",
        "nGoodElectronsHH",
        "nGoodElectronsHbb",
        "nGoodMuonsHH",
        "nGoodMuonsHbb",
        "ak8FatJetNumWTagged",
        "ak8FatJetLowestWTaggedTxbb",
        "ak8FatJetWTaggedMsd",
        "ak8FatJetWTaggedParticleNetMass",
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
        xsecs=None,
        save_ak15=False,
        save_systematics=True,
        lp_sfs=True,
        inference=True,
        save_all=False,
    ):
        if xsecs is None:
            xsecs = {}
        super().__init__()

        self.XSECS = xsecs  # in pb
        self.save_ak15 = save_ak15

        # save systematic variations
        self._systematics = save_systematics

        # save LP SFs
        self._lp_sfs = lp_sfs

        # run inference
        self._inference = inference

        # save all branches or only necessary ones
        self._save_all = save_all

        # for tagger model and preprocessing dict
        self.tagger_resources_path = Path(__file__).parent.resolve() / "tagger_resources"

        # MET filters
        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        package_path = Path(__file__).parent.parent.resolve()
        with (package_path / "data/metfilters.json").open("rb") as filehandler:
            self.metfilters = json.load(filehandler)

        self._accumulator = processor.dict_accumulator({})

        logging.info(
            f"Running skimmer with inference {self._inference} and systematics {self._systematics} and save all {self._save_all}."
        )

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events: ak.Array):
        """Runs event processor for different types of jets"""

        year = events.metadata["dataset"].split("_")[0]
        year_nosuffix = year.replace("APV", "")
        dataset = "_".join(events.metadata["dataset"].split("_")[1:])

        isData = "JetHT" in dataset
        isSignal = (
            "GluGluToHHTobbVV" in dataset
            or "XToYHTo2W2BTo4Q2B" in dataset
            or "VBF_HHTobbVV" in dataset
        )

        if not isData:
            # remove events with pileup weights un-physically large
            events = self.pileup_cutoff(events, year, cutoff=4)

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

        cutflow = OrderedDict()
        cutflow["all"] = n_events

        selection_args = (selection, cutflow, isData, gen_weights)

        num_jets = 2
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
                skimmed_events = {**skimmed_events, **vars_dict}

        # used for normalization to cross section below
        gen_selected = (
            selection.all(*selection.names)
            if len(selection.names)
            else np.ones(len(events)).astype(bool)
        )
        logging.info(f"Passing gen selection: {np.sum(gen_selected)} / {len(events)}")

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
                    logging.warning(f"Missing HLT {trigger}!")

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
                ak.fill_none(
                    ak.any(
                        (
                            (check_fatjets.pt > 30.0)
                            & (check_fatjets.eta > -3.2)
                            & (check_fatjets.eta < -1.3)
                            & (check_fatjets.phi > -1.57)
                            & (check_fatjets.phi < -0.87)
                        ),
                        -1,
                    ),
                    False,
                )
                | ak.fill_none(
                    ak.any(
                        (
                            (vbf_jets.eta > -3.2)
                            & (vbf_jets.eta < -1.3)
                            & (vbf_jets.phi > -1.57)
                            & (vbf_jets.phi < -0.87)
                        ),
                        -1,
                    ),
                    False,
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

        # VBF Hbb selection from https://github.com/jennetd/hbb-coffea/blob/85bc3692be9e0e0a0c82ae3c78e22cdf5b3e4d68/boostedhiggs/vhbbprocessor.py#L283-L307
        # https://indico.cern.ch/event/1154430/#b-471403-higgs-meeting-special

        goodelectronHbb = (
            (electrons.pt > 20)
            & (abs(electrons.eta) < 2.5)
            & (electrons.miniPFRelIso_all < 0.4)
            & (electrons.cutBased >= electrons.LOOSE)
        )
        nelectronsHbb = ak.sum(goodelectronHbb, axis=1)
        goodelectronsHbb = electrons[goodelectronHbb]

        goodmuonHbb = (
            (muons.pt > 10) & (abs(muons.eta) < 2.4) & (muons.pfRelIso04_all < 0.25) & muons.looseId
        )
        nmuonsHbb = ak.sum(goodmuonHbb, axis=1)
        goodmuonsHbb = muons[goodmuonHbb]

        # HH4b lepton vetoes:
        # https://cms.cern.ch/iCMS/user/noteinfo?cmsnoteid=CMS%20AN-2020/231 Section 7.1.2
        # In order to be considered in the lepton veto step, a muon (electron) is required to to pass the selections described in Section 5.2, and to have pT > 15 GeV (pT > 20 GeV), and |η| < 2.4 (2.5).
        # A muon is also required to pass loose identification criteria as detailed in [35] and mini-isolation
        # (miniPFRelIso all < 0.4). An electron is required to pass mvaFall17V2noIso WP90 identification as well as mini-isolation (miniPFRelIso all < 0.4).

        goodelectronHH = (
            (electrons.pt > 20)
            & (abs(electrons.eta) < 2.5)
            & (electrons.miniPFRelIso_all < 0.4)
            & (electrons.mvaFall17V2noIso_WP90)
        )
        nelectronsHH = ak.sum(goodelectronHH, axis=1)
        goodelectronsHH = electrons[goodelectronHH]

        goodmuonHH = (
            (muons.pt > 15)
            & (abs(muons.eta) < 2.4)
            & (muons.miniPFRelIso_all < 0.4)
            & muons.looseId
        )
        nmuonsHH = ak.sum(goodmuonHH, axis=1)
        goodmuonsHH = muons[goodmuonHH]

        skimmed_events["nGoodElectronsHbb"] = nelectronsHbb.to_numpy()
        skimmed_events["nGoodElectronsHH"] = nelectronsHH.to_numpy()
        skimmed_events["nGoodMuonsHbb"] = nmuonsHbb.to_numpy()
        skimmed_events["nGoodMuonsHH"] = nmuonsHH.to_numpy()

        # XHY->bbWW semi-resolved channel veto
        Wqq_score = (fatjets.particleNetMD_Xqq + fatjets.particleNetMD_Xcc) / (
            fatjets.particleNetMD_Xqq + fatjets.particleNetMD_Xcc + fatjets.particleNetMD_QCD
        )

        skimmed_events["ak8FatJetNumWTagged"] = ak.sum(Wqq_score[:, :3] >= 0.8, axis=1).to_numpy()

        sorted_wqq_score = np.argsort(pad_val(Wqq_score, 3, 0, 1), axis=1)

        # get TXbb score of the lowest-Wqq-tagged jet
        skimmed_events["ak8FatJetLowestWTaggedTxbb"] = pad_val(fatjets["Txbb"], 3, 0, 1)[
            np.arange(len(fatjets)), sorted_wqq_score[:, 0]
        ]

        # save both SD and regressed masses of the two W-tagged AK8 jets
        # Amitav will optimize mass cut soon

        mass_dict = {"particleNet_mass": "ParticleNetMass", "msoftdrop": "Msd"}
        for mkey, mlabel in mass_dict.items():
            mass_vals = pad_val(fatjets[mkey], 3, 0, 1)
            mv_fj1 = mass_vals[np.arange(len(fatjets)), sorted_wqq_score[:, 2]]
            mv_fj2 = mass_vals[np.arange(len(fatjets)), sorted_wqq_score[:, 1]]
            skimmed_events[f"ak8FatJetWTagged{mlabel}"] = np.stack([mv_fj1, mv_fj2]).T

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

        totals_dict = {"nevents": n_events}

        if isData:
            skimmed_events["weight"] = np.ones(n_events)
        else:
            weights_dict, totals_temp = self.add_weights(
                events,
                year,
                dataset,
                gen_weights,
                fatjets,
                num_jets,
                vbf_jets,
                gen_selected,
                goodelectronsHbb,
                goodelectronsHH,
                goodmuonsHbb,
                goodmuonsHH,
            )
            skimmed_events = {**skimmed_events, **weights_dict}
            totals_dict = {**totals_dict, **totals_temp}

        ##############################
        # Reshape and apply selections
        ##############################

        sel_all = selection.all(*selection.names)

        skimmed_events = {
            key: value.reshape(len(skimmed_events["weight"]), -1)[sel_all]
            for (key, value) in skimmed_events.items()
        }

        bb_mask = bb_mask[sel_all]

        ################
        # Lund plane SFs
        ################

        if isSignal and self._systematics and self._lp_sfs:
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
                            selected_sfs[key] = get_lund_SFs(
                                year,
                                events[sel_all][selector],
                                (
                                    i
                                    if self._save_all
                                    else skimmed_events["ak8FatJetHVV"][selector][:, 1]
                                ),  # giving HVV jet index if only doing LP SFs for HVV jet
                                fatjets[sel_all][selector],
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
                logging.info("No signal events selected")
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

        pddf = self.to_pandas(skimmed_events)
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
        self.dump_table(pddf, fname)

        return {year: {dataset: {"totals": totals_dict, "cutflow": cutflow}}}

    def postprocess(self, accumulator):
        return accumulator

    def add_weights(
        self,
        events,
        year,
        dataset,
        gen_weights,
        fatjets,
        num_jets,
        vbf_jets,
        gen_selected,
        goodelectronsHbb,
        goodelectronsHH,
        goodmuonsHbb,
        goodmuonsHH,
    ) -> tuple[dict, dict]:
        """Adds weights and variations, saves totals for all norm preserving weights and variations"""
        weights = Weights(len(events), storeIndividual=True)
        weights.add("genweight", gen_weights)

        add_pileup_weight(weights, year, events.Pileup.nPU.to_numpy())
        add_pileupid_weights(weights, year, vbf_jets, events.GenJet, wp="M")  # needs awkward 1.10
        add_VJets_kFactors(weights, events.GenPart, dataset)
        add_ps_weight(weights, events.PSWeight)
        add_trig_effs(weights, fatjets, year, num_jets)

        if year in ("2016APV", "2016", "2017"):
            weights.add(
                "L1EcalPrefiring",
                events.L1PreFiringWeight.Nom,
                events.L1PreFiringWeight.Up,
                events.L1PreFiringWeight.Dn,
            )

        logging.debug("weights ", weights._weights.keys())

        ###################### Save all the weights and variations ######################

        # these weights should not change the overall normalization, so are saved separately
        norm_preserving_weights = HHbbVV.hh_vars.norm_preserving_weights

        # dictionary of all weights and variations
        weights_dict = {}
        # dictionary of total # events for norm preserving variations for normalization in postprocessing
        totals_dict = {}

        # nominal
        weights_dict["weight"] = weights.weight()

        # without trigger efficiencies (in case they need to be revised later)
        weights_dict["weight_noTrigEffs"] = weights.partial_weight(exclude=["trig_effs"])

        # norm preserving weights, used to do normalization in post-processing
        weight_np = weights.partial_weight(include=norm_preserving_weights)
        totals_dict["np_nominal"] = np.sum(weight_np[gen_selected])

        # variations
        if self._systematics:
            for systematic in list(weights.variations):
                weights_dict[f"weight_{systematic}"] = weights.weight(modifier=systematic)

                if utils.remove_variation_suffix(systematic) in norm_preserving_weights:
                    var_weight = weights.partial_weight(
                        include=norm_preserving_weights, modifier=systematic
                    )

                    # need to save total # events for each variation for normalization in post-processing
                    totals_dict[f"np_{systematic}"] = np.sum(var_weight[gen_selected])

        ###################### alpha_S and PDF variations ######################

        # alpha_s variations, only for HH and ttbar
        if "GluGluToHHTobbVV" in dataset or "VBF_HHTobbVV" in dataset or dataset.startswith("TTTo"):
            scale_weights = get_scale_weights(events)
            weights_dict["scale_weights"] = scale_weights * weights_dict["weight"][:, np.newaxis]
            totals_dict["np_scale_weights"] = np.sum(
                (scale_weights * weight_np[:, np.newaxis])[gen_selected], axis=0
            )

        if "GluGluToHHTobbVV" in dataset or "VBF_HHTobbVV" in dataset:
            pdf_weights = get_pdf_weights(events)
            weights_dict["pdf_weights"] = pdf_weights * weights_dict["weight"][:, np.newaxis]
            totals_dict["np_pdf_weights"] = np.sum(
                (pdf_weights * weight_np[:, np.newaxis])[gen_selected], axis=0
            )

        ###################### Normalization (Step 1) ######################

        weight_norm = self.get_dataset_norm(year, dataset)
        # normalize all the weights to xsec, needs to be divided by totals in Step 2 in post-processing
        for key, val in weights_dict.items():
            weights_dict[key] = val * weight_norm

        # save the unnormalized weight, to confirm that it's been normalized in post-processing
        weights_dict["weight_noxsec"] = weights.weight()
        # save pileup weight for debugging
        weights_dict["single_weight_pileup"] = weights.partial_weight(include=["pileup"])

        ###################### Separate Lepton ID Scale Factors ######################

        # saved separately for now TODO: incorporate above next time if lepton vetoes are useful
        lepton_weights = Weights(len(events), storeIndividual=True)
        add_lepton_id_weights(
            lepton_weights, year, goodelectronsHbb, "electron", "Loose", label="_hbb"
        )
        add_lepton_id_weights(
            lepton_weights, year, goodelectronsHH, "electron", "wp90noiso", label="_hh"
        )
        add_lepton_id_weights(lepton_weights, year, goodmuonsHbb, "muon", "Loose", label="_hbb")
        add_lepton_id_weights(lepton_weights, year, goodmuonsHH, "muon", "Loose", label="_hh")

        lepton_weights_dict = {
            f"single_weight_{key}": val
            for key, val in list(lepton_weights._weights.items())
            + list(lepton_weights._modifiers.items())
        }

        weights_dict = {**weights_dict, **lepton_weights_dict}

        return weights_dict, totals_dict

    def getDijetVars(
        self, ak8FatJetVars: dict, bb_mask: np.ndarray, pt_shift: str = None, mass_shift: str = None
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
