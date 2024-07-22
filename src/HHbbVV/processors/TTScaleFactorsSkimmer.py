"""
Skimmer for scale factors validation.
Author(s): Raghav Kansal
"""

from __future__ import annotations

import gzip
import json
import logging
import pickle
from collections import OrderedDict
from pathlib import Path

import awkward as ak
import numpy as np
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents.methods import vector
from coffea.processor import dict_accumulator

from .corrections import (
    add_btag_weights,
    add_lepton_weights,
    add_pileup_weight,
    add_pileupid_weights,
    add_top_pt_weight,
    get_jec_jets,
    get_lund_SFs,
)
from .GenSelection import (
    gen_selection_HH4V,
    gen_selection_HHbbVV,
    ttbar_scale_factor_matching,
)
from .SkimmerABC import SkimmerABC
from .TaggerInference import runInferenceTriton
from .utils import P4, Weights, add_selection, pad_val

ak.behavior.update(vector.behavior)
logging.basicConfig(level=logging.INFO)


MU_PDGID = 13

# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {
    "GluGluToHHTobbVV_node_cHHH1": gen_selection_HHbbVV,
    "GluGluToHHTobbVV_node_cHHH1_pn4q": gen_selection_HHbbVV,
    "jhu_HHbbWW": gen_selection_HHbbVV,
    "GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow": gen_selection_HH4V,
    "GluGluToHHTo4V_node_cHHH1": gen_selection_HH4V,
    "GluGluHToWWTo4q_M-125": gen_selection_HH4V,
}

# btag medium WP's https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
# btagWPs = {"2016APV": 0.6001, "2016": 0.5847, "2017": 0.4506, "2018": 0.4168}  # for deepCSV
btagWPs = {"2016APV": 0.2598, "2016": 0.2489, "2017": 0.3040, "2018": 0.2783}  # deepJet


num_prongs = 3


class TTScaleFactorsSkimmer(SkimmerABC):
    """
    Skims nanoaod files, saving selected branches and events passing selection cuts
    (and triggers for data), in a top control region for validation Lund Plane SFs

    Args:
        xsecs (dict, optional): sample cross sections,
          if sample not included no lumi and xsec will not be applied to weights
    """

    HLTs = {  # noqa: RUF012
        "2016": ["TkMu50", "Mu50"],
        "2017": ["Mu50", "OldMu100", "TkMu100"],
        "2018": ["Mu50", "OldMu100", "TkMu100"],
    }

    # key is name in nano files, value will be the name in the skimmed output
    skim_vars = {  # noqa: RUF012
        "FatJet": {
            **P4,
            "msoftdrop": "Msd",
            "particleNet_mass": "ParticleNetMass",
            "particleNetMD_QCD": "ParticleNetMD_QCD",
            "particleNetMD_Xbb": "ParticleNetMD_Xbb",
            "particleNet_H4qvsQCD": "ParticleNet_Th4q",
            "nConstituents": "nPFCands",
        },
        "Jet": P4,
        "FatJetDerived": ["tau21", "tau32", "tau43", "tau42", "tau41"],
        "GenHiggs": P4,
        "other": {"MET_pt": "MET_pt", "MET_phi": "MET_phi"},
    }

    muon_selection = {  # noqa: RUF012
        "Id": "tight",
        "pt": 60,
        "eta": 2.4,
        "miniPFRelIso_all": 0.1,
        "dxy": 0.2,
        "count": 1,
        "delta_trigObj": 0.15,
    }

    ak8_jet_selection = {  # noqa: RUF012
        "pt": 500.0,
        "msd": [125, 250],
        "eta": 2.5,
        "delta_phi_muon": 2,
        "jetId": "tight",
    }

    ak4_jet_selection = {  # noqa: RUF012
        "pt": 30,  # from JME-18-002
        "eta": 2.4,
        "delta_phi_muon": 2,
        "jetId": "tight",
        "puId": 4,  # loose pileup ID
        "btagWP": btagWPs,
        "ht": 250,
        "num": 2,
        # "closest_muon_dr": 0.4,
        # "closest_muon_ptrel": 25,
    }

    met_selection = {"pt": 50}  # noqa: RUF012

    lepW_selection = {"pt": 100}  # noqa: RUF012

    num_jets = 1

    top_matchings = ["top_matched", "w_matched", "unmatched"]  # noqa: RUF012

    def __init__(self, xsecs=None, inference: bool = True):
        if xsecs is None:
            xsecs = {}
        super().__init__()

        # TODO: Check if this is correct
        self.XSECS = xsecs  # in pb

        self._inference = inference

        # find corrections path using this file's path
        package_path = Path(__file__).parent.parent.resolve()
        with gzip.open(package_path / "corrections/corrections.pkl.gz", "rb") as filehandler:
            self.corrections = pickle.load(filehandler)

        # MET filters
        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        with (package_path / "data/metfilters.json").open("rb") as filehandler:
            self.metfilters = json.load(filehandler)

        # for tagger model and preprocessing dict
        self.tagger_resources_path = Path(__file__).parent.resolve() / "tagger_resources"

        self._accumulator = dict_accumulator({})

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events: ak.Array):
        """Returns skimmed events which pass preselection cuts (and triggers if data) with the branches listed in ``self.skim_vars``"""

        # print("processing")

        year = events.metadata["dataset"][:4]
        dataset = events.metadata["dataset"][5:]

        isData = ("JetHT" in dataset) or ("SingleMuon" in dataset)

        if not isData:
            # remove events with pileup weights un-physically large
            events = self.pileup_cutoff(events, year, cutoff=4)

        gen_weights = events["genWeight"].to_numpy() if not isData else None
        n_events = len(events) if isData else np.sum(gen_weights)
        selection = PackedSelection()

        # if n_events != 2915270.0:
        #     print(n_events)
        #     return {}

        cutflow = OrderedDict()
        cutflow["all"] = n_events

        selection_args = (selection, cutflow, isData, gen_weights)

        skimmed_events = {}

        ######################
        # Selection
        ######################

        # Following https://indico.cern.ch/event/1101433/contributions/4775247/

        # triggers
        # OR-ing HLT triggers
        HLT_triggered = np.any(
            np.array(
                [events.HLT[trigger] for trigger in self.HLTs[year] if trigger in events.HLT.fields]
            ),
            axis=0,
        )
        add_selection("trigger", HLT_triggered, *selection_args)

        # objects
        num_ak4_jets = 2
        num_jets = 1
        muon = events.Muon
        fatjets = get_jec_jets(events, year) if not isData else events.FatJet
        ak4_jets = get_jec_jets(events, year, fatjets=False) if not isData else events.Jet
        met = events.MET

        # at least one good reconstructed primary vertex
        add_selection("npvsGood", events.PV.npvsGood >= 1, *selection_args)

        # muon
        muon_selector = (
            (muon[f"{self.muon_selection['Id']}Id"])
            * (muon.pt > self.muon_selection["pt"])
            * (np.abs(muon.eta) < self.muon_selection["eta"])
            * (muon.miniPFRelIso_all < self.muon_selection["miniPFRelIso_all"])
            * (np.abs(muon.dxy) < self.muon_selection["dxy"])
        )

        muon_selector = muon_selector * (
            ak.count(events.Muon.pt[muon_selector], axis=1) == self.muon_selection["count"]
        )
        muon = ak.pad_none(muon[muon_selector], 1, axis=1)[:, 0]

        muon_selector = ak.any(muon_selector, axis=1)

        # 1024 - Mu50 trigger (https://algomez.web.cern.ch/algomez/testWeb/PFnano_content_v02.html#TrigObj)
        trigObj_muon = events.TrigObj[
            (events.TrigObj.id == MU_PDGID) * (events.TrigObj.filterBits >= 1024)
        ]

        muon_selector = muon_selector * ak.any(
            np.abs(muon.delta_r(trigObj_muon)) <= self.muon_selection["delta_trigObj"],
            axis=1,
        )

        add_selection("muon", muon_selector, *selection_args)

        # met
        met_selection = met.pt >= self.met_selection["pt"]

        metfilters = np.ones(len(events), dtype="bool")
        metfilterkey = "data" if isData else "mc"
        for mf in self.metfilters[year][metfilterkey]:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]

        add_selection("met", met_selection * metfilters, *selection_args)

        # leptonic W selection
        add_selection("lepW", (met + muon).pt >= self.lepW_selection["pt"], *selection_args)
        # add_selection("lepW", met.pt + muon.pt >= self.lepW_selection["pt"], *selection_args)

        # ak8 jet selection
        fatjet_selector = (
            (fatjets.pt > self.ak8_jet_selection["pt"])
            * (fatjets.msoftdrop > self.ak8_jet_selection["msd"][0])
            * (fatjets.msoftdrop < self.ak8_jet_selection["msd"][1])
            * (np.abs(fatjets.eta) < self.ak8_jet_selection["eta"])
            * (np.abs(fatjets.delta_phi(muon)) > self.ak8_jet_selection["delta_phi_muon"])
            * fatjets.isTight
        )

        leading_fatjets = ak.pad_none(fatjets[fatjet_selector], num_jets, axis=1)[:, :num_jets]
        fatjet_idx = ak.argmax(fatjet_selector, axis=1)  # gets first index which is true
        fatjet_selector = ak.any(fatjet_selector, axis=1)

        add_selection("ak8_jet", fatjet_selector, *selection_args)

        # ak4 jet

        # kinematic / ID selection
        ak4_jet_selector = (
            ak4_jets.isTight
            # pileup ID should only be applied for pT < 50 jets (https://twiki.cern.ch/twiki/bin/view/CMS/PileupJetIDUL)
            * ((ak4_jets.puId % 2 == 1) + (ak4_jets.pt >= 50))
            * (ak4_jets.pt > self.ak4_jet_selection["pt"])
            * (np.abs(ak4_jets.eta) < self.ak4_jet_selection["eta"])
        )
        ak4_jets_selected = ak.fill_none(ak4_jets[ak4_jet_selector], [], axis=0)

        # b-tagged and dPhi from muon < 2
        ak4_jet_selector_btag_muon = ak4_jet_selector * (
            (ak4_jets.btagDeepFlavB > self.ak4_jet_selection["btagWP"][year])
            * (np.abs(ak4_jets.delta_phi(muon)) < self.ak4_jet_selection["delta_phi_muon"])
        )
        bjets_selected = ak.fill_none(ak4_jets[ak4_jet_selector_btag_muon], [], axis=0)

        ak4_selection = (
            # at least 1 b-tagged jet close to the muon
            (ak.any(ak4_jet_selector_btag_muon, axis=1))
            # at least 2 ak4 jets overall
            * (ak.sum(ak4_jet_selector, axis=1) >= self.ak4_jet_selection["num"])
            # ht > 250
            * (ak.sum(ak4_jets_selected.pt, axis=1) >= self.ak4_jet_selection["ht"])
        )

        add_selection("ak4_jet", ak4_selection, *selection_args)

        # 2018 HEM cleaning
        # https://indico.cern.ch/event/1249623/contributions/5250491/attachments/2594272/4477699/HWW_0228_Draft.pdf
        if year == "2018":
            hem_cleaning = (
                ((events.run >= 319077) & isData)  # if data check if in Runs C or D
                # else for MC randomly cut based on lumi fraction of C&D
                | ((np.random.rand(len(events)) < 0.632) & ~isData)
            ) & (
                ak.fill_none(
                    ak.any(
                        (
                            (leading_fatjets.pt > 30.0)
                            & (leading_fatjets.eta > -3.2)
                            & (leading_fatjets.eta < -1.3)
                            & (leading_fatjets.phi > -1.57)
                            & (leading_fatjets.phi < -0.87)
                        ),
                        -1,
                    ),
                    False,
                )
                | ((events.MET.phi > -1.62) & (events.MET.pt < 470.0) & (events.MET.phi < -0.62))
            )

            add_selection("hem_cleaning", ~np.array(hem_cleaning).astype(bool), *selection_args)

        # select vars
        # AK4 jets just for plots for JME
        ak4JetVars = {
            f"ak4Jet{key}": pad_val(ak4_jets_selected[var], num_ak4_jets, axis=1)
            for (var, key) in self.skim_vars["Jet"].items()
        }

        bJetVars = {
            f"bJet{key}": pad_val(bjets_selected[var], 1, axis=1)
            for (var, key) in self.skim_vars["Jet"].items()
        }

        ak8FatJetVars = {
            f"ak8FatJet{key}": pad_val(leading_fatjets[var], num_jets, axis=1)
            for (var, key) in self.skim_vars["FatJet"].items()
        }

        for var in self.skim_vars["FatJetDerived"]:
            if var.startswith("tau"):
                taunum = pad_val(fatjets[f"tau{var[3]}"], num_jets, axis=1)
                tauden = pad_val(fatjets[f"tau{var[4]}"], num_jets, axis=1)
                ak8FatJetVars[var] = taunum / tauden

        otherVars = {
            key: events[var.split("_")[0]]["_".join(var.split("_")[1:])].to_numpy()
            for (var, key) in self.skim_vars["other"].items()
        }

        skimmed_events = {**skimmed_events, **ak4JetVars, **bJetVars, **ak8FatJetVars, **otherVars}

        ####################################
        # Particlenet h4q vs qcd, xbb vs qcd
        ####################################

        skimmed_events["ak8FatJetParticleNetMD_Txbb"] = pad_val(
            leading_fatjets.particleNetMD_Xbb
            / (leading_fatjets.particleNetMD_QCD + leading_fatjets.particleNetMD_Xbb),
            num_jets,
            -1,
            axis=1,
        )

        skimmed_events["ak8FatJetParticleNetMD_Txqq"] = pad_val(
            leading_fatjets.particleNetMD_Xqq
            / (leading_fatjets.particleNetMD_QCD + leading_fatjets.particleNetMD_Xqq),
            num_jets,
            -1,
            axis=1,
        )

        skimmed_events["ak8FatJetParticleNetMD_Txcc"] = pad_val(
            leading_fatjets.particleNetMD_Xcc
            / (leading_fatjets.particleNetMD_QCD + leading_fatjets.particleNetMD_Xcc),
            num_jets,
            -1,
            axis=1,
        )

        skimmed_events["ak8FatJetParticleNetMD_Txqc"] = pad_val(
            (leading_fatjets.particleNetMD_Xcc + leading_fatjets.particleNetMD_Xqq)
            / (
                leading_fatjets.particleNetMD_QCD
                + leading_fatjets.particleNetMD_Xqq
                + leading_fatjets.particleNetMD_Xcc
            ),
            num_jets,
            -1,
            axis=1,
        )

        #########################
        # Weights
        #########################

        totals_dict = {"nevents": n_events}

        if isData:
            skimmed_events["weight"] = np.ones(n_events)
        else:
            weights_dict, totals_temp = self.add_weights(
                events, dataset, year, gen_weights, muon, ak4_jets_selected
            )
            skimmed_events = {**skimmed_events, **weights_dict}
            totals_dict = {**totals_dict, **totals_temp}

        #########################
        # Lund Plane SFs
        #########################

        lp_hist = None

        if dataset in ["SingleTop", "TTToSemiLeptonic", "TTToSemiLeptonic_ext1"]:
            match_dict, gen_quarks, had_bs = ttbar_scale_factor_matching(
                events, leading_fatjets[:, 0], selection_args
            )
            print(cutflow)
            top_matched = match_dict["top_matched"].astype(bool) * selection.all(*selection.names)

            skimmed_events = {**skimmed_events, **match_dict}

            if np.any(top_matched):
                sf_dict, lp_hist = get_lund_SFs(
                    year,
                    events[top_matched],
                    fatjets[top_matched],
                    fatjet_idx[top_matched].to_numpy(),
                    num_prongs,
                    gen_quarks[top_matched],
                    weights_dict["weight"][top_matched],
                    trunc_gauss=True,
                    lnN=True,
                    gen_bs=had_bs[top_matched],  # do b/l ratio uncertainty for tops as well
                )

                # fill zeros for all non-top-matched events
                for key, val in list(sf_dict.items()):
                    # plus 1 for the nominal values
                    arr = np.zeros((len(events), val.shape[1]))
                    arr[top_matched] = val
                    sf_dict[key] = arr

                skimmed_events = {**skimmed_events, **sf_dict}

        ##############################
        # Apply selections
        ##############################

        skimmed_events = {
            key: value[selection.all(*selection.names)] for (key, value) in skimmed_events.items()
        }

        ######################
        # HWW Tagger Inference
        ######################

        if self._inference:
            print("pre-inference")

            pnet_vars = runInferenceTriton(
                self.tagger_resources_path,
                events[selection.all(*selection.names)],
                num_jets=1,
                in_jet_idx=fatjet_idx[selection.all(*selection.names)],
                jets=ak.flatten(leading_fatjets[selection.all(*selection.names)]),
                all_outputs=False,
            )

            print("post-inference")

            skimmed_events = {
                **skimmed_events,
                **dict(pnet_vars.items()),
            }

        if len(skimmed_events["weight"]):
            pddf = self.to_pandas(skimmed_events)
            fname = (
                events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
            )
            self.dump_table(pddf, fname)

        ret_dict = {year: {dataset: {"totals": totals_dict, "cutflow": cutflow}}}

        if lp_hist is not None:
            ret_dict[year][dataset]["lp_hist"] = lp_hist

        print(ret_dict)
        return ret_dict

    def postprocess(self, accumulator):
        return accumulator

    def add_weights(self, events, dataset, year, gen_weights, muon, ak4_jets) -> tuple[dict, dict]:
        """Adds weights and variations, saves totals for all norm preserving weights and variations"""
        weights = Weights(len(events), storeIndividual=True)
        weights.add("genweight", gen_weights)

        add_pileup_weight(weights, year, events.Pileup.nPU.to_numpy())
        add_lepton_weights(weights, year, muon)  # includes both ID and trigger SFs
        add_btag_weights(weights, year, ak4_jets)
        add_pileupid_weights(weights, year, ak4_jets, events.GenJet, wp="L")

        if year in ("2016APV", "2016", "2017"):
            weights.add(
                "L1EcalPrefiring",
                events.L1PreFiringWeight.Nom,
                events.L1PreFiringWeight.Up,
                events.L1PreFiringWeight.Dn,
            )

        if dataset.startswith("TTTo"):
            add_top_pt_weight(weights, events)

        ###################### Save all the weights and variations ######################

        # these weights should not change the overall normalization, so are saved separately
        norm_preserving_weights = ["genweight", "pileup"]

        # dictionary of all weights and variations
        weights_dict = {}
        # dictionary of total # events for norm preserving variations for normalization in postprocessing
        totals_dict = {}

        # nominal
        weights_dict["weight"] = weights.weight()
        weights_dict["weight_nobtagSFs"] = weights.partial_weight(exclude=["btagSF"])

        # norm preserving weights, used to do normalization in post-processing
        weight_np = weights.partial_weight(include=norm_preserving_weights)
        totals_dict["np_nominal"] = np.sum(weight_np)

        ###################### Normalization (Step 1) ######################

        weight_norm = self.get_dataset_norm(year, dataset)
        # normalize all the weights to xsec, needs to be divided by totals in Step 2 in post-processing
        for key, val in weights_dict.items():
            weights_dict[key] = val * weight_norm

        # save the unnormalized weight, to confirm that it's been normalized in post-processing
        weights_dict["weight_noxsec"] = weights.weight()

        weights_dict["genWeight"] = gen_weights

        return weights_dict, totals_dict
