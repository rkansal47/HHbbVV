"""
Skimmer for scale factors validation.
Author(s): Raghav Kansal
"""

from collections import OrderedDict
import numpy as np
import awkward as ak
import pandas as pd

from coffea.processor import ProcessorABC, dict_accumulator
from coffea.analysis_tools import Weights, PackedSelection
from coffea.nanoevents.methods import nanoaod

from coffea.lookup_tools.dense_lookup import dense_lookup
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.nanoevents.methods.nanoaod import FatJetArray
from coffea.nanoevents.methods import vector

ak.behavior.update(vector.behavior)

import uproot

import fastjet

import pathlib
import pickle, json
import gzip
import os

from typing import Dict, Tuple, List

from .GenSelection import gen_selection_HHbbVV, gen_selection_HH4V, ttbar_scale_factor_matching
from .TaggerInference import runInferenceTriton
from .utils import pad_val, add_selection
from .corrections import add_pileup_weight, add_lepton_weights, add_top_pt_weight, get_jec_jets


P4 = {
    "eta": "Eta",
    "phi": "Phi",
    "mass": "Mass",
    "pt": "Pt",
}

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

# deepcsv medium WP's https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
btagWPs = {"2016APV": 0.6001, "2016": 0.5847, "2017": 0.4506, "2018": 0.4168}


# jet definitions
dR = 0.8
cadef = fastjet.JetDefinition(fastjet.cambridge_algorithm, dR)
ktdef = fastjet.JetDefinition(fastjet.kt_algorithm, dR)
num_prongs = 3


def lund_SFs(
    events: NanoEventsArray,
    fatjet_idx: ak.Array,
    ratio_smeared_lookups: List[dense_lookup],
    ratio_lnN_smeared_lookups: List[dense_lookup],
) -> np.ndarray:
    """
    Calculates scale factors for the leading jet in events based on splittings in the primary Lund Plane.

    Lookup tables should be binned in [subjet_pt, ln(0.8/Delta), ln(kT/GeV)].

    Returns nominal scale factors for each lookup table in the ``ratio_smeared_lookups`` list.

    Args:
        events (NanoEventsArray): nano events
        ratio_smeared_lookups (List[dense_lookup]): list of lookup tables with smeared values

    Returns:
        nd.ndarray: SF values per jet for each smearing, shape ``[n_jets, n_sf_toys]``.
    """

    # get pfcands of the top-matched jets
    ak8_pfcands = events.FatJetPFCands
    ak8_pfcands = ak8_pfcands[ak8_pfcands.jetIdx == fatjet_idx]
    pfcands = events.PFCands[ak8_pfcands.pFCandsIdx]

    # need to convert to such a structure
    pfcands_vector_ptetaphi = ak.Array(
        [
            [{kin_key: cand[kin_key] for kin_key in P4} for cand in event_cands]
            for event_cands in pfcands
        ],
        with_name="PtEtaPhiMLorentzVector",
    )

    # cluster first with kT
    kt_clustering = fastjet.ClusterSequence(pfcands_vector_ptetaphi, ktdef)
    kt_subjets = kt_clustering.exclusive_jets(num_prongs)
    # save subjet pT
    kt_subjets_pt = np.sqrt(kt_subjets.px**2 + kt_subjets.py**2)
    # get constituents
    kt_subjet_consts = kt_clustering.exclusive_jets_constituents(num_prongs)

    # then re-cluster with CA
    # won't need to flatten once https://github.com/scikit-hep/fastjet/pull/145 is released
    ca_clustering = fastjet.ClusterSequence(ak.flatten(kt_subjet_consts, axis=1), cadef)
    lds = ak.flatten(ca_clustering.exclusive_jets_lund_declusterings(1), axis=1)

    # flatten and save offsets to unflatten afterwards
    ld_offsets = lds.kt.layout.offsets
    flat_logD = np.log(0.8 / ak.flatten(lds).Delta).to_numpy()
    flat_logkt = np.log(ak.flatten(lds).kt).to_numpy()
    # repeat subjet pt for each lund declustering
    flat_subjet_pt = np.repeat(ak.flatten(kt_subjets_pt), ak.count(lds.kt, axis=1)).to_numpy()

    sf_vals = []
    # could be parallelised but not sure if memory / time trade-off is worth it
    for i, ratio_nom_lookup in enumerate(ratio_smeared_lookups):
        ratio_nom_vals = ratio_nom_lookup(flat_subjet_pt, flat_logD, flat_logkt)
        # recover jagged event structure
        reshaped_ratio_nom_vals = ak.Array(
            ak.layout.ListOffsetArray64(ld_offsets, ak.layout.NumpyArray(ratio_nom_vals))
        )
        # nominal values are product of all lund plane SFs
        sf_vals.append(
            # multiply subjet SFs per jet
            np.prod(
                # per-subjet SF
                ak.prod(reshaped_ratio_nom_vals, axis=1).to_numpy().reshape(-1, num_prongs),
                axis=1,
            )
        )

    sf_vals = np.array(sf_vals).T  # output shape: ``[n_jets, n_sf_toys]``

    sf_lnN_vals = []
    # could be parallelised but not sure if memory / time trade-off is worth it
    for i, ratio_nom_lookup in enumerate(ratio_lnN_smeared_lookups):
        ratio_nom_vals = ratio_nom_lookup(flat_subjet_pt, flat_logD, flat_logkt)
        # recover jagged event structure
        reshaped_ratio_nom_vals = ak.Array(
            ak.layout.ListOffsetArray64(ld_offsets, ak.layout.NumpyArray(ratio_nom_vals))
        )
        # nominal values are product of all lund plane SFs
        sf_lnN_vals.append(
            # multiply subjet SFs per jet
            np.prod(
                # per-subjet SF
                ak.prod(reshaped_ratio_nom_vals, axis=1).to_numpy().reshape(-1, num_prongs),
                axis=1,
            )
        )

    sf_lnN_vals = np.array(sf_lnN_vals).T  # output shape: ``[n_jets, n_sf_toys]``

    return sf_vals, sf_lnN_vals


class TTScaleFactorsSkimmer(ProcessorABC):
    """
    Skims nanoaod files, saving selected branches and events passing selection cuts
    (and triggers for data), in a top control region for validation Lund Plane SFs

    Args:
        xsecs (dict, optional): sample cross sections,
          if sample not included no lumi and xsec will not be applied to weights
    """

    # from https://cds.cern.ch/record/2724492/files/DP2020_035.pdf
    LUMI = {"2016APV": 20e3, "2016": 16e3, "2017": 41e3, "2018": 59e3}  # in pb^-1

    HLTs = {
        "2016": ["TkMu50", "Mu50"],
        "2017": ["Mu50", "OldMu100", "TkMu100"],
        "2018": ["Mu50", "OldMu100", "TkMu100"],
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
            "nConstituents": "nPFCands",
        },
        "FatJetDerived": ["tau21", "tau32", "tau43", "tau42", "tau41"],
        "GenHiggs": P4,
        "other": {"MET_pt": "MET_pt"},
    }

    muon_selection = {
        "Id": "tight",
        "pt": 60,
        "eta": 2.4,
        "miniPFRelIso_all": 0.1,
        "dxy": 0.2,
        "count": 1,
        "delta_trigObj": 0.15,
    }

    ak8_jet_selection = {
        "pt": 400.0,
        "msd": [125, 250],
        "eta": 2.5,
        "delta_phi_muon": 2,
        "jetId": 2,  # tight ID bit
    }

    ak4_jet_selection = {
        "pt": 25,
        "eta": 2.4,
        "delta_phi_muon": 2,
        "jetId": 2,
        "puId": 4,  # loose pileup ID
        "btagWP": btagWPs,
    }

    met_selection = {"pt": 50}

    lepW_selection = {"pt": 100}

    num_jets = 1

    top_matchings = ["top_matched", "w_matched", "unmatched"]

    def __init__(self, xsecs={}):
        super(TTScaleFactorsSkimmer, self).__init__()

        # TODO: Check if this is correct
        self.XSECS = xsecs  # in pb

        # find corrections path using this file's path
        package_path = str(pathlib.Path(__file__).parent.parent.resolve())
        with gzip.open(package_path + "/corrections/corrections.pkl.gz", "rb") as filehandler:
            self.corrections = pickle.load(filehandler)

        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        with open(package_path + "/data/metfilters.json", "rb") as filehandler:
            self.metfilters = json.load(filehandler)

        # for tagger model and preprocessing dict
        self.tagger_resources_path = (
            str(pathlib.Path(__file__).parent.resolve()) + "/tagger_resources/"
        )

        # initialize lund plane scale factors lookups
        f = uproot.open(package_path + "/corrections/ratio_ktjets_nov7.root")

        # 3D histogram: [subjet_pt, ln(0.8/Delta), ln(kT/GeV)]
        ratio_nom = f["ratio_nom"].to_numpy()
        ratio_nom_errs = f["ratio_nom"].errors()
        ratio_nom_edges = ratio_nom[1:]
        ratio_nom = ratio_nom[0]

        n_sf_toys = 100
        np.random.seed(42)
        rand_noise = np.random.normal(size=[n_sf_toys, *ratio_nom.shape])

        # produces array of shape ``[n_sf_toys, subjet_pt bins, ln(0.8/Delta) bins, ln(kT/GeV) bins]``
        ratio_nom_smeared = ratio_nom + (ratio_nom_errs * rand_noise)
        ratio_nom_smeared = np.maximum(ratio_nom_smeared, 0)
        # save n_sf_toys lookups
        self.ratio_smeared_lookups = [dense_lookup(ratio_nom, ratio_nom_edges)] + [
            dense_lookup(ratio_nom_smeared[i], ratio_nom_edges) for i in range(n_sf_toys)
        ]

        # revised smearing (0s -> 1s, normal -> lnN)
        zero_noms = ratio_nom == 0
        ratio_nom[zero_noms] = 1
        ratio_nom_errs[zero_noms] = 0

        kappa = (ratio_nom + ratio_nom_errs) / ratio_nom
        ratio_nom_smeared = ratio_nom * np.power(kappa, rand_noise)
        self.ratio_lnN_smeared_lookups = [dense_lookup(ratio_nom, ratio_nom_edges)] + [
            dense_lookup(ratio_nom_smeared[i], ratio_nom_edges) for i in range(n_sf_toys)
        ]

        self.n_sf_toys = n_sf_toys

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

        # print("processing")

        year = events.metadata["dataset"][:4]
        dataset = events.metadata["dataset"][5:]

        isData = ("JetHT" in dataset) or ("SingleMuon" in dataset)
        signGenWeights = None if isData else np.sign(events["genWeight"])
        # TODO: undo!!!!
        # signGenWeights = np.ones(len(events))
        n_events = len(events) if isData else int(np.sum(signGenWeights))
        selection = PackedSelection()
        weights = Weights(len(events), storeIndividual=True)

        cutflow = OrderedDict()
        cutflow["all"] = n_events

        selection_args = (selection, cutflow, isData, signGenWeights)

        skimmed_events = {}

        # gen vars - saving HH, bb, VV, and 4q 4-vectors + Higgs children information
        # if dataset in gen_selection_dict:
        #     skimmed_events = {
        #         **skimmed_events,
        #         **gen_selection_dict[dataset](events, selection, cutflow, signGenWeights, P4),
        #     }

        # Event Selection
        # Following https://indico.cern.ch/event/1101433/contributions/4775247/

        # triggers
        # OR-ing HLT triggers
        if isData or not isData:  # CASE uses triggers for MC too
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
            add_selection("trigger", HLT_triggered, *selection_args)

        # objects
        num_jets = 1
        muon = events.Muon
        fatjets = get_jec_jets(events, year) if not isData else events.FatJet
        ak4_jets = events.Jet
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
        # add_selection("lepW", (met + muon).pt >= self.lepW_selection["pt"], *selection_args)
        add_selection("lepW", met.pt + muon.pt >= self.lepW_selection["pt"], *selection_args)

        # ak8 jet selection
        fatjet_selector = (
            (fatjets.pt > self.ak8_jet_selection["pt"])
            # * (fatjets.msoftdrop > self.ak8_jet_selection["msd"][0])
            # * (fatjets.msoftdrop < self.ak8_jet_selection["msd"][1])
            * (np.abs(fatjets.eta) < self.ak8_jet_selection["eta"])
            * (np.abs(fatjets.delta_phi(muon)) > self.ak8_jet_selection["delta_phi_muon"])
            * (
                fatjets.jetId & self.ak8_jet_selection["jetId"] == self.ak8_jet_selection["jetId"]
            )  # tight ID
        )

        leading_fatjets = ak.pad_none(fatjets[fatjet_selector], num_jets, axis=1)[:, :num_jets]
        fatjet_idx = ak.argmax(fatjet_selector, axis=1)  # gets first index which is true
        fatjet_selector = ak.any(fatjet_selector, axis=1)

        add_selection("ak8_jet", fatjet_selector, *selection_args)

        # ak4 jet
        # leading_btag_jet = ak.flatten(
        #     ak.pad_none(events.Jet[ak.argsort(events.Jet.btagDeepB, axis=1)[:, -1:]], 1, axis=1)
        # )

        ak4_jet_selector = (
            (
                ak4_jets.jetId & self.ak4_jet_selection["jetId"] == self.ak4_jet_selection["jetId"]
            )  # tight ID
            # * (ak4_jets.puId >= self.ak4_jet_selection["puId"])
            * (ak4_jets.puId % 2 == 1)
            * (ak4_jets.pt > self.ak4_jet_selection["pt"])
            * (np.abs(ak4_jets.eta) < self.ak4_jet_selection["eta"])
            * (np.abs(ak4_jets.delta_phi(muon)) < self.ak4_jet_selection["delta_phi_muon"])
            * (ak4_jets.btagDeepB > self.ak4_jet_selection["btagWP"]["2018"])
        )

        ak4_jet_selector = ak.any(ak4_jet_selector, axis=1)
        add_selection("ak4_jet", ak4_jet_selector, *selection_args)

        # select vars

        ak8FatJetVars = {
            f"ak8FatJet{key}": pad_val(fatjets[var], num_jets, -99999, axis=1)
            for (var, key) in self.skim_vars["FatJet"].items()
        }

        for var in self.skim_vars["FatJetDerived"]:
            if var.startswith("tau"):
                taunum = pad_val(fatjets[f"tau{var[3]}"], num_jets, -99999, axis=1)
                tauden = pad_val(fatjets[f"tau{var[4]}"], num_jets, -99999, axis=1)
                ak8FatJetVars[var] = taunum / tauden

        otherVars = {
            key: events[var.split("_")[0]]["_".join(var.split("_")[1:])].to_numpy()
            for (var, key) in self.skim_vars["other"].items()
        }

        skimmed_events = {**skimmed_events, **ak8FatJetVars, **otherVars}

        # particlenet h4q vs qcd, xbb vs qcd

        skimmed_events["ak8FatJetParticleNetMD_Txbb"] = pad_val(
            fatjets.particleNetMD_Xbb / (fatjets.particleNetMD_QCD + fatjets.particleNetMD_Xbb),
            num_jets,
            -1,
            axis=1,
        )

        # calc weights

        if isData:
            skimmed_events["weight"] = np.ones(n_events)
        else:
            skimmed_events["genWeight"] = events.genWeight.to_numpy()
            add_pileup_weight(weights, year, events.Pileup.nPU.to_numpy())
            add_lepton_weights(weights, year, muon)

            if dataset.startswith("TTTo"):
                add_top_pt_weight(weights, events)

            # this still needs to be normalized with the acceptance of the pre-selection (done now in post processing)
            if dataset in self.XSECS:
                skimmed_events["weight"] = (
                    np.sign(skimmed_events["genWeight"])
                    * self.XSECS[dataset]
                    * self.LUMI[year]
                    * weights.weight()
                )
            else:
                skimmed_events["weight"] = np.sign(skimmed_events["genWeight"])

        if dataset in ["TTToSemiLeptonic", "TTToSemiLeptonic_ext1"]:
            match_dict = ttbar_scale_factor_matching(events, leading_fatjets[:, 0], selection_args)
            top_matched = match_dict["top_matched"].astype(bool) * selection.all(*selection.names)

            if np.any(top_matched):
                sf_vals, sf_lnN_vals = lund_SFs(
                    events[top_matched],
                    fatjet_idx[top_matched],
                    self.ratio_smeared_lookups,
                    self.ratio_lnN_smeared_lookups,
                )

                sf_dict = {"lp_sf": sf_vals, "lp_sf_lnN": sf_lnN_vals}

                # fill zeros for all non-top-matched events
                for key, val in list(sf_dict.items()):
                    # plus 1 for the nominal values
                    arr = np.zeros((len(events), self.n_sf_toys + 1))
                    arr[top_matched] = val
                    sf_dict[key] = arr
            else:
                sf_dict = {
                    "lp_sf": np.zeros((len(events), self.n_sf_toys + 1)),
                    "lp_sf_lnN": np.zeros((len(events), self.n_sf_toys + 1)),
                }

            skimmed_events = {**skimmed_events, **match_dict, **sf_dict}

        # apply selections

        skimmed_events = {
            key: value[selection.all(*selection.names)] for (key, value) in skimmed_events.items()
        }

        # apply HWW4q tagger
        print("pre-inference")

        pnet_vars = runInferenceTriton(
            self.tagger_resources_path, events[selection.all(*selection.names)], num_jets=1, jet_idx=fatjet_idx
        )

        print("post-inference")

        pnet_vars = {}

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

        # print(cutflow)

        return {year: {dataset: {"nevents": n_events, "cutflow": cutflow}}}

    def postprocess(self, accumulator):
        return accumulator
