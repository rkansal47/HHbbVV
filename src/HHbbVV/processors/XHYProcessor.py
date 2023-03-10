"""
Get # of quarks per HVV fatjet for resonant samples.

Author(s): Raghav Kansal
"""

import numpy as np
import awkward as ak
import pandas as pd

from coffea import processor
from coffea.analysis_tools import Weights, PackedSelection
import vector
from hist import Hist

import pathlib
import pickle
import gzip
import os

from typing import Dict
from collections import OrderedDict

from .GenSelection import gen_selection_HYbbVV
from .utils import pad_val, add_selection, concatenate_dicts, P4
from .corrections import (
    get_jec_key,
    get_jec_jets,
    get_jmsr,
)

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def update(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out


class XHYProcessor(processor.ProcessorABC):
    """
    Histograms of quarks captured per YVV fatjet.
    """

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

    def __init__(self):
        super(XHYProcessor, self).__init__()

        self._accumulator = processor.dict_accumulator({})

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events: ak.Array):
        """Runs event processor for different types of jets"""

        year = events.metadata["dataset"].split("_")[0]
        year_nosuffix = year.replace("APV", "")
        dataset = "_".join(events.metadata["dataset"].split("_")[1:])
        isData = False

        # only signs for HH
        # TODO: check if this is also the case for HY
        gen_weights = np.sign(events["genWeight"])

        n_events = np.sum(gen_weights)
        selection = PackedSelection()
        weights = Weights(len(events), storeIndividual=True)

        cutflow = OrderedDict()
        cutflow["all"] = n_events

        selection_args = (selection, cutflow, isData, gen_weights)

        num_jets = 2
        fatjets, jec_shifted_vars = get_jec_jets(events, year, isData, self.jecs)

        # change to year with suffix after updated JMS/R values
        jmsr_shifted_vars = get_jmsr(fatjets, num_jets, year_nosuffix)

        skimmed_events = {}

        vars_dict, _ = gen_selection_HYbbVV(events, fatjets, selection, cutflow, gen_weights, P4)
        skimmed_events = {**skimmed_events, **vars_dict}

        # FatJet vars

        ak8FatJetVars = {
            f"ak8FatJet{key}": pad_val(fatjets[var], num_jets, axis=1)
            for (var, key) in self.skim_vars["FatJet"].items()
        }

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

        ######################
        # Selection
        ######################

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

        # reshape and apply selections
        sel_all = selection.all(*selection.names)

        skimmed_events = {
            key: value.reshape(len(events), -1)[sel_all] for (key, value) in skimmed_events.items()
        }

        # initialize histograms
        h = Hist.new.Var(
            [0, 1, 2, 3, 4, 5], name="numquarks", label="Number of quarks in AK8 Jet"
        ).Double()

        if len(skimmed_events["ak8FatJetHVVNumProngs"]):
            h.fill(numquarks=skimmed_events["ak8FatJetHVVNumProngs"].squeeze())

        return {"h": h, "cutflow": cutflow}

    def postprocess(self, accumulator):
        return accumulator
