# building off https://github.com/nsmith-/boostedhiggs/blob/master/boostedhiggs/corrections.py

import os
import numpy as np
import awkward
import gzip
import pickle
from coffea.lookup_tools.lookup_base import lookup_base

with gzip.open(os.path.join(os.path.dirname(__file__), "data", "corrections.pkl.gz")) as fin:
    compiled = pickle.load(fin)

# hotfix some crazy large weights
compiled["2017_pileupweight"]._values = np.minimum(5, compiled["2017_pileupweight"]._values)
compiled["2018_pileupweight"]._values = np.minimum(5, compiled["2018_pileupweight"]._values)


class SoftDropWeight(lookup_base):
    def _evaluate(self, pt, eta):
        gpar = np.array([1.00626, -1.06161, 0.0799900, 1.20454])
        cpar = np.array(
            [1.09302, -0.000150068, 3.44866e-07, -2.68100e-10, 8.67440e-14, -1.00114e-17]
        )
        fpar = np.array(
            [1.27212, -0.000571640, 8.37289e-07, -5.20433e-10, 1.45375e-13, -1.50389e-17]
        )
        genw = gpar[0] + gpar[1] * np.power(pt * gpar[2], -gpar[3])
        ptpow = np.power.outer(pt, np.arange(cpar.size))
        cenweight = np.dot(ptpow, cpar)
        forweight = np.dot(ptpow, fpar)
        weight = np.where(np.abs(eta) < 1.3, cenweight, forweight)
        return genw * weight


_softdrop_weight = SoftDropWeight()


def corrected_msoftdrop(fatjets):
    sf = _softdrop_weight(fatjets.pt, fatjets.eta)
    sf = np.maximum(1e-5, sf)
    try:
        # pancakes have the raw value
        dazsle_msd = fatjets.msoftdrop_raw
    except AttributeError:
        # for nanoaod we have to work back to it
        # TODO: this should be ak.sum(..., axis=-1) but not working
        dazsle_msd = (fatjets.subjets * (1 - fatjets.subjets.rawFactor)).sum().mass
    return dazsle_msd * sf


def n2ddt_shift(fatjets, year="2017"):
    return compiled[f"{year}_n2ddt_rho_pt"](fatjets.qcdrho, fatjets.pt)


def add_pileup_weight(weights, nPU, year="2017", dataset=None):
    if year == "2017" and dataset in compiled["2017_pileupweight_dataset"]:
        weights.add(
            "pileup_weight",
            compiled["2017_pileupweight_dataset"][dataset](nPU),
            compiled["2017_pileupweight_dataset_puUp"][dataset](nPU),
            compiled["2017_pileupweight_dataset_puDown"][dataset](nPU),
        )
    else:
        weights.add(
            "pileup_weight",
            compiled[f"{year}_pileupweight"](nPU),
            compiled[f"{year}_pileupweight_puUp"](nPU),
            compiled[f"{year}_pileupweight_puDown"](nPU),
        )


def add_VJets_NLOkFactor(weights, genBosonPt, year, dataset):
    if year == "2017" and "ZJetsToQQ_HT" in dataset:
        nlo_over_lo_qcd = compiled["2017_Z_nlo_qcd"](genBosonPt)
        nlo_over_lo_ewk = compiled["Z_nlo_over_lo_ewk"](genBosonPt)
    elif year == "2017" and "WJetsToQQ_HT" in dataset:
        nlo_over_lo_qcd = compiled["2017_W_nlo_qcd"](genBosonPt)
        nlo_over_lo_ewk = compiled["W_nlo_over_lo_ewk"](genBosonPt)
    elif year == "2016" and "DYJetsToQQ" in dataset:
        nlo_over_lo_qcd = compiled["2016_Z_nlo_qcd"](genBosonPt)
        nlo_over_lo_ewk = compiled["Z_nlo_over_lo_ewk"](genBosonPt)
    elif year == "2016" and "WJetsToQQ" in dataset:
        nlo_over_lo_qcd = compiled["2016_W_nlo_qcd"](genBosonPt)
        nlo_over_lo_ewk = compiled["W_nlo_over_lo_ewk"](genBosonPt)
    else:
        return
    weights.add("VJets_NLOkFactor", nlo_over_lo_qcd * nlo_over_lo_ewk)


def add_jetTriggerWeight(weights, jet_msd, jet_pt, year):
    nom = compiled[f"{year}_trigweight_msd_pt"](jet_msd, jet_pt)
    up = compiled[f"{year}_trigweight_msd_pt_trigweightUp"](jet_msd, jet_pt)
    down = compiled[f"{year}_trigweight_msd_pt_trigweightDown"](jet_msd, jet_pt)
    weights.add("jet_trigger", nom, up, down)
