import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import correctionlib
import awkward as ak

from coffea.analysis_tools import Weights
from coffea.lookup_tools.dense_lookup import dense_lookup
from coffea.nanoevents.methods.nanoaod import MuonArray, JetArray, FatJetArray, GenParticleArray
from coffea.nanoevents.methods.base import NanoEventsArray

from coffea.nanoevents.methods import vector

ak.behavior.update(vector.behavior)

import pathlib
import gzip
import pickle
import json

import utils
from hh_vars import txbb_wps


package_path = str(pathlib.Path(__file__).parent.parent.resolve())


txbb_sf_lookups = {}


def _load_txbb_sfs(year: str):
    """Create 2D lookup tables in [Txbb, pT] for Txbb SFs from given year"""

    # https://coli.web.cern.ch/coli/.cms/btv/boohft-calib/20221201_bb_ULNanoV9_PNetXbbVsQCD_ak8_ext_2016APV/4_fit/
    with open(package_path + f"/corrections/txbb_sfs/txbb_sf_ul_{year}.json", "r") as f:
        txbb_sf = json.load(f)

    wps = ["LP", "MP", "HP"]

    txbb_bins = np.array([0] + [txbb_wps[year][wp] for wp in wps] + [1])
    pt_bins = np.array([300, 350, 400, 450, 500, 600, 100000])

    edges = (txbb_bins, pt_bins)

    keys = ["central", "high", "low"]

    vals = {key: [] for key in keys}

    for key in keys:
        for wp in wps:
            wval = []
            for (low, high) in zip(pt_bins[:-1], pt_bins[1:]):
                wval.append(txbb_sf[f"{wp}_pt{low}to{high}"]["final"][key])
            vals[key].append(wval)

    vals = {key: np.array(val) for key, val in list(vals.items())}

    txbb_sf_lookups[year] = {
        "nom": dense_lookup(vals["central"], edges),
        "up": dense_lookup(vals["central"] + vals["high"], edges),
        "down": dense_lookup(vals["central"] - vals["low"], edges),
    }


def apply_txbb_sfs(
    events: NanoEventsArray,
    bb_mask: pd.DataFrame,
    year: str,
    weight_key: str = "finalWeight",
):
    """Applies nominal values to ``weight_key`` and stores up down variations"""
    if not year in txbb_sf_lookups:
        _load_txbb_sfs(year)

    bb_txbb = utils.get_feat(events, "bbFatJetParticleNetMD_Txbb", bb_mask)
    bb_pt = utils.get_feat(events, "bbFatJetPt", bb_mask)

    for var in ["up", "down"]:
        events[f"{weight_key}_txbb_{var}"] = events[weight_key] * txbb_sf_lookups[year][var](
            bb_txbb, bb_pt
        )

    events[weight_key] = events[weight_key] * txbb_sf_lookups[year]["nom"](bb_txbb, bb_pt)


def get_uncorr_trig_eff_unc(
    events: NanoEventsArray, bb_mask: pd.DataFrame, year: str, weight_key: str = "finalWeight"
):
    """Get uncorrelated i.e. statistical trigger efficiency uncertainty on samples' yields"""
    return
