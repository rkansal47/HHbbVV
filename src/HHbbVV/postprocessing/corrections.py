from __future__ import annotations

import json
import pickle
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import utils
from coffea.lookup_tools.dense_lookup import dense_lookup
from coffea.nanoevents.methods import vector
from coffea.nanoevents.methods.base import NanoEventsArray
from hist import Hist
from hist.intervals import clopper_pearson_interval
from numpy.typing import ArrayLike

from HHbbVV.hh_vars import data_key, qcd_key, txbb_wps

ak.behavior.update(vector.behavior)

package_path = Path(__file__).parent.parent.resolve()


txbb_sf_lookups = {}


def _load_txbb_sfs(year: str):
    """Create 2D lookup tables in [Txbb, pT] for Txbb SFs from given year"""

    # https://coli.web.cern.ch/coli/.cms/btv/boohft-calib/20221201_bb_ULNanoV9_PNetXbbVsQCD_ak8_ext_2016APV/4_fit/
    with (package_path / f"corrections/txbb_sfs/txbb_sf_ul_{year}.json").open() as f:
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
            for low, high in zip(pt_bins[:-1], pt_bins[1:]):
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
    if year not in txbb_sf_lookups:
        _load_txbb_sfs(year)

    bb_txbb = utils.get_feat(events, "bbFatJetParticleNetMD_Txbb", bb_mask)
    bb_pt = utils.get_feat(events, "bbFatJetPt", bb_mask)

    for var in ["up", "down"]:
        if len(events[weight_key]):
            events[f"{weight_key}_txbb_{var}"] = events[weight_key] * txbb_sf_lookups[year][var](
                bb_txbb, bb_pt
            )
        else:
            events[f"{weight_key}_txbb_{var}"] = events[weight_key]

    if len(events[weight_key]):
        txbb_nom = txbb_sf_lookups[year]["nom"](bb_txbb, bb_pt)
        for wkey in utils.get_all_weights(events):
            if len(events[wkey].shape) > 1:
                events[wkey] *= txbb_nom[:, np.newaxis]
            else:
                events[wkey] *= txbb_nom


trig_effs = {}
trig_errs = {}


def _load_trig_effs(year: str):
    with Path(f"../corrections/trigEffs/{year}_combined.pkl").open("rb") as filehandler:
        combined = pickle.load(filehandler)

    # sum over TH4q bins
    trig_effs[year] = combined["num"][:, sum, :, :] / combined["den"][:, sum, :, :]

    intervals = clopper_pearson_interval(
        combined["num"][:, sum, :, :].view(flow=True),
        combined["den"][:, sum, :, :].view(flow=True),
    )

    trig_errs[year] = (intervals[1] - intervals[0]) / 2


def _get_uncorr_trig_eff_unc_per_sample(
    events: NanoEventsArray, bb_mask: pd.DataFrame, year: str, weight_key: str = "finalWeight"
):
    """Get uncorrelated i.e. statistical trigger efficiency uncertainty on samples' yields"""

    if year not in trig_effs:
        _load_trig_effs(year)

    effs = trig_effs[year]
    errs = trig_errs[year]

    # same binning as efficiencies
    hists = {"bb": Hist(*effs.axes), "VV": Hist(*effs.axes)}

    totals = []
    total_errs = []

    for jet, h in hists.items():
        h.fill(
            jet1txbb=utils.get_feat(events, f"{jet}FatJetParticleNetMD_Txbb", bb_mask),
            jet1pt=utils.get_feat(events, f"{jet}FatJetPt", bb_mask),
            jet1msd=utils.get_feat(events, f"{jet}FatJetMsd", bb_mask),
            weight=utils.get_feat(events, f"{weight_key}_noTrigEffs"),
        )

        total = np.sum(h.values(flow=True) * np.nan_to_num(effs.view(flow=True)))
        totals.append(total)

        total_err = np.linalg.norm((h.values(flow=True) * np.nan_to_num(errs)).reshape(-1))
        total_errs.append(total_err)

    total = np.sum(totals)
    total_err = np.linalg.norm(total_errs)

    return total, total_err


def get_uncorr_trig_eff_unc(
    events_dict: dict[str, NanoEventsArray],
    bb_masks: dict[str, pd.DataFrame],
    year: str,
    sel: dict[str, ArrayLike] = None,
    weight_key: str = "finalWeight",
):
    """Get uncorrelated i.e. statistical trigger efficiency uncertainty on samples' yields"""

    totals = []
    total_errs = []

    for sample, events in events_dict.items():
        if sample not in [data_key, qcd_key]:
            if not len(events) or (sel is not None and not len(events[sel[sample]])):
                continue

            total, total_err = _get_uncorr_trig_eff_unc_per_sample(
                events[sel[sample]] if sel is not None else events,
                bb_masks[sample][sel[sample]] if sel is not None else bb_masks[sample],
                year,
                weight_key,
            )

            totals.append(total)
            total_errs.append(total_err)

    total = np.sum(totals)
    total_err = np.linalg.norm(total_errs)

    return total, total_err


def postprocess_lpsfs(
    events: pd.DataFrame,
    num_jets: int = 2,
    num_lp_sf_toys: int = 100,
    save_all: bool = True,
):
    """
    (1) Splits LP SFs into bb and VV based on gen matching.
    (2) Sets defaults for unmatched jets.
    (3) Cuts of SFs at 10 and normalises.

    Args:
        events (pd.DataFrame): The input DataFrame containing the events.
        num_jets (int, optional): The number of jets. Defaults to 2.
        num_lp_sf_toys (int, optional): The number of LP SF toys. Defaults to 100.
        save_all (bool, optional): Whether to save all LP SFs or only the nominal values. Defaults to True.

    """

    # for jet in ["bb", "VV"]:
    for jet in ["VV"]:
        # temp dict
        td = {}

        # defaults of 1 for jets which aren't matched to anything - i.e. no SF
        for key in ["lp_sf_nom", "lp_sf_sys_down", "lp_sf_sys_up"]:
            td[key] = np.ones(len(events))

        td["lp_sf_toys"] = np.ones((len(events), num_lp_sf_toys))
        td["lp_sf_pt_extrap_vars"] = np.ones((len(events), num_lp_sf_toys))

        # defaults of 0 - i.e. don't contribute to unc.
        for key in [
            "lp_sf_double_matched_event",
            "lp_sf_unmatched_quarks",
            "lp_sf_boundary_quarks",
        ]:
            td[key] = np.zeros(len(events))

        # lp sfs saved for both jets
        if events["lp_sf_sys_up"].shape[1] == 2:
            raise NotImplementedError(
                "Updated LP SF post-processing not implemented yet for both jets with SFs"
            )
            # get events where we have a gen-matched fatjet
            # ignore rare case (~0.002%) where two jets are matched to same gen Higgs
            events.loc[np.sum(events[f"ak8FatJetH{jet}"], axis=1) > 1, f"ak8FatJetH{jet}"] = 0
            jet_match = events[f"ak8FatJetH{jet}"].astype(bool)

            # fill values from matched jets
            for j in range(num_jets):
                offset = num_lp_sf_toys + 1
                td["lp_sf_nom"][jet_match[j]] = events["lp_sf_lnN"][jet_match[j]][j * offset]
                td["lp_sf_toys"][jet_match[j]] = events["lp_sf_lnN"][jet_match[j]].loc[
                    :, j * offset + 1 : (j + 1) * offset - 1
                ]

                for key in [
                    "lp_sf_sys_down",
                    "lp_sf_sys_up",
                    "lp_sf_double_matched_event",
                    "lp_sf_unmatched_quarks",
                    "lp_sf_boundary_quarks",
                ]:
                    td[key][jet_match[j]] = events[key][jet_match[j]][j]

        # lp sfs saved only for hvv jet
        elif events["lp_sf_sys_up"].shape[1] == 1:  # noqa: RET506
            # get events where we have a gen-matched HVV fatjet
            # ignoring rare case (~0.002%) where two jets are matched to same gen Higgs
            jet_match = np.sum(events[f"ak8FatJetH{jet}"], axis=1) == 1

            # fill values from matched jets
            td["lp_sf_nom"][jet_match] = events["lp_sf_lnN"][jet_match][0]
            td["lp_sf_toys"][jet_match] = events["lp_sf_lnN"][jet_match].loc[:, 1:]
            td["lp_sf_pt_extrap_vars"][jet_match] = events["lp_sf_pt_extrap_vars"][
                jet_match
            ].to_numpy()

            for key in [
                "lp_sf_sys_down",
                "lp_sf_sys_up",
                "lp_sf_double_matched_event",
                "lp_sf_unmatched_quarks",
                "lp_sf_boundary_quarks",
            ]:
                td[key][jet_match] = events[key][jet_match].squeeze()

        else:
            raise ValueError("LP SF shapes are invalid")

        for key in [
            "lp_sf_nom",
            "lp_sf_toys",
            "lp_sf_sys_down",
            "lp_sf_sys_up",
            "lp_sf_pt_extrap_vars",
        ]:
            # cut off at 10
            td[key] = np.minimum(td[key], 10)
            # normalise
            td[key] = td[key] / np.mean(td[key], axis=0)

        # add to dataframe
        if save_all:
            td = pd.concat(
                [pd.DataFrame(v.reshape(v.shape[0], -1)) for k, v in td.items()],
                axis=1,
                keys=[f"{jet}_{key}" for key in td],
            )

            events = pd.concat((events, td), axis=1)
        else:
            key = "lp_sf_nom"
            events[f"{jet}_{key}"] = td[key]

    return events


def get_lpsf(
    events: pd.DataFrame, sel: np.ndarray = None, VV: bool = True, weight_key: str = "finalWeight"
):
    """
    Calculates LP SF and uncertainties in current phase space. ``postprocess_lpsfs`` must be called first.
    Assumes bb/VV candidates are matched correctly - this is false for <0.1% events for the cuts we use.
    """

    jet = "VV" if VV else "bb"
    if sel is not None:
        events = events[sel]

    tot_matched = np.sum(np.sum(events[f"ak8FatJetH{jet}"].astype(bool)))

    weight = events[weight_key].to_numpy()
    tot_pre = np.sum(weight)
    tot_post = np.sum(weight * events[f"{jet}_lp_sf_nom"][0])
    lp_sf = tot_post / tot_pre

    uncs = {}

    # difference in yields between up and down shifts on LP SFs
    uncs["syst_unc"] = np.abs(
        (
            np.sum(events[f"{jet}_lp_sf_sys_up"][0] * weight)
            - np.sum(events[f"{jet}_lp_sf_sys_down"][0] * weight)
        )
        / 2
        / tot_post
    )

    # std of yields after all smearings
    uncs["stat_unc"] = (
        np.std(np.sum(weight[:, np.newaxis] * events[f"{jet}_lp_sf_toys"].to_numpy(), axis=0))
        / tot_post
    )

    # pt extrapolation uncertainty is the std of all pt param variations
    uncs["sj_pt_unc"] = (
        np.std(
            np.sum(weight[:, np.newaxis] * events[f"{jet}_lp_sf_pt_extrap_vars"].to_numpy(), axis=0)
        )
        / tot_post
    )

    if VV:
        # OR of double matched and boundary quarks
        # >0.1 to avoid floating point errors
        quark_unmatched = (
            events[f"{jet}_lp_sf_double_matched_event"][0]
            + events[f"{jet}_lp_sf_boundary_quarks"][0]
        ) > 0.1

        sj_matching_unc = np.sum(quark_unmatched)

        # check quark <-> subjet matching for 2, 3, 4 prong-ed jets
        # ignoring events which are already double matched / have boundary quarks to avoid double counting
        num_prongs = events["ak8FatJetHVVNumProngs"][0][~quark_unmatched]
        for nump in range(2, 5):
            sj_matching_unc += (
                np.sum(
                    events[f"{jet}_lp_sf_unmatched_quarks"][0][~quark_unmatched][num_prongs == nump]
                )
                / nump
            )

        uncs["sj_matching_unc"] = sj_matching_unc / tot_matched
    else:
        num_prongs = 2
        uncs["sj_matching_unc"] = (
            (np.sum(events[f"{jet}_lp_sf_unmatched_quarks"][0]) / num_prongs)
            + np.sum(events[f"{jet}_lp_sf_double_matched_event"][0])
        ) / tot_matched

    tot_rel_unc = np.linalg.norm(list(uncs.values()))
    tot_unc = lp_sf * tot_rel_unc

    return lp_sf, tot_unc, uncs
