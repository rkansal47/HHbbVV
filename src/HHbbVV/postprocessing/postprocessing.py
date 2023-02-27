"""
Post processing skimmed parquet files (output of bbVVSkimmer processor):
(1) Applies weights / scale factors,
(2) Assigns bb, VV jets
(3) Derives extra dijet kinematic variables
(4) (optionally) Loads BDT predictions (BDT is trained separately in TrainBDT.py)
(5) (optionally) Makes control plots
(6) (optionally) Plots and saves signal and control region templates for final fits.

Author(s): Raghav Kansal, Cristina Mantilla Suarez
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
from hist import Hist

import utils
import plotting

from numpy.typing import ArrayLike
from typing import Dict, List, Tuple
from inspect import cleandoc
from textwrap import dedent

import corrections
from hh_vars import (
    years,
    sig_key,
    data_key,
    qcd_key,
    bg_keys,
    samples,
    txbb_wps,
    jec_shifts,
    jmsr_shifts,
)
from utils import CUT_MAX_VAL

from pprint import pprint
from copy import deepcopy

import importlib

_ = importlib.reload(utils)
_ = importlib.reload(plotting)


# # Both Jet's Msds > 50 & at least one jet with Txbb > 0.8
# filters = [
#     [
#         ("('ak8FatJetMsd', '0')", ">=", 50),
#         ("('ak8FatJetMsd', '1')", ">=", 50),
#         ("('ak8FatJetParticleNetMD_Txbb', '0')", ">=", 0.8),
#     ],
#     [
#         ("('ak8FatJetMsd', '0')", ">=", 50),
#         ("('ak8FatJetMsd', '1')", ">=", 50),
#         ("('ak8FatJetParticleNetMD_Txbb', '1')", ">=", 0.8),
#     ],
# ]

filters = None

# {var: (bins, label)}
control_plot_vars = {
    "MET_pt": ([50, 0, 250], r"$p^{miss}_T$ (GeV)"),
    "DijetEta": ([50, -8, 8], r"$\eta^{jj}$"),
    "DijetPt": ([50, 0, 750], r"$p_T^{jj}$ (GeV)"),
    "DijetMass": ([50, 500, 3000], r"$m^{jj}$ (GeV)"),
    "bbFatJetEta": ([50, -2.4, 2.4], r"$\eta^{bb}$"),
    "bbFatJetPt": ([50, 300, 1300], r"$p^{bb}_T$ (GeV)"),
    "bbFatJetParticleNetMass": ([50, 0, 300], r"$m^{bb}_{reg}$ (GeV)"),
    "bbFatJetMsd": ([50, 0, 300], r"$m^{bb}_{msd}$ (GeV)"),
    "bbFatJetParticleNetMD_Txbb": ([50, 0.8, 1], r"$p^{bb}_{Txbb}$"),
    "VVFatJetEta": ([50, -2.4, 2.4], r"$\eta^{VV}$"),
    "VVFatJetPt": ([50, 300, 1300], r"$p^{VV}_T$ (GeV)"),
    "VVFatJetParticleNetMass": ([50, 0, 300], r"$m^{VV}_{reg}$ (GeV)"),
    "VVFatJetMsd": ([50, 0, 300], r"$m^{VV}_{msd}$ (GeV)"),
    "VVFatJetParticleNet_Th4q": ([50, 0, 1], r"Prob($H \to 4q$) vs Prob(QCD) (Non-MD)"),
    "VVFatJetParTMD_THWW4q": (
        [50, 0, 1],
        r"Prob($H \to VV \to 4q$) vs Prob(QCD) (Mass-Decorrelated)",
    ),
    "VVFatJetParTMD_probT": ([50, 0, 1], r"Prob(Top) (Mass-Decorrelated)"),
    "bbFatJetPtOverDijetPt": ([50, 0, 40], r"$p^{bb}_T / p_T^{jj}$"),
    "VVFatJetPtOverDijetPt": ([50, 0, 40], r"$p^{VV}_T / p_T^{jj}$"),
    "VVFatJetPtOverbbFatJetPt": ([50, 0.4, 2.0], r"$p^{VV}_T / p^{bb}_T$"),
    "nGoodMuons": ([3, 0, 3], r"# of Muons"),
    "nGoodElectrons": ([3, 0, 3], r"# of Electrons"),
    "nGoodJets": ([5, 0, 5], r"# of AK4 B-Jets"),
    "BDTScore": ([50, 0, 1], r"BDT Score"),
}


# {label: {cutvar: [min, max], ...}, ...}
selection_regions_year = {
    "pass": {
        "BDTScore": [0.986, CUT_MAX_VAL],
        "bbFatJetParticleNetMD_Txbb": ["HP", CUT_MAX_VAL],
    },
    "fail": {
        "bbFatJetParticleNetMD_Txbb": [0.8, "HP"],
    },
    "BDTOnly": {
        "BDTScore": [0.986, CUT_MAX_VAL],
    },
}

selection_regions_label = {"pass": "Pass", "fail": "Fail", "BDTOnly": "BDT Cut"}

selection_regions = {}

for year in years:
    sr = deepcopy(selection_regions_year)

    for region in sr:
        for cuts in sr[region]:
            for i in range(2):
                if sr[region][cuts][i] == "HP":
                    sr[region][cuts][i] = txbb_wps[year]["HP"]

    selection_regions[year] = sr


del year  # creates bugs later

scan_regions = {}

for bdtcut in np.arange(0.97, 1, 0.002):
    for bbcut in np.arange(0.97, 1, 0.002):
        cutstr = f"bdtcut_{bdtcut}_bbcut_{bbcut}"
        scan_regions[cutstr] = {
            "passCat1": {
                "BDTScore": [bdtcut, CUT_MAX_VAL],
                "bbFatJetParticleNetMD_Txbb": [bbcut, CUT_MAX_VAL],
            },
            "fail": {
                "bbFatJetParticleNetMD_Txbb": [0.8, bbcut],
            },
        }


# bb msd is final shape var
shape_var = ("bbFatJetMsd", r"$m^{bb}_{Reg}$ (GeV)")
shape_bins = [20, 50, 250]  # num bins, min, max
blind_window = [100, 150]


weight_shifts = {
    "pileup": [sig_key] + bg_keys,
    "PDFalphaS": [sig_key],
    "ISRPartonShower": [sig_key, "V+Jets"],
    "FSRPartonShower": [sig_key, "V+Jets"],
}

weight_labels = {
    "pileup": "Pileup",
    "PDFalphaS": "PDF",
    "ISRPartonShower": "ISR Parton Shower",
    "FSRPartonShower": "FSR Parton Shower",
}


def main(args):
    if not (args.control_plots or args.templates or args.scan):
        print("You need to pass at least one of --control-plots, --templates, or --scan")
        return

    # make plot, template dirs if needed
    _make_dirs(args)

    # save cutflow as pandas table
    cutflow = pd.DataFrame(index=list(samples.keys()))
    systematics = {}

    # utils.remove_empty_parquets(samples_dir, year)
    events_dict = utils.load_samples(args.data_dir, samples, args.year, filters)
    utils.add_to_cutflow(events_dict, "BDTPreselection", "weight", cutflow)

    # print weighted sample yields
    for sample in events_dict:
        tot_weight = np.sum(events_dict[sample]["weight"].values)
        print(f"Pre-selection {sample} yield: {tot_weight:.2f}")

    apply_weights(events_dict, args.year, cutflow)
    bb_masks = bb_VV_assignment(events_dict)
    events_dict[sig_key] = postprocess_lpsfs(events_dict[sig_key])
    cutflow.to_csv(f"{args.plot_dir}/cutflows/bdt_cutflow.csv")
    print(cutflow)

    print("\nLoading BDT predictions\n")
    load_bdt_preds(
        events_dict, args.year, args.data_dir, list(samples.keys()), jec_jmsr_shifts=True
    )

    if args.control_plots:
        print("\nMaking control plots\n")
        control_plots(
            events_dict, bb_masks, control_plot_vars, f"{args.plot_dir}/control_plots/", show=False
        )

    # BDT LP SF
    sel, cf = utils.make_selection(
        selection_regions[args.year]["BDTOnly"], events_dict, bb_masks, prev_cutflow=cutflow
    )
    lp_sf, unc, uncs = get_lpsf(events_dict[sig_key], sel[sig_key])
    print(f"BDT LP Scale Factor: {lp_sf:.2f} ± {unc:.2f}")
    print(uncs)
    systematics["lp_sf_unc"] = unc / lp_sf

    print(cf)
    check_weights(events_dict)

    if args.templates:
        print("\nMaking templates\n")

        templates = {}

        for jshift in [""] + jec_shifts + jmsr_shifts:
            print(jshift)
            ttemps, tsyst = get_templates(
                events_dict,
                bb_masks,
                args.year,
                selection_regions[args.year],
                shape_var,
                shape_bins,
                blind_window,
                plot_dir=args.plot_dir,
                prev_cutflow=cutflow,
                weight_shifts=weight_shifts,
                jshift=jshift,
                show=False,
            )

            templates = {**templates, **ttemps}
            systematics = {**systematics, **tsyst}

        print("\nSaving templates\n")
        save_templates(templates, blind_window, args.template_file, systematics)
        print("\nSaved templates\n")

    if args.scan:
        os.system(f"mkdir -p {args.template_file}")

        templates = {}

        for cutstr, region in scan_regions.items():
            print(cutstr)
            templates[cutstr] = get_templates(
                events_dict,
                bb_masks,
                region,
                shape_var,
                shape_bins,
                blind_window,
                plot_dir=args.plot_dir,
                prev_cutflow=cutflow,
                cutstr=cutstr,
            )
            save_templates(templates[cutstr], blind_window, f"{args.template_file}/{cutstr}.pkl")


def _make_dirs(args):
    if args.plot_dir != "":
        os.system(f"mkdir -p {args.plot_dir}/cutflows/")
        os.system(f"mkdir -p {args.plot_dir}/control_plots/")
        os.system(f"mkdir -p {args.plot_dir}/templates/")

    if args.template_file:
        from pathlib import Path

        path = Path(args.template_file)

        if path.exists() and not args.overwrite_template:
            print(
                "Template file already exists -- exiting! (Use --overwrite-template if you wish to overwrite)"
            )
            sys.exit()

        os.system(f"mkdir -p {path.parent}")


def apply_weights(
    events_dict: Dict[str, pd.DataFrame],
    year: str,
    cutflow: pd.DataFrame,
    weight_key: str = "finalWeight",
):
    """
    Applies (1) 2D trigger scale factors, (2) QCD scale facotr.

    Args:
        cutflow (pd.DataFrame): cutflow to which to add yields after scale factors.
        weight_key (str): column in which to store scaled weights in. Defaults to "finalWeight".

    """
    from coffea.lookup_tools.dense_lookup import dense_lookup

    with open(f"../corrections/trigEffs/{year}_combined.pkl", "rb") as filehandler:
        combined = pickle.load(filehandler)

    # sum over TH4q bins
    effs_txbb = combined["num"][:, sum, :, :] / combined["den"][:, sum, :, :]

    ak8TrigEffsLookup = dense_lookup(
        np.nan_to_num(effs_txbb.view(flow=False), 0), np.squeeze(effs_txbb.axes.edges)
    )

    for sample in events_dict:
        events = events_dict[sample]
        if sample == data_key:
            events[weight_key] = events["weight"]
        else:
            fj_trigeffs = ak8TrigEffsLookup(
                events["ak8FatJetParticleNetMD_Txbb"].values,
                events["ak8FatJetPt"].values,
                events["ak8FatJetMsd"].values,
            )
            # combined eff = 1 - (1 - fj1_eff) * (1 - fj2_eff)
            combined_trigEffs = 1 - np.prod(1 - fj_trigeffs, axis=1, keepdims=True)
            events[f"{weight_key}_noTrigEffs"] = events["weight"]
            events[weight_key] = events["weight"] * combined_trigEffs

    utils.add_to_cutflow(events_dict, "TriggerEffs", weight_key, cutflow)

    # calculate QCD scale factor
    if qcd_key in events_dict:
        trig_yields = cutflow["TriggerEffs"]
        non_qcd_bgs_yield = np.sum(
            [
                trig_yields[sample]
                for sample in events_dict
                if sample not in {sig_key, qcd_key, data_key}
            ]
        )
        QCD_SCALE_FACTOR = (trig_yields[data_key] - non_qcd_bgs_yield) / trig_yields[qcd_key]
        events_dict[qcd_key][weight_key] *= QCD_SCALE_FACTOR

        print(f"{QCD_SCALE_FACTOR = }")

        utils.add_to_cutflow(events_dict, "QCD SF", weight_key, cutflow)


def bb_VV_assignment(events_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Creates a dataframe of masks for extracting the bb or VV candidates.
    bb candidate is chosen based on higher Txbb score.

    Returns:
        Dict[str, pd.DataFrame]: ``bb_masks`` dict of boolean masks for each sample,
          of shape ``[num_events, 2]``.

    """
    bb_masks = {}

    for sample, events in events_dict.items():
        txbb = events["ak8FatJetParticleNetMD_Txbb"]
        bb_mask = txbb[0] >= txbb[1]
        bb_masks[sample] = pd.concat((bb_mask, ~bb_mask), axis=1)

    return bb_masks


def postprocess_lpsfs(
    events: pd.DataFrame,
    num_jets: int = 2,
    num_lp_sf_toys: int = 100,
    save_all: bool = True,
    weight_key: str = "finalWeight",
):
    """
    (1) Splits LP SFs into bb and VV based on gen matching.
    (2) Sets defaults for unmatched jets.
    (3) Cuts of SFs at 10 and normalises.
    """

    for jet in ["bb", "VV"]:
        # ignore rare case (~0.002%) where two jets are matched to same gen Higgs
        events.loc[np.sum(events[f"ak8FatJetH{jet}"], axis=1) > 1, f"ak8FatJetH{jet}"] = 0
        jet_match = events[f"ak8FatJetH{jet}"].astype(bool)

        # temp dict
        td = {}
        # td["tot_matched"] = np.sum(np.sum(jet_match))

        # defaults of 1 for jets which aren't matched to anything - i.e. no SF
        for key in ["lp_sf_nom", "lp_sf_sys_down", "lp_sf_sys_up"]:
            td[key] = np.ones(len(events))

        td["lp_sf_toys"] = np.ones((len(events), num_lp_sf_toys))

        # defaults of 0 - i.e. don't contribute to unc.
        for key in ["lp_sf_double_matched_event", "lp_sf_unmatched_quarks", "lp_sf_num_sjpt_gt350"]:
            td[key] = np.zeros(len(events))

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
                "lp_sf_num_sjpt_gt350",
            ]:
                td[key][jet_match[j]] = events[key][jet_match[j]][j]

        for key in ["lp_sf_nom", "lp_sf_toys", "lp_sf_sys_down", "lp_sf_sys_up"]:
            # cut off at 10
            td[key] = np.minimum(td[key], 10)
            # normalise
            td[key] = td[key] / np.mean(td[key], axis=0)

        # add to dataframe
        if save_all:
            td = pd.concat(
                [pd.DataFrame(v.reshape(v.shape[0], -1)) for k, v in td.items()],
                axis=1,
                keys=[f"{jet}_{key}" for key in td.keys()],
            )

            events = pd.concat((events, td), axis=1)
        else:
            key = "lp_sf_nom"
            events[f"{jet}_{key}"] = td[key]

    events[weight_key + "_noLP"] = events[weight_key]
    events[weight_key] = events[weight_key][0] * events["VV_lp_sf_nom"][0]
    events[weight_key + "_noTrigEffs"] = (
        events[weight_key + "_noTrigEffs"][0] * events["VV_lp_sf_nom"][0]
    )

    return events


def get_lpsf(events: pd.DataFrame, sel: np.ndarray = None, VV: bool = True):
    """Calculates LP SF and uncertainties in current phase space. ``postprocess_lpsfs`` must be called first."""

    jet = "VV" if VV else "bb"
    if sel is not None:
        events = events[sel]

    tot_matched = np.sum(np.sum(events[f"ak8FatJetH{jet}"].astype(bool)))

    weight = events["finalWeight_noLP"].values
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
        np.std(np.sum(weight[:, np.newaxis] * events[f"{jet}_lp_sf_toys"].values, axis=0))
        / tot_post
    )

    # fraction of subjets > 350 * 0.21 measured by CASE
    uncs["sj_pt_unc"] = (np.sum(events[f"{jet}_lp_sf_num_sjpt_gt350"][0]) / tot_matched) * 0.21

    if VV:
        num_prongs = events["ak8FatJetHVVNumProngs"][0]

        sj_matching_unc = np.sum(events[f"{jet}_lp_sf_double_matched_event"][0])
        for nump in range(2, 5):
            sj_matching_unc += (
                np.sum(events[f"{jet}_lp_sf_unmatched_quarks"][0][num_prongs == nump]) / nump
            )

        uncs["sj_matching_unc"] = sj_matching_unc / tot_matched
    else:
        num_prongs = 2
        uncs["sj_matching_unc"] = (
            (np.sum(events[f"{jet}_lp_sf_unmatched_quarks"][0]) / num_prongs)
            + np.sum(events[f"{jet}_lp_sf_double_matched_event"][0])
        ) / tot_matched

    tot_rel_unc = np.linalg.norm([val for val in uncs.values()])
    tot_unc = lp_sf * tot_rel_unc

    return lp_sf, tot_unc, uncs


def load_bdt_preds(
    events_dict: Dict[str, pd.DataFrame],
    year: str,
    bdt_preds_dir: str,
    bdt_sample_order: List[str],
    jec_jmsr_shifts: bool = False,
):
    """
    Loads the BDT scores for each event and saves in the dataframe in the "BDTScore" column.
    If ``jec_jmsr_shifts``, also loads BDT preds for every JEC / JMSR shift in MC.

    Args:
        bdt_preds (str): Path to the bdt_preds .npy file.
        bdt_sample_order (List[str]): Order of samples in the predictions file.

    """
    bdt_preds = np.load(f"{bdt_preds_dir}/{year}_preds.npy")

    if jec_jmsr_shifts:
        shift_preds = {
            jshift: np.load(f"{bdt_preds_dir}/{year}_preds_{jshift}.npy")
            for jshift in jec_shifts + jmsr_shifts
        }

    i = 0
    for sample in bdt_sample_order:
        events = events_dict[sample]
        num_events = len(events)
        events["BDTScore"] = bdt_preds[i : i + num_events]

        if jec_jmsr_shifts and sample != data_key:
            for jshift in jec_shifts + jmsr_shifts:
                events["BDTScore_" + jshift] = shift_preds[jshift][i : i + num_events]

        i += num_events

    assert i == len(bdt_preds), f"# events {i} != # of BDT preds {len(bdt_preds)}"


def control_plots(
    events_dict: Dict[str, pd.DataFrame],
    bb_masks: Dict[str, pd.DataFrame],
    control_plot_vars: Dict[str, Tuple],
    plot_dir: str,
    weight_key: str = "finalWeight",
    hists: Dict = {},
    show: bool = False,
):
    """
    Makes and plots histograms of each variable in ``control_plot_vars``.

    Args:
        control_plot_vars (Dict[str, Tuple]): Dictionary of variables to plot, formatted as
          {var1: ([num bins, min, max], label), var2...}.

    """

    from PyPDF2 import PdfMerger

    sig_scale = np.sum(events_dict[data_key][weight_key]) / np.sum(events_dict[sig_key][weight_key])
    print(f"{sig_scale = }")

    for var, (bins, label) in control_plot_vars.items():
        # print(var)
        if var not in hists:
            hists[var] = utils.singleVarHist(
                events_dict, var, bins, label, bb_masks, weight_key=weight_key
            )

    with open(f"{plot_dir}/hists.pkl", "wb") as f:
        pickle.dump(hists, f)

    merger_control_plots = PdfMerger()

    for var, var_hist in hists.items():
        name = f"{plot_dir}/{var}.pdf"
        plotting.ratioHistPlot(
            var_hist,
            bg_keys,
            name=name,
            sig_scale=sig_scale,
            show=show,
        )
        merger_control_plots.append(name)

    merger_control_plots.write(f"{plot_dir}/ControlPlots.pdf")
    merger_control_plots.close()

    return hists


def check_weights(events_dict):
    # Check for 0 weights - would be an issue for weight shifts
    print(
        "Any 0 weights:",
        np.any(
            [
                np.any(events["weight_nonorm"] == 0)
                for key, events in events_dict.items()
                if key != data_key
            ]
        ),
    )


def get_templates(
    events_dict: Dict[str, pd.DataFrame],
    bb_masks: Dict[str, pd.DataFrame],
    year: str,
    selection_regions: Dict[str, Dict],
    shape_var: Tuple[str],
    shape_bins: List[float],
    blind_window: List[float],
    plot_dir: str = "",
    prev_cutflow: pd.DataFrame = None,
    weight_key: str = "finalWeight",
    cutstr: str = "",
    weight_shifts: Dict = {},
    jshift: str = "",
    show: bool = False,
) -> Dict[str, Hist]:
    """
    (1) Makes histograms for each region in the ``selection_regions`` dictionary,
    (2) Applies the Txbb scale factor in the pass region,
    (3) Calculates trigger uncertainty,
    (4) Calculates weight variations if ``weight_shifts`` is not empty (and ``jshift`` is ""),
    (5) Takes JEC / JSMR shift into account if ``jshift`` is not empty,
    (6) Saves a plot of each (if ``plot_dir`` is not "").

    Args:
        selection_region (Dict[str, Dict]): Dictionary of cuts for each region
          formatted as {region1: {cutvar1: [min, max], ...}, ...}.
        shape_var (Tuple[str]): final shape var: (var name, var plotting label).
        shape_bins (List[float]): binning for shape var: [num bins, min, max].
        blind_window (List[float]): signal window to blind: [min, max] (min, max should be bin edges).
        cutstr (str): optional string to add to plot file names.

    Returns:
        Dict[str, Hist]: dictionary of templates, saved as hist.Hist objects.

    """

    do_jshift = jshift != ""
    jlabel = "" if not do_jshift else "_" + jshift
    templates, systematics = {}, {}

    var = shape_var[0]

    # print(selection_regions)

    for label, region in selection_regions.items():
        pass_region = label.startswith("pass")

        if label == "BDTOnly":
            continue

        if not do_jshift:
            print(label)

        sel, cf = utils.make_selection(
            region, events_dict, bb_masks, prev_cutflow=prev_cutflow, jshift=jshift
        )
        cf.to_csv(f"{plot_dir}/cutflows/{label}_cutflow{jlabel}.csv")

        if not do_jshift:
            systematics[label] = {}
            systematics[label]["trig_unc"] = corrections.get_uncorr_trig_eff_unc(
                events_dict, bb_masks, year, sel
            )

        # ParticleNetMD Txbb SFs
        sig_events = deepcopy(events_dict[sig_key][sel[sig_key]])
        sig_bb_mask = bb_masks[sig_key][sel[sig_key]]
        if pass_region:
            corrections.apply_txbb_sfs(sig_events, sig_bb_mask, year, weight_key)

        # set up samples
        hist_samples = list(events_dict.keys())

        if not do_jshift:
            for shift in ["down", "up"]:
                if pass_region:
                    hist_samples.append(f"{sig_key}_txbb_{shift}")

                for wshift, wsamples in weight_shifts.items():
                    for wsample in wsamples:
                        hist_samples.append(f"{wsample}_{wshift}_{shift}")

        h = (
            Hist.new.StrCat(hist_samples, name="Sample")
            .Reg(*shape_bins, name=var, label=shape_var[1])
            .Weight()
        )

        for sample in events_dict:
            events = sig_events if sample == sig_key else events_dict[sample][sel[sample]]

            bb_mask = bb_masks[sample][sel[sample]]
            fill_var = (
                var if sample == data_key or not do_jshift else utils.check_get_jec_var(var, jshift)
            )
            fill_data = {var: utils.get_feat(events, fill_var, bb_mask)}
            weight = events[weight_key].values.squeeze()
            h.fill(Sample=sample, **fill_data, weight=weight)

            if not do_jshift:
                # add weight variations
                for wshift, wsamples in weight_shifts.items():
                    if sample in wsamples:
                        # print(wshift)
                        for skey, shift in [("Down", "down"), ("Up", "up")]:
                            # reweight based on diff between up/down and nominal weights
                            sweight = (
                                weight
                                * (
                                    events[f"weight_{wshift}{skey}"][0] / events["weight_nonorm"]
                                ).values.squeeze()
                            )
                            h.fill(Sample=f"{sample}_{wshift}_{shift}", **fill_data, weight=sweight)

        if pass_region:
            # blind Higgs mass window in pass region in data
            utils.blindBins(h, blind_window, data_key)

        if pass_region and not do_jshift:
            # ParticleNetMD Txbb SFs
            fill_data = {var: utils.get_feat(sig_events, var, sig_bb_mask)}
            for shift in ["down", "up"]:
                h.fill(
                    Sample=f"{sig_key}_txbb_{shift}",
                    **fill_data,
                    weight=sig_events[f"{weight_key}_txbb_{shift}"],
                )

        templates[label + jlabel] = h

        if plot_dir != "":
            sig_scale = utils.getSignalPlotScaleFactor(events_dict, selection=sel)
            plot_params = {"hists": h, "bg_keys": bg_keys, "sig_scale": sig_scale / 2, "show": show}

            if not do_jshift:
                title = f"{selection_regions_label[label]} Region Pre-Fit Shapes"
            else:
                title = f"{selection_regions_label[label]} Region {jshift} Shapes"

            plotting.ratioHistPlot(
                **plot_params,
                title=title,
                name=f"{plot_dir}/templates/{cutstr}{label}_region_bb_mass{jlabel}.pdf",
            )

            if not do_jshift:
                for wshift, wsamples in weight_shifts.items():
                    wlabel = weight_labels[wshift]

                    if wsamples == [sig_key]:
                        plotting.ratioHistPlot(
                            **plot_params,
                            sig_err=wshift,
                            title=f"{selection_regions_label[label]} Region {wlabel} Unc. Shapes",
                            name=f"{plot_dir}/templates/{cutstr}{label}_region_bb_mass_{wshift}.pdf",
                        )
                    else:
                        for skey, shift in [("Down", "down"), ("Up", "up")]:
                            plotting.ratioHistPlot(
                                **plot_params,
                                variation=(wshift, shift, wsamples),
                                title=f"{selection_regions_label[label]} Region {wlabel} Unc. {skey} Shapes",
                                name=f"{plot_dir}/templates/{cutstr}{label}_region_bb_mass_{wshift}_{shift}.pdf",
                            )

            if pass_region and not do_jshift:
                plotting.ratioHistPlot(
                    **plot_params,
                    sig_err="txbb",
                    title=rf"{selection_regions_label[label]} Region $T_{{Xbb}}$ Shapes",
                    name=f"{plot_dir}/templates/{cutstr}{label}_region_bb_mass_txbb.pdf",
                )

    return templates, systematics


def save_templates(
    templates: Dict[str, Hist],
    blind_window: List[float],
    template_file: str,
    systematics: Dict = None,
):
    """Creates blinded copies of each region's templates and saves a pickle of the templates"""

    from copy import deepcopy

    for label, template in list(templates.items()):
        blinded_template = deepcopy(template)
        utils.blindBins(blinded_template, blind_window)
        templates[f"{label}Blinded"] = blinded_template

    if systematics is not None:
        templates["systematics"] = systematics

    with open(template_file, "wb") as f:
        pickle.dump(templates, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        default="../../../../data/skimmer/Apr28/",
        help="path to skimmed parquet",
        type=str,
    )

    parser.add_argument(
        "--year",
        default="2017",
        choices=["2016", "2016APV", "2017", "2018"],
        type=str,
    )

    parser.add_argument(
        "--bdt-preds",
        help="path to bdt predictions, if empty, don't load",
        default="../../../../data/skimmer/Apr28/absolute_weights_preds.npy",
        type=str,
    )

    parser.add_argument(
        "--plot-dir",
        help="If making control or template plots, path to directory to save them in",
        default="",
        type=str,
    )

    parser.add_argument(
        "--template-file",
        help="If saving templates, path to file to save them in. If scanning, directory to save in.",
        default="",
        type=str,
    )

    utils.add_bool_arg(parser, "control-plots", "make control plots", default=False)
    utils.add_bool_arg(parser, "templates", "save m_bb templates using bdt cut", default=False)
    utils.add_bool_arg(
        parser, "overwrite-template", "if template file already exists, overwrite it", default=False
    )
    utils.add_bool_arg(parser, "scan", "Scan BDT + Txbb cuts and save templates", default=False)

    args = parser.parse_args()
    main(args)
