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

from typing import Dict, List, Tuple
from inspect import cleandoc
from textwrap import dedent

from sample_labels import sig_key, data_key, qcd_key, bg_keys, samples
from utils import CUT_MAX_VAL

from pprint import pprint

import importlib

_ = importlib.reload(utils)
_ = importlib.reload(plotting)


# Both Jet's Msds > 50 & at least one jet with Txbb > 0.8
filters = [
    [
        ("('ak8FatJetMsd', '0')", ">=", 50),
        ("('ak8FatJetMsd', '1')", ">=", 50),
        ("('ak8FatJetParticleNetMD_Txbb', '0')", ">=", 0.8),
    ],
    [
        ("('ak8FatJetMsd', '0')", ">=", 50),
        ("('ak8FatJetMsd', '1')", ">=", 50),
        ("('ak8FatJetParticleNetMD_Txbb', '1')", ">=", 0.8),
    ],
]

# {var: (bins, label)}
control_plot_vars = {
    "MET_pt": ([50, 0, 250], r"$p^{miss}_T$ (GeV)"),
    "DijetEta": ([50, -8, 8], r"$\eta^{jj}$"),
    "DijetPt": ([50, 0, 750], r"$p_T^{jj}$ (GeV)"),
    "DijetMass": ([50, 0, 2500], r"$m^{jj}$ (GeV)"),
    "bbFatJetEta": ([50, -3, 3], r"$\eta^{bb}$"),
    "bbFatJetPt": ([50, 200, 1000], r"$p^{bb}_T$ (GeV)"),
    "bbFatJetMsd": ([50, 20, 250], r"$m^{bb}$ (GeV)"),
    "bbFatJetParticleNetMD_Txbb": ([50, 0, 1], r"$p^{bb}_{Txbb}$"),
    "VVFatJetEta": ([50, -3, 3], r"$\eta^{VV}$"),
    "VVFatJetPt": ([50, 200, 1000], r"$p^{VV}_T$ (GeV)"),
    "VVFatJetMsd": ([50, 20, 500], r"$m^{VV}$ (GeV)"),
    "VVFatJetParticleNet_Th4q": ([50, 0, 1], r"Probability($H \to 4q$)"),
    "VVFatJetParticleNetHWWMD_THWW4q": ([50, 0, 1], r"Probability($H \to VV \to 4q$)"),
    "bbFatJetPtOverDijetPt": ([50, 0, 40], r"$p^{bb}_T / p_T^{jj}$"),
    "VVFatJetPtOverDijetPt": ([50, 0, 40], r"$p^{VV}_T / p_T^{jj}$"),
    "VVFatJetPtOverbbFatJetPt": ([50, 0.4, 2.5], r"$p^{VV}_T / p^{bb}_T$"),
    "BDTScore": ([50, 0, 1], r"BDT Score"),
}

# {label: {cutvar: [min, max], ...}, ...}
selection_regions = {
    "passCat1": {
        "BDTScore": [0.986, CUT_MAX_VAL],
        "bbFatJetParticleNetMD_Txbb": [0.976, CUT_MAX_VAL],
    },
    "fail": {
        "bbFatJetParticleNetMD_Txbb": [0.8, 0.976],
    },
}

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
shape_var = ("bbFatJetMsd", r"$m^{bb}$ (GeV)")
shape_bins = [20, 50, 250]  # num bins, min, max
blind_window = [100, 150]


# # for local interactive testing
# args = type("test", (object,), {})()
# args.data_dir = "../../../../data/skimmer/Apr28/"
# args.plot_dir = "../../../PostProcess/plots/04_07_scan"
# args.year = "2017"
# args.bdt_preds = f"{args.data_dir}/absolute_weights_preds.npy"
# args.template_file = "templates/04_07_scan"
# args.overwrite_template = True
# args.control_plots = False
# args.templates = False
# args.scan = True


def main(args):
    if not (args.control_plots or args.templates or args.scan):
        print("You need to pass at least one of --control-plots, --templates, or --scan")
        return

    # make plot, template dirs if needed
    make_dirs(args)

    # save cutflow as pandas table
    overall_cutflow = pd.DataFrame(index=list(samples.keys()))

    events_dict = utils.load_samples(args.data_dir, samples, args.year, filters)
    utils.add_to_cutflow(events_dict, "BDTPreselection", "weight", overall_cutflow)
    print("\nLoaded Events\n")

    apply_weights(events_dict, args.year, overall_cutflow)
    bb_masks = bb_VV_assignment(events_dict)
    derive_variables(events_dict, bb_masks)
    print("\nWeights, masks, derived variables\n")

    if args.bdt_preds != "":
        load_bdt_preds(events_dict, args.bdt_preds, bdt_sample_order)

    if args.control_plots:
        control_plots(events_dict, bb_masks, control_plot_vars, args.plot_dir)
        print("\nMade control plots\n")

    if args.templates:
        if args.bdt_preds != "":
            templates = get_templates(
                events_dict,
                bb_masks,
                selection_regions,
                shape_var,
                shape_bins,
                blind_window,
                plot_dir=args.plot_dir,
                prev_cutflow=overall_cutflow,
            )
            save_templates(templates, blind_window, args.template_file)
            print("\nMade and saved templates\n")
        else:
            print("bdt-preds need to be given for templates")

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
                prev_cutflow=overall_cutflow,
                cutstr=cutstr,
            )
            save_templates(templates[cutstr], blind_window, f"{args.template_file}/{cutstr}.pkl")


def make_dirs(args):
    if args.plot_dir:
        os.system(f"mkdir -p {args.plot_dir}")

    if args.template_file:
        from pathlib import Path

        path = Path(args.template_file)

        if path.exists() and not args.overwrite_template:
            print("Template file already exists -- exiting!")
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

    with open(
        f"../corrections/trigEffs/AK8JetHTTriggerEfficiency_{year}.hist", "rb"
    ) as filehandler:
        ak8TrigEffs = pickle.load(filehandler)

    ak8TrigEffsLookup = dense_lookup(
        np.nan_to_num(ak8TrigEffs.view(flow=False), 0), np.squeeze(ak8TrigEffs.axes.edges)
    )

    for sample in events_dict:
        events = events_dict[sample]
        if sample == data_key:
            events[weight_key] = events["weight"]
        else:
            fj_trigeffs = ak8TrigEffsLookup(
                events["ak8FatJetPt"].values, events["ak8FatJetMsd"].values
            )
            # combined eff = 1 - (1 - fj1_eff) * (1 - fj2_eff)
            combined_trigEffs = 1 - np.prod(1 - fj_trigeffs, axis=1, keepdims=True)
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
    events: pd.DataFrame, num_jets: int = 2, num_lp_sf_toys: int = 100, save_all: bool = True
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

    return events


def get_lpsf(events: pd.DataFrame, sel: np.ndarray = None, VV: bool = True):
    """Calculates LP SF and uncertainties in current phase space. ``postprocess_lpsfs`` must be called first."""

    jet = "VV" if VV else "bb"
    if sel is not None:
        events = events[sel]

    tot_matched = np.sum(np.sum(events[f"ak8FatJetH{jet}"].astype(bool)))

    weight = events["finalWeight_preLP"].values
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


def derive_variables(events_dict: Dict[str, pd.DataFrame], bb_masks: Dict[str, pd.DataFrame]):
    """Derives more dijet kinematic variables for control plots."""
    for sample, events in events_dict.items():
        bb_mask = bb_masks[sample]

        fatjet_vectors = utils.make_vector(events, "ak8FatJet")
        Dijet = fatjet_vectors[:, 0] + fatjet_vectors[:, 1]

        events["DijetPt"] = Dijet.pt
        events["DijetMass"] = Dijet.M
        events["DijetEta"] = Dijet.eta

        events["bbFatJetPtOverDijetPt"] = (
            utils.get_feat(events, "bbFatJetPt", bb_mask) / events["DijetPt"]
        )
        events["VVFatJetPtOverDijetPt"] = (
            utils.get_feat(events, "VVFatJetPt", bb_mask) / events["DijetPt"]
        )
        events["VVFatJetPtOverbbFatJetPt"] = utils.get_feat(
            events, "VVFatJetPt", bb_mask
        ) / utils.get_feat(events, "bbFatJetPt", bb_mask)


def load_bdt_preds(
    events_dict: Dict[str, pd.DataFrame], bdt_preds: str, bdt_sample_order: List[str]
):
    """
    Loads the BDT scores for each event and saves in the dataframe in the "BDTScore" column.

    Args:
        bdt_preds (str): Path to the bdt_preds .npy file.
        bdt_sample_order (List[str]): Order of samples in the predictions file.

    """
    bdt_preds = np.load(bdt_preds)

    i = 0
    for sample in bdt_sample_order:
        events = events_dict[sample]
        num_events = len(events)
        events["BDTScore"] = bdt_preds[i : i + num_events]
        i += num_events


def control_plots(
    events_dict: Dict[str, pd.DataFrame],
    bb_masks: Dict[str, pd.DataFrame],
    control_plot_vars: Dict[str, Tuple],
    plot_dir: str,
    weight_key: str = "finalWeight",
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

    hists = {}

    for var, (bins, label) in control_plot_vars.items():
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
        )
        merger_control_plots.append(name)

    merger_control_plots.write(f"{plot_dir}/ControlPlots.pdf")
    merger_control_plots.close()


def get_templates(
    events_dict: Dict[str, pd.DataFrame],
    bb_masks: Dict[str, pd.DataFrame],
    selection_regions: Dict[str, Dict],
    shape_var: Tuple[str],
    shape_bins: List[float],
    blind_window: List[float],
    plot_dir: str = "",
    prev_cutflow: pd.DataFrame = None,
    weight_key: str = "finalWeight",
    cutstr: str = "",
) -> Dict[str, Hist]:
    """
    (1) Makes histograms for each region in the ``selection_regions`` dictionary,
    (2) Saves a plot of each (if ``plot_dir`` is not ""),
    (3) And for the Pass region calculates the signal and data-driven bg estimate.

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

    # print(selection_regions)


    print(bg_keys)
    selections, cutflows, templates = {}, {}, {}

    for label, region in selection_regions.items():
        # print(region)
        pass_region = label.startswith("pass")

        sel, cf = utils.make_selection(region, events_dict, bb_masks, prev_cutflow=prev_cutflow)
        if plot_dir != "":
            cf.to_csv(f"{plot_dir}/{cutstr}{label}_cutflow.csv")

        template = utils.singleVarHist(
            events_dict,
            shape_var[0],
            shape_bins,
            shape_var[1],
            bb_masks,
            weight_key=weight_key,
            selection=sel,
            blind_region=blind_window if pass_region else None,
        )

        sig_scale = utils.getSignalPlotScaleFactor(events_dict, selection=sel)

        if plot_dir != "":
            plotting.ratioHistPlot(
                template,
                bg_keys,
                name=f"{plot_dir}/{cutstr}{label}_region_bb_mass.pdf",
                sig_scale=sig_scale / 2,
                show=False,
            )

        if pass_region:
            pass_sig_yield, pass_bg_yield = utils.getSigSidebandBGYields(
                shape_var[0],
                blind_window,
                events_dict,
                bb_masks,
                selection=sel,
            )

            # cutstrn = cutstr + "\n" if cutstr != "" else ""
            print(
                cleandoc(
                    f"""Pass region signal yield: {pass_sig_yield}
                    background yield from data in sidebands: {pass_bg_yield}"""
                )
            )

        selections[label] = sel
        cutflows[label] = cf
        templates[label] = template

    return templates


def save_templates(templates: Dict[str, Hist], blind_window: List[float], template_file: str):
    """Creates blinded copies of each region's templates and saves a pickle of the templates"""

    from copy import deepcopy

    for label, template in list(templates.items()):
        blinded_template = deepcopy(template)
        utils.blindBins(blinded_template, blind_window)
        templates[f"{label}Blinded"] = blinded_template

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
