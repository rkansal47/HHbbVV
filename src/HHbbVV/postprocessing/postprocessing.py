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

from dataclasses import dataclass, field
from collections import OrderedDict

import os
import sys
import pickle, json
import itertools

import numpy as np
import pandas as pd

# from pandas.errors import SettingWithCopyWarning
import hist
from hist import Hist

import utils
import plotting

from numpy.typing import ArrayLike
from typing import Dict, List, Tuple
from inspect import cleandoc
from textwrap import dedent

import corrections
from corrections import postprocess_lpsfs, get_lpsf
from hh_vars import (
    years,
    nonres_sig_keys,
    res_sig_keys,
    data_key,
    qcd_key,
    bg_keys,
    samples,
    nonres_samples,
    res_samples,
    BDT_sample_order,
    txbb_wps,
    jec_shifts,
    jmsr_shifts,
)
from utils import CUT_MAX_VAL

from pprint import pprint
from copy import deepcopy
import warnings

import argparse


# ignore these because they don't seem to apply
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class ShapeVar:
    """Class to store attributes of the variable to be fit on.

    Args:
        var (str): variable name
        label (str): variable label
        bins (List[int]): bins
        reg (bool, optional): Use a regular axis or variable binning. Defaults to True.
        blind_window (List[int], optional): if blinding . Defaults to None.
    """

    def __init__(
        self,
        var: str,
        label: str,
        bins: List[int],
        reg: bool = True,
        blind_window: List[int] = None,
    ):
        self.var = var
        self.label = label
        self.blind_window = blind_window

        # create axis used for histogramming
        if reg:
            self.axis = hist.axis.Regular(*bins, name=var, label=label)
        else:
            self.axis = hist.axis.Variable(bins, name=var, label=label)


@dataclass
class Syst:
    samples: List[str] = None
    years: List[str] = field(default_factory=lambda: years)
    label: str = None


@dataclass
class Region:
    cuts: Dict = None
    label: str = None


# Both Jet's Regressed Mass above 50, electron veto included in new samples
new_filters = [
    [
        ("('ak8FatJetParticleNetMass', '0')", ">=", 50),
        ("('ak8FatJetParticleNetMass', '1')", ">=", 50),
    ],
]

old_filters = [
    [
        ("('ak8FatJetParticleNetMass', '0')", ">=", 50),
        ("('ak8FatJetParticleNetMass', '1')", ">=", 50),
        ("('nGoodElectrons', '0')", "==", 0),
    ],
]

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


def get_nonres_selection_regions(
    year: str,
    txbb_wp: str = "HP",
    bdt_wp: float = 0.99,
):
    pt_cuts = [300, CUT_MAX_VAL]
    txbb_cut = txbb_wps[year][txbb_wp]

    return {
        # {label: {cutvar: [min, max], ...}, ...}
        "pass": Region(
            cuts={
                "bbFatJetPt": pt_cuts,
                "VVFatJetPt": pt_cuts,
                "BDTScore": [bdt_wp, CUT_MAX_VAL],
                "bbFatJetParticleNetMD_Txbb": [txbb_cut, CUT_MAX_VAL],
            },
            label="Pass",
        ),
        "fail": Region(
            cuts={
                "bbFatJetPt": pt_cuts,
                "VVFatJetPt": pt_cuts,
                "bbFatJetParticleNetMD_Txbb": [0.8, txbb_cut],
            },
            label="Fail",
        ),
        "lpsf": Region(
            cuts={  # cut for which LP SF is calculated
                "BDTScore": [bdt_wp, CUT_MAX_VAL],
            },
            label="LP SF Cut",
        ),
    }


def get_res_selection_regions(
    year: str, mass_window: List[float] = [110, 145], txbb_wp: str = "HP", thww_wp: float = 0.96
):
    pt_cuts = [300, CUT_MAX_VAL]
    mwsize = mass_window[1] - mass_window[0]
    mw_sidebands = [
        [mass_window[0] - mwsize / 2, mass_window[0]],
        [mass_window[1], mass_window[1] + mwsize / 2],
    ]
    txbb_cut = txbb_wps[year][txbb_wp]

    return {
        # "unblinded" regions:
        "pass": Region(
            cuts={
                "bbFatJetPt": pt_cuts,
                "VVFatJetPt": pt_cuts,
                "bbFatJetParticleNetMass": mass_window,
                "bbFatJetParticleNetMD_Txbb": [txbb_cut, CUT_MAX_VAL],
                "VVFatJetParTMD_THWWvsT": [thww_wp, CUT_MAX_VAL],
            },
            label="Pass",
        ),
        "fail": Region(
            cuts={
                "bbFatJetPt": pt_cuts,
                "VVFatJetPt": pt_cuts,
                "bbFatJetParticleNetMass": mass_window,
                "bbFatJetParticleNetMD_Txbb": [0.8, txbb_cut],
                "VVFatJetParTMD_THWWvsT": [-CUT_MAX_VAL, thww_wp],
            },
            label="Fail",
        ),
        # "blinded" validation regions:
        "passBlinded": Region(
            cuts={
                "bbFatJetPt": pt_cuts,
                "VVFatJetPt": pt_cuts,
                "bbFatJetParticleNetMass": mw_sidebands,
                "bbFatJetParticleNetMD_Txbb": [txbb_cut, CUT_MAX_VAL],
                "VVFatJetParTMD_THWWvsT": [thww_wp, CUT_MAX_VAL],
            },
            label="Validation Pass",
        ),
        "failBlinded": Region(
            cuts={
                "bbFatJetPt": pt_cuts,
                "VVFatJetPt": pt_cuts,
                "bbFatJetParticleNetMass": mw_sidebands,
                "bbFatJetParticleNetMD_Txbb": [0.8, txbb_cut],
                "VVFatJetParTMD_THWWvsT": [-CUT_MAX_VAL, thww_wp],
            },
            label="Validation Fail",
        ),
        # cut for which LP SF is calculated
        "lpsf": Region(
            cuts={"VVFatJetParTMD_THWWvsT": [thww_wp, CUT_MAX_VAL]},
            label="LP SF Cut",
        ),
    }


# fitting on bb regressed mass for nonresonant
nonres_shape_vars = [
    ShapeVar(
        "bbFatJetParticleNetMass",
        r"$m^{bb}_{Reg}$ (GeV)",
        [20, 50, 250],
        reg=True,
        blind_window=[100, 150],
    )
]


# fitting on VV regressed mass + dijet mass for resonant
res_shape_vars = [
    ShapeVar(
        "VVFatJetParticleNetMass",
        r"$m^{VV}_{Reg}$ (GeV)",
        list(range(50, 110, 10)) + list(range(110, 200, 15)) + [200, 220, 250],
        reg=False,
    ),
    ShapeVar(
        "DijetMass",
        r"$m^{jj}$ (GeV)",
        list(range(800, 1400, 100)) + [1400, 1600, 2000, 3000, 4400],
        reg=False,
    ),
]

nonres_scan_cuts = ["txbb", "bdt"]
res_scan_cuts = ["txbb", "thww"]


# TODO: check which of these applies to resonant as well
weight_shifts = {
    "pileup": Syst(samples=nonres_sig_keys + res_sig_keys + bg_keys, label="Pileup"),
    "PDFalphaS": Syst(samples=nonres_sig_keys, label="PDF"),
    "QCDscale": Syst(samples=nonres_sig_keys, label="QCDscale"),
    "ISRPartonShower": Syst(samples=nonres_sig_keys + ["V+Jets"], label="ISR Parton Shower"),
    "FSRPartonShower": Syst(samples=nonres_sig_keys + ["V+Jets"], label="FSR Parton Shower"),
    "L1EcalPrefiring": Syst(
        samples=nonres_sig_keys + res_sig_keys + bg_keys,
        years=["2016APV", "2016", "2017"],
        label="L1 ECal Prefiring",
    ),
    # "top_pt": ["TT"],
}


def main(args):
    shape_vars, scan, scan_cuts, scan_wps = _init(args)
    sig_keys, sig_samples, bg_keys, bg_samples = _process_samples(args)
    all_samples = sig_keys + bg_keys
    _make_dirs(args, scan, scan_cuts, scan_wps)  # make plot, template dirs if needed
    cutflow = pd.DataFrame(index=all_samples)  # save cutflow as pandas table

    # utils.remove_empty_parquets(samples_dir, year)
    events_dict = _load_samples(args, bg_samples, sig_samples, cutflow)

    for sample in events_dict:
        tot_weight = np.sum(events_dict[sample]["weight"].values)
        print(f"Pre-selection {sample} yield: {tot_weight:.2f}")

    bb_masks = bb_VV_assignment(events_dict)

    if "finalWeight_noTrigEffs" not in events_dict[list(events_dict.keys())[0]]:
        # trigger effs (if not already from processor)
        apply_weights(events_dict, args.year, cutflow)
        print("\nCutflow\n", cutflow)
        # THWW score vs Top (if not already from processor)
        derive_variables(events_dict)

    if args.plot_dir != "":
        cutflow.to_csv(f"{args.plot_dir}/preselection_cutflow.csv")

    print("\nCutflow\n", cutflow)

    # Load BDT Scores
    bdt_preds_dir = None
    if not args.resonant:
        print("\nLoading BDT predictions")
        bdt_preds_dir = f"{args.data_dir}/inferences/" if args.bdt_preds == "" else args.bdt_preds
        load_bdt_preds(
            events_dict,
            args.year,
            bdt_preds_dir,
            all_samples,
            jec_jmsr_shifts=True,
        )
        print("Loaded BDT preds\n")

    # Control plots
    if args.control_plots:
        print("\nMaking control plots\n")
        control_plots(
            events_dict,
            bb_masks,
            sig_keys,
            control_plot_vars,
            f"{args.plot_dir}/ControlPlots/{args.year}/",
            args.year,
            # sig_splits=sig_splits,
            show=False,
        )

    if args.templates:
        for wps in scan_wps:  # if not scanning, this will just be a single WP
            cutstr = "_".join([f"{cut}_{wp}" for cut, wp in zip(scan_cuts, wps)]) if scan else ""
            template_dir = f"{args.template_dir}/{cutstr}/{args.templates_name}/"

            cutargs = {f"{cut}_wp": wp for cut, wp in zip(scan_cuts, wps)}
            selection_regions = (
                get_nonres_selection_regions(args.year, **cutargs)
                if not args.resonant
                else get_res_selection_regions(args.year, **cutargs)
            )

            print(cutstr)
            # load pre-calculated systematics and those for different years if saved already
            systs_file = f"{template_dir}/systematics.json"
            systematics = _check_load_systematics(systs_file, args.year)

            # Lund plane SFs
            lpsfs(
                events_dict,
                bb_masks,
                sig_keys,
                sig_samples,
                cutflow,
                selection_regions["lpsf"],
                systematics,
                args.lp_sf_all_years,
                BDT_sample_order,
                bdt_preds_dir,
                template_dir,
                systs_file,
                args.signal_data_dir,
            )

            # Check for 0 weights - would be an issue for weight shifts
            check_weights(events_dict)

            print("\nMaking templates")
            templates = {}

            jshifts = [""] + jec_shifts + jmsr_shifts if args.do_jshifts else [""]
            for jshift in jshifts:
                print(jshift)
                plot_dir = (
                    f"{args.plot_dir}/templates/{cutstr}/" f"{'jshifts/' if jshift != '' else ''}"
                    if args.plot_dir != ""
                    else ""
                )
                temps = get_templates(
                    events_dict,
                    bb_masks,
                    args.year,
                    sig_keys,
                    selection_regions,
                    shape_vars,
                    systematics,
                    template_dir,
                    bg_keys=bg_keys,
                    plot_dir=plot_dir,
                    prev_cutflow=cutflow,
                    # sig_splits=sig_splits,
                    weight_shifts=weight_shifts,
                    jshift=jshift,
                    blind_pass=True if args.resonant else False,
                    show=False,
                    plot_shifts=args.plot_shifts,
                )
                templates = {**templates, **temps}

            print("\nSaving templates")
            save_templates(
                templates, f"{template_dir}/{args.year}_templates.pkl", args.resonant, shape_vars
            )

            with open(systs_file, "w") as f:
                json.dump(systematics, f)


def _init(args):
    if not (args.control_plots or args.templates or args.scan):
        print("You need to pass at least one of --control-plots, --templates, or --scan")
        return

    if not args.resonant:
        scan = len(args.nonres_txbb_wp) > 1 or len(args.nonres_bdt_wp) > 1
        scan_wps = list(itertools.product(args.nonres_txbb_wp, args.nonres_bdt_wp))
        scan_cuts = nonres_scan_cuts
        shape_vars = nonres_shape_vars
    else:
        scan = len(args.res_txbb_wp) > 1 or len(args.res_thww_wp) > 1
        scan_wps = list(itertools.product(args.res_txbb_wp, args.res_thww_wp))
        scan_cuts = res_scan_cuts
        shape_vars = res_shape_vars

    return shape_vars, scan, scan_cuts, scan_wps


def _process_samples(args):
    sig_samples = res_samples if args.resonant else nonres_samples

    if args.read_sig_samples:
        # read all signal samples in directory
        read_year = args.year if args.year != "all" else "2017"
        read_samples = os.listdir(f"{args.signal_data_dir}/{args.year}")
        sig_samples = OrderedDict()
        for sample in read_samples:
            if sample.startswith("NMSSM_XToYHTo2W2BTo4Q2B_MX-"):
                mY = int(sample.split("-")[-1])
                mX = int(sample.split("NMSSM_XToYHTo2W2BTo4Q2B_MX-")[1].split("_")[0])

                sig_samples[f"X[{mX}]->H(bb)Y[{mY}](VV)"] = sample

    if args.sig_samples is not None:
        for sig_key, sample in list(sig_samples.items()):
            if sample not in args.sig_samples:
                del sig_samples[sig_key]

    bg_samples = deepcopy(samples)
    for bg_key, sample in list(bg_samples.items()):
        if bg_key not in args.bg_keys and bg_key != data_key:
            del bg_samples[bg_key]

    if not args.resonant:
        for key in sig_samples.copy():
            keep = False
            for bkeys in BDT_sample_order:
                if bkeys in key:
                    keep = True
            if not keep:
                del sig_samples[key]

        for key in bg_samples.copy():
            if key not in BDT_sample_order:
                del bg_samples[key]

    if not args.data:
        del bg_samples[data_key]

    sig_keys = list(sig_samples.keys())
    bg_keys = list(bg_samples.keys())

    print("Sig keys: ", sig_keys)
    print("Sig samples: ", sig_samples)
    print("BG keys: ", bg_keys)
    print("BG Samples: ", bg_samples)

    return sig_keys, sig_samples, bg_keys, bg_samples


def _make_dirs(args, scan, scan_cuts, scan_wps):
    if args.plot_dir != "":
        args.plot_dir = f"{args.plot_dir}/{args.year}/"
        os.system(f"mkdir -p {args.plot_dir}")

        if args.control_plots:
            os.system(f"mkdir -p {args.plot_dir}/control_plots/")

        os.system(f"mkdir -p {args.plot_dir}/templates/")

        if scan:
            for wps in scan_wps:
                cutstr = "_".join([f"{cut}_{wp}" for cut, wp in zip(scan_cuts, wps)])
                os.system(f"mkdir -p {args.plot_dir}/templates/{cutstr}/")
                os.system(f"mkdir -p {args.plot_dir}/templates/{cutstr}/wshifts")
                os.system(f"mkdir -p {args.plot_dir}/templates/{cutstr}/jshifts")
        else:
            os.system(f"mkdir -p {args.plot_dir}/templates/wshifts")
            os.system(f"mkdir -p {args.plot_dir}/templates/jshifts")

            if args.resonant:
                os.system(f"mkdir -p {args.plot_dir}/templates/hists2d")

    if args.template_dir != "":
        if scan:
            for wps in scan_wps:
                cutstr = "_".join([f"{cut}_{wp}" for cut, wp in zip(scan_cuts, wps)])
                os.system(f"mkdir -p {args.template_dir}/{cutstr}/{args.templates_name}/")
        else:
            os.system(f"mkdir -p {args.template_dir}/{args.templates_name}/")


def _check_load_systematics(systs_file: str, year: str):
    if os.path.exists(systs_file):
        print("Loading systematics")
        with open(systs_file, "r") as f:
            systematics = json.load(f)
    else:
        systematics = {}

    if year not in systematics:
        systematics[year] = {}

    return systematics


def _load_samples(args, samples, sig_samples, cutflow):
    filters = old_filters if args.old_processor else new_filters
    events_dict = None
    if args.signal_data_dir:
        events_dict = utils.load_samples(args.signal_data_dir, sig_samples, args.year, filters)
    if args.data_dir:
        events_dict_data = utils.load_samples(args.data_dir, samples, args.year, filters)
        if events_dict:
            events_dict = utils.merge_dictionaries(events_dict, events_dict_data)
        else:
            events_dict = events_dict_data

    print(events_dict.keys())

    utils.add_to_cutflow(events_dict, "Pre-selection", "weight", cutflow)

    print("")
    # print weighted sample yields
    wkey = "finalWeight" if "finalWeight" in list(events_dict.values())[0] else "weight"
    for sample in events_dict:
        tot_weight = np.sum(events_dict[sample][wkey].values)
        print(f"Pre-selection {sample} yield: {tot_weight:.2f}")

    return events_dict


def apply_weights(
    events_dict: Dict[str, pd.DataFrame],
    year: str,
    cutflow: pd.DataFrame = None,
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
            
    if cutflow is not None:
        utils.add_to_cutflow(events_dict, "TriggerEffs", weight_key, cutflow)

    # calculate QCD scale factor
    if qcd_key in events_dict:
        trig_yields = cutflow["TriggerEffs"]
        non_qcd_bgs_yield = np.sum(
            [
                trig_yields[sample]
                for sample in events_dict
                if sample not in {*nonres_sig_keys, qcd_key, data_key, *res_sig_keys}
            ]
        )
        QCD_SCALE_FACTOR = (trig_yields[data_key] - non_qcd_bgs_yield) / trig_yields[qcd_key]
        events_dict[qcd_key][weight_key] *= QCD_SCALE_FACTOR

        print(f"\n{QCD_SCALE_FACTOR = }")

        if cutflow is not None:
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


def derive_variables(events_dict: Dict[str, pd.DataFrame]):
    """Add HWW vs (QCD + Top) discriminant"""
    for sample, events in events_dict.items():
        if "VVFatJetParTMD_THWWvsT" in events:
            continue

        h4qvst = (events["ak8FatJetParTMD_probHWW3q"] + events["ak8FatJetParTMD_probHWW4q"]) / (
            events["ak8FatJetParTMD_probHWW3q"]
            + events["ak8FatJetParTMD_probHWW4q"]
            + events["ak8FatJetParTMD_probQCD"]
            + events["ak8FatJetParTMD_probT"]
        )

        events_dict[sample] = pd.concat(
            [events, pd.concat([h4qvst], axis=1, keys=["ak8FatJetParTMD_THWWvsT"])], axis=1
        )


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
    bdt_preds = np.load(f"{bdt_preds_dir}/{year}/preds.npy")
    multiclass = len(bdt_preds.shape) > 1

    if jec_jmsr_shifts:
        shift_preds = {
            jshift: np.load(f"{bdt_preds_dir}/{year}/preds_{jshift}.npy")
            for jshift in jec_shifts + jmsr_shifts
        }

    i = 0
    for sample in bdt_sample_order:
        events = events_dict[sample]
        num_events = len(events)

        if not multiclass:
            events["BDTScore"] = bdt_preds[i : i + num_events]
        else:
            events["BDTScore"] = bdt_preds[i : i + num_events, 0]
            events["BDTScoreQCD"] = bdt_preds[i : i + num_events, 1]
            events["BDTScoreTT"] = bdt_preds[i : i + num_events, 2]
            events["BDTScoreVJets"] = 1 - np.sum(bdt_preds[i : i + num_events], axis=1)

        if jec_jmsr_shifts and sample != data_key:
            for jshift in jec_shifts + jmsr_shifts:
                if not multiclass:
                    events["BDTScore_" + jshift] = shift_preds[jshift][i : i + num_events]
                else:
                    events["BDTScore_" + jshift] = shift_preds[jshift][i : i + num_events, 0]
                    events["BDTScoreQCD_" + jshift] = shift_preds[jshift][i : i + num_events, 1]
                    events["BDTScoreTT_" + jshift] = shift_preds[jshift][i : i + num_events, 2]
                    events["BDTScoreVJets_" + jshift] = 1 - np.sum(
                        shift_preds[jshift][i : i + num_events], axis=1
                    )

        i += num_events

    assert i == len(bdt_preds), f"# events {i} != # of BDT preds {len(bdt_preds)}"


def get_lpsf_all_years(
    full_events_dict: Dict[str, pd.DataFrame],
    sig_key: str,
    data_dir: str,
    samples: Dict,
    lp_region: Region,
    bdt_preds_dir: str = None,
    bdt_sample_order: List[str] = None,
):
    print("Getting LP SF for all years combined")
    events_all = []
    sels_all = []

    # (column name, number of subcolumns)
    load_columns = [
        ("weight", 1),
        ("weight_noTrigEffs", 1),
        ("ak8FatJetPt", 2),
        ("ak8FatJetMsd", 2),
        ("ak8FatJetHVV", 2),
        ("ak8FatJetHVVNumProngs", 1),
        ("ak8FatJetParticleNetMD_Txbb", 2),
        ("VVFatJetParTMD_THWWvsT", 1),
        ("lp_sf_lnN", 101),
        ("lp_sf_sys_down", 1),
        ("lp_sf_sys_up", 1),
        ("lp_sf_double_matched_event", 1),
        ("lp_sf_unmatched_quarks", 1),
        ("lp_sf_num_sjpt_gt350", 1),
    ]

    # nonresonant samples use old skimmer #TODO: update!!!
    if bdt_preds_dir is not None:
        load_columns.remove(("weight_noTrigEffs", 1))
        load_columns.remove(("VVFatJetParTMD_THWWvsT", 1))

        load_columns += [
            ("ak8FatJetParTMD_probHWW3q", 2),
            ("ak8FatJetParTMD_probHWW4q", 2),
            ("ak8FatJetParTMD_probQCD", 2),
            ("ak8FatJetParTMD_probT", 2),
        ]

    # reformat into ("column name", "idx") format for reading multiindex columns
    column_labels = []
    for key, num_columns in load_columns:
        for i in range(num_columns):
            column_labels.append(f"('{key}', '{i}')")

    for year in years:
        events_dict = utils.load_samples(
            data_dir, {sig_key: samples[sig_key]}, year, new_filters, column_labels
        )

        # print weighted sample yields
        wkey = "finalWeight" if "finalWeight" in list(events_dict.values())[0] else "weight"
        tot_weight = np.sum(events_dict[sig_key][wkey].values)

        bb_masks = bb_VV_assignment(events_dict)

        if "finalWeight_noTrigEffs" not in events_dict[sig_key]:
            apply_weights(events_dict, year)
            derive_variables(events_dict)

        events_dict[sig_key] = postprocess_lpsfs(events_dict[sig_key])

        if bdt_preds_dir is not None:
            # load bdt preds for sig only
            bdt_preds = np.load(f"{bdt_preds_dir}/{year}/preds.npy")
            multiclass = len(bdt_preds.shape) > 1
            i = 0
            for sample in bdt_sample_order:
                if sample != sig_key:
                    i += len(full_events_dict[sample])
                    continue
                else:
                    events = events_dict[sample]
                    num_events = len(events)
                    if not multiclass:
                        events["BDTScore"] = bdt_preds[i : i + num_events]
                    else:
                        events["BDTScore"] = bdt_preds[i : i + num_events, 0]
                    break

        sel, _ = utils.make_selection(lp_region.cuts, events_dict, bb_masks)

        events_all.append(events_dict[sig_key])
        sels_all.append(sel[sig_key])

    events = pd.concat(events_all, axis=0)
    sel = np.concatenate(sels_all, axis=0)

    return get_lpsf(events, sel)


def lpsfs(
    events_dict: Dict[str, pd.DataFrame],
    bb_masks: Dict[str, pd.DataFrame],
    sig_keys: List[str],
    sig_samples: Dict[str, str],
    cutflow: pd.DataFrame,
    lp_selection_region: Region,
    systematics: Dict,
    all_years: bool = False,
    bdt_sample_order: List[str] = None,
    bdt_preds_dir: str = None,
    template_dir: str = None,
    systs_file: str = None,
    data_dir: str = None,
):
    """
    1) Calculates LP SFs for each signal, if not already in ``systematics``
        - Does it for all years once if args.lp_sf_all_years or just for the given year
    2) Saves them to ``systs_file`` and CSV for posterity
    """
    for sig_key in sig_keys:
        sf_table = OrderedDict()  # format SFs for each sig key in a table
        if sig_key not in systematics or "lp_sf" not in systematics[sig_key]:
            print(f"\nGetting LP SFs for {sig_key}")

            systematics[sig_key] = {}

            # SFs are correlated across all years so needs to be calculated with full dataset
            if all_years:
                lp_sf, unc, uncs = get_lpsf_all_years(
                    events_dict,
                    sig_key,
                    data_dir,
                    sig_samples,
                    lp_selection_region,
                    bdt_preds_dir,
                    bdt_sample_order,
                )
            # Only for testing, can do just for a single year
            else:
                warnings.warn(f"LP SF only calculated from single year's samples", RuntimeWarning)
                # calculate only for current year
                events_dict[sig_key] = postprocess_lpsfs(events_dict[sig_key])
                sel, cf = utils.make_selection(
                    lp_selection_region.cuts,
                    events_dict,
                    bb_masks,
                    prev_cutflow=cutflow,
                )
                lp_sf, unc, uncs = get_lpsf(events_dict[sig_key], sel[sig_key])

            print(f"LP Scale Factor for {sig_key}: {lp_sf:.2f} ± {unc:.2f}")

            systematics[sig_key]["lp_sf"] = lp_sf
            systematics[sig_key]["lp_sf_unc"] = unc / lp_sf
            sf_table[sig_key] = {"SF": f"{lp_sf:.2f} ± {unc:.2f}", **uncs}

        if template_dir is not None:
            if len(sf_table):
                sf_df = pd.DataFrame(index=sig_keys)
                for key in sf_table[sig_key]:
                    sf_df[key] = [sf_table[skey][key] for skey in sig_keys]

                sf_df.to_csv(f"{template_dir}/lpsfs.csv")

    if systs_file is not None:
        with open(systs_file, "w") as f:
            json.dump(systematics, f)

    utils.add_to_cutflow(events_dict, "LP SF", "finalWeight", cutflow)


def control_plots(
    events_dict: Dict[str, pd.DataFrame],
    bb_masks: Dict[str, pd.DataFrame],
    sig_keys: List[str],
    control_plot_vars: Dict[str, Tuple],
    plot_dir: str,
    year: str,
    weight_key: str = "finalWeight",
    hists: Dict = {},
    cutstr: str = "",
    sig_splits: List[List[str]] = None,
    bg_keys: List[str] = bg_keys,
    selection: Dict[str, np.ndarray] = None,
    sig_scale_dict: Dict[str, float] = None,
    combine_pdf: bool = True,
    show: bool = False,
):
    """
    Makes and plots histograms of each variable in ``control_plot_vars``.

    Args:
        control_plot_vars (Dict[str, Tuple]): Dictionary of variables to plot, formatted as
          {var1: ([num bins, min, max], label), var2...}.
        sig_splits: split up signals into different plots (in case there are too many for one)

    """

    from PyPDF2 import PdfMerger

    # sig_scale_dict = utils.getSignalPlotScaleFactor(events_dict, sig_keys)
    # sig_scale_dict = {sig_key: 5e3 for sig_key in sig_keys}
    # sig_scale_dict["HHbbVV"] = 2e5

    if sig_scale_dict is None:
        sig_scale_dict = {sig_key: 1 for sig_key in sig_keys}
        sig_scale_dict["HHbbVV"] = 2e5

    # print(f"{sig_scale_dict = }")

    print(control_plot_vars)
    print(selection)

    for var, (bins, label) in control_plot_vars.items():
        if var not in hists:
            hists[var] = utils.singleVarHist(
                events_dict, var, bins, label, bb_masks, weight_key=weight_key, selection=selection
            )

    with open(f"{plot_dir}/hists.pkl", "wb") as f:
        pickle.dump(hists, f)

    if sig_splits is None:
        sig_splits = [sig_keys]

    for i, plot_sig_keys in enumerate(sig_splits):
        tplot_dir = plot_dir if len(sig_splits) == 1 else f"{plot_dir}/sigs{i}/"
        tsig_scale_dict = {key: sig_scale_dict.get(key, 1) for key in plot_sig_keys}

        merger_control_plots = PdfMerger()

        for var, var_hist in hists.items():
            name = f"{tplot_dir}/{cutstr}{var}.pdf"
            plotting.ratioHistPlot(
                var_hist,
                year,
                plot_sig_keys,
                bg_keys,
                name=name,
                sig_scale_dict=tsig_scale_dict,
                show=show,
            )
            merger_control_plots.append(name)

        if combine_pdf:
            merger_control_plots.write(f"{tplot_dir}/{cutstr}{year}_ControlPlots.pdf")
        merger_control_plots.close()

    return hists


def check_weights(events_dict):
    # Check for 0 weights - would be an issue for weight shifts
    print(
        "\nAny 0 weights:",
        np.any(
            [
                np.any(events["weight_nonorm"] == 0)
                for key, events in events_dict.items()
                if key != data_key
            ]
        ),
    )


def _get_fill_data(
    events: pd.DataFrame, bb_mask: pd.DataFrame, shape_vars: List[ShapeVar], jshift: str = ""
):
    return {
        shape_var.var: utils.get_feat(
            events,
            shape_var.var if jshift == "" else utils.check_get_jec_var(shape_var.var, jshift),
            bb_mask,
        )
        for shape_var in shape_vars
    }


def get_templates(
    events_dict: Dict[str, pd.DataFrame],
    bb_masks: Dict[str, pd.DataFrame],
    year: str,
    sig_keys: List[str],
    selection_regions: Dict[str, Region],
    shape_vars: List[ShapeVar],
    systematics: Dict,
    template_dir: str = "",
    bg_keys: List[str] = bg_keys,
    plot_dir: str = "",
    prev_cutflow: pd.DataFrame = None,
    weight_key: str = "finalWeight",
    sig_splits: List[List[str]] = None,
    weight_shifts: Dict = {},
    jshift: str = "",
    plot_shifts: bool = False,
    pass_ylim: int = None,
    fail_ylim: int = None,
    blind_pass: bool = False,
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
        selection_region (Dict[str, Dict]): Dictionary of ``Region``s including cuts and labels.
        bg_keys (list[str]): background keys to plot.

    Returns:
        Dict[str, Hist]: dictionary of templates, saved as hist.Hist objects.

    """
    do_jshift = jshift != ""
    jlabel = "" if not do_jshift else "_" + jshift
    templates = {}

    for rname, region in selection_regions.items():
        pass_region = rname.startswith("pass")

        if rname == "lpsf":
            continue

        if not do_jshift:
            print(rname)

        # make selection, taking JEC/JMC variations into account
        sel, cf = utils.make_selection(
            region.cuts, events_dict, bb_masks, prev_cutflow=prev_cutflow, jshift=jshift
        )

        if template_dir != "":
            cf.to_csv(f"{template_dir}/{rname}_cutflow{jlabel}.csv")

        # trigger uncertainties
        if not do_jshift:
            systematics[year][rname] = {}
            total, total_err = corrections.get_uncorr_trig_eff_unc(events_dict, bb_masks, year, sel)
            systematics[year][rname]["trig_total"] = total
            systematics[year][rname]["trig_total_err"] = total_err
            print(f"Trigger SF Unc.: {total_err / total:.3f}\n")

        # ParticleNetMD Txbb and ParT LP SFs
        sig_events = {}
        for sig_key in sig_keys:
            sig_events[sig_key] = deepcopy(events_dict[sig_key][sel[sig_key]])
            sig_bb_mask = bb_masks[sig_key][sel[sig_key]]

            if pass_region:
                # scale signal by LP SF
                for wkey in [weight_key, f"{weight_key}_noTrigEffs"]:
                    sig_events[sig_key][wkey] *= systematics[sig_key]["lp_sf"]

                corrections.apply_txbb_sfs(sig_events[sig_key], sig_bb_mask, year, weight_key)

        # if not do_jshift:
        #     print("\nCutflow:\n", cf)

        # set up samples
        hist_samples = list(events_dict.keys())

        if not do_jshift:
            # set up weight-based variations
            for shift in ["down", "up"]:
                if pass_region:
                    for sig_key in sig_keys:
                        hist_samples.append(f"{sig_key}_txbb_{shift}")

                for wshift, wsyst in weight_shifts.items():
                    # if year in wsyst.years:
                    # add to the axis even if not applied to this year to make it easier to sum later
                    for wsample in wsyst.samples:
                        if wsample in events_dict:
                            hist_samples.append(f"{wsample}_{wshift}_{shift}")

        # histograms
        h = Hist(
            hist.axis.StrCategory(hist_samples, name="Sample"),
            *[shape_var.axis for shape_var in shape_vars],
            storage="weight",
        )

        # fill histograms
        for sample in events_dict:
            events = sig_events[sample] if sample in sig_keys else events_dict[sample][sel[sample]]
            if not len(events):
                continue

            bb_mask = bb_masks[sample][sel[sample]]
            fill_data = _get_fill_data(
                events, bb_mask, shape_vars, jshift=jshift if sample != data_key else None
            )
            weight = events[weight_key].values.squeeze()
            h.fill(Sample=sample, **fill_data, weight=weight)

            if not do_jshift:
                # add weight variations
                for wshift, wsyst in weight_shifts.items():
                    if sample in wsyst.samples and year in wsyst.years:
                        # print(wshift)
                        for skey, shift in [("Down", "down"), ("Up", "up")]:
                            if "QCDscale" in wshift:
                                # QCDscale7pt/QCDscale4
                                # https://github.com/LPC-HH/HHLooper/blob/master/python/prepare_card_SR_final.py#L263-L288
                                sweight = (
                                    weight
                                    * (
                                         events[f"weight_QCDscale7pt{skey}"][0] /  events["weight_QCDscale4pt"]
                                    ).values.squeeze()
                                )
                            else:
                                # reweight based on diff between up/down and nominal weights
                                sweight = (
                                    weight
                                    * (
                                        events[f"weight_{wshift}{skey}"][0] / events["weight_nonorm"]
                                    ).values.squeeze()
                                )
                            h.fill(Sample=f"{sample}_{wshift}_{shift}", **fill_data, weight=sweight)

        if pass_region:
            # blind signal mass windows in pass region in data
            for i, shape_var in enumerate(shape_vars):
                if shape_var.blind_window is not None:
                    utils.blindBins(h, shape_var.blind_window, data_key, axis=i)

        if pass_region and not do_jshift:
            for sig_key in sig_keys:
                if not len(sig_events[sig_key]):
                    continue

                # ParticleNetMD Txbb SFs
                fill_data = _get_fill_data(
                    sig_events[sig_key], bb_masks[sig_key][sel[sig_key]], shape_vars
                )
                for shift in ["down", "up"]:
                    h.fill(
                        Sample=f"{sig_key}_txbb_{shift}",
                        **fill_data,
                        weight=sig_events[sig_key][f"{weight_key}_txbb_{shift}"],
                    )

        templates[rname + jlabel] = h

        # plot templates incl variations
        if plot_dir != "" and (not do_jshift or plot_shifts):
            if pass_region:
                sig_scale_dict = {"HHbbVV": 1, **{skey: 1 for skey in res_sig_keys}}

            title = (
                f"{region.label} Region Pre-Fit Shapes"
                if not do_jshift
                else f"{region.label} Region {jshift} Shapes"
            )

            if sig_splits is None:
                sig_splits = [sig_keys]

            for i, shape_var in enumerate(shape_vars):
                for j, plot_sig_keys in enumerate(sig_splits):
                    split_str = "" if len(sig_splits) == 1 else f"sigs{j}_"
                    plot_params = {
                        "hists": h.project(0, i + 1),
                        "sig_keys": plot_sig_keys,
                        "bg_keys": bg_keys,
                        "sig_scale_dict": {key: sig_scale_dict[key] for key in plot_sig_keys}
                        if pass_region
                        else None,
                        "show": show,
                        "year": year,
                        "ylim": pass_ylim if pass_region else fail_ylim,
                        "plot_data": not (rname == "pass" and blind_pass),
                    }

                    plot_name = (
                        f"{plot_dir}/"
                        f"{'jshifts/' if do_jshift else ''}"
                        f"{split_str}{rname}_region_{shape_var.var}"
                    )

                    plotting.ratioHistPlot(
                        **plot_params,
                        title=title,
                        name=f"{plot_name}{jlabel}.pdf",
                    )

                    if not do_jshift and plot_shifts:
                        plot_name = (
                            f"{plot_dir}/wshifts/" f"{split_str}{rname}_region_{shape_var.var}"
                        )

                        for wshift, wsyst in weight_shifts.items():
                            if wsyst.samples == [sig_key]:
                                plotting.ratioHistPlot(
                                    **plot_params,
                                    sig_err=wshift,
                                    title=f"{region.label} Region {wsyst.label} Unc. Shapes",
                                    name=f"{plot_name}_{wshift}.pdf",
                                )
                            else:
                                for skey, shift in [("Down", "down"), ("Up", "up")]:
                                    plotting.ratioHistPlot(
                                        **plot_params,
                                        variation=(wshift, shift, wsyst.samples),
                                        title=f"{region.label} Region {wsyst.label} Unc. {skey} Shapes",
                                        name=f"{plot_name}_{wshift}_{shift}.pdf",
                                    )

                        if pass_region:
                            plotting.ratioHistPlot(
                                **plot_params,
                                sig_err="txbb",
                                title=rf"{region.label} Region $T_{{Xbb}}$ Shapes",
                                name=f"{plot_name}_txbb.pdf",
                            )

    return templates


def save_templates(
    templates: Dict[str, Hist], template_file: str, resonant: bool, shape_vars: List[ShapeVar]
):
    """Creates blinded copies of each region's templates and saves a pickle of the templates"""

    if not resonant:
        from copy import deepcopy

        blind_window = shape_vars[0].blind_window

        for label, template in list(templates.items()):
            blinded_template = deepcopy(template)
            utils.blindBins(blinded_template, blind_window)
            templates[f"{label}Blinded"] = blinded_template

    with open(template_file, "wb") as f:
        pickle.dump(templates, f)

    print("Saved templates to", template_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default=None,
        help="path to skimmed parquet",
        type=str,
    )

    parser.add_argument(
        "--signal-data-dir",
        default="",
        help="path to skimmed signal parquets, if different from other data",
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
        help="path to bdt predictions directory, will look in `data dir`/inferences/ by default",
        default="",
        type=str,
    )

    parser.add_argument(
        "--plot-dir",
        help="If making control or template plots, path to directory to save them in",
        default="",
        type=str,
    )

    parser.add_argument(
        "--template-dir",
        help="If saving templates, path to file to save them in. If scanning, directory to save in.",
        default="",
        type=str,
    )

    parser.add_argument(
        "--templates-name",
        help="If saving templates, optional name for folder (comes under cuts directory if scanning).",
        default="",
        type=str,
    )

    utils.add_bool_arg(parser, "resonant", "for resonant or nonresonant", default=False)
    utils.add_bool_arg(parser, "control-plots", "make control plots", default=False)
    utils.add_bool_arg(parser, "templates", "save m_bb templates using bdt cut", default=False)
    utils.add_bool_arg(
        parser, "overwrite-template", "if template file already exists, overwrite it", default=False
    )
    utils.add_bool_arg(parser, "do-jshifts", "Do JEC/JMC variations", default=True)
    utils.add_bool_arg(parser, "plot-shifts", "Plot systematic variations as well", default=False)
    utils.add_bool_arg(parser, "lp-sf-all-years", "Calculate one LP SF for all run 2", default=True)

    parser.add_argument(
        "--sig-samples",
        help="specify signal samples. By default, will use the samples defined in `hh_vars`.",
        nargs="*",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--bg-keys",
        help="specify background samples",
        nargs="*",
        default=["QCD", "TT", "ST", "V+Jets", "Diboson"],
        type=str,
    )

    utils.add_bool_arg(
        parser, "read-sig-samples", "read signal samples from directory", default=False
    )

    utils.add_bool_arg(parser, "data", "include data", default=True)

    utils.add_bool_arg(parser, "old-processor", "temp arg for old processed samples", default=False)

    parser.add_argument(
        "--nonres-txbb-wp",
        help="Txbb WP for signal region. If multiple arguments, will make templates for each.",
        default=["MP"],
        choices=["LP", "MP", "HP"],
        nargs="*",
        type=str,
    )

    parser.add_argument(
        "--nonres-bdt-wp",
        help="BDT WP for signal region. If multiple arguments, will make templates for each.",
        default=[0.998],
        nargs="*",
        type=float,
    )

    parser.add_argument(
        "--res-txbb-wp",
        help="Txbb WP for signal region. If multiple arguments, will make templates for each.",
        default=["HP"],
        choices=["LP", "MP", "HP"],
        nargs="*",
        type=str,
    )

    parser.add_argument(
        "--res-thww-wp",
        help="Thww WP for signal region. If multiple arguments, will make templates for each.",
        default=[0.96],
        nargs="*",
        type=float,
    )

    args = parser.parse_args()

    if args.template_dir == "":
        print("Need to set --template-dir. Exiting.")
        sys.exit()

    if args.signal_data_dir == "":
        args.signal_data_dir = args.data_dir

    main(args)
