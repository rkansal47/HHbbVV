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

from collections import OrderedDict
import os
import sys
import pickle, json

import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning
import hist
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
    nonres_sig_keys,
    res_sig_keys,
    data_key,
    qcd_key,
    bg_keys,
    samples,
    nonres_samples,
    res_samples,
    txbb_wps,
    jec_shifts,
    jmsr_shifts,
)
from utils import CUT_MAX_VAL

from pprint import pprint
from copy import deepcopy
import warnings


# ignore these because they don't seem to apply
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


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

bdt_cut = 0.99

# {label: {cutvar: [min, max], ...}, ...}
nonres_selection_regions_year = {
    "pass": {
        "bbFatJetPt": [300, CUT_MAX_VAL],
        "VVFatJetPt": [300, CUT_MAX_VAL],
        "BDTScore": [bdt_cut, CUT_MAX_VAL],
        "bbFatJetParticleNetMD_Txbb": ["HP", CUT_MAX_VAL],
    },
    "fail": {
        "bbFatJetPt": [300, CUT_MAX_VAL],
        "VVFatJetPt": [300, CUT_MAX_VAL],
        "bbFatJetParticleNetMD_Txbb": [0.8, "HP"],
    },
    "lpsf": {  # cut for which LP SF is calculated
        "BDTScore": [bdt_cut, CUT_MAX_VAL],
    },
}

res_selection_regions_year = {
    # "unblinded" regions:
    "pass": {
        "bbFatJetPt": [300, CUT_MAX_VAL],
        "VVFatJetPt": [300, CUT_MAX_VAL],
        "bbFatJetParticleNetMass": [110, 145],
        "bbFatJetParticleNetMD_Txbb": ["HP", CUT_MAX_VAL],
        "VVFatJetParTMD_THWWvsT": [0.96, CUT_MAX_VAL],
    },
    "fail": {
        "bbFatJetPt": [300, CUT_MAX_VAL],
        "VVFatJetPt": [300, CUT_MAX_VAL],
        "bbFatJetParticleNetMass": [110, 145],
        "bbFatJetParticleNetMD_Txbb": [0.8, "HP"],
        "VVFatJetParTMD_THWWvsT": [-CUT_MAX_VAL, 0.96],
    },
    # "blinded" validation regions:
    "passBlinded": {
        "bbFatJetPt": [300, CUT_MAX_VAL],
        "VVFatJetPt": [300, CUT_MAX_VAL],
        "bbFatJetParticleNetMass": [[92.5, 110], [145, 162.5]],
        "bbFatJetParticleNetMD_Txbb": ["HP", CUT_MAX_VAL],
        "VVFatJetParTMD_THWWvsT": [0.96, CUT_MAX_VAL],
    },
    "failBlinded": {
        "bbFatJetPt": [300, CUT_MAX_VAL],
        "VVFatJetPt": [300, CUT_MAX_VAL],
        "bbFatJetParticleNetMass": [[92.5, 110], [145, 162.5]],
        "bbFatJetParticleNetMD_Txbb": [0.8, "HP"],
        "VVFatJetParTMD_THWWvsT": [-CUT_MAX_VAL, 0.96],
    },
    "lpsf": {  # cut for which LP SF is calculated
        "VVFatJetParTMD_THWWvsT": [0.96, CUT_MAX_VAL],
    },
}

selection_regions_label = {
    "pass": "Pass",
    "fail": "Fail",
    "BDTOnly": "BDT Cut",
    "lpsf": "LP SF Cut",
    "top": "Top",
    "pass_valid": "Validation Pass",
    "fail_valid": "Validation Fail",
    "passBlinded": "Validation Pass",
    "failBlinded": "Validation Fail",
    "pass_valid_eveto": "Validation Pass + e Veto",
    "fail_valid_eveto": "Validation Fail + e Veto",
    "pass_valid_eveto_hwwvt": "Validation Pass, Cut on THWWvsT + e Veto",
    "fail_valid_eveto_hwwvt": "Validation Fail, Cut on THWWvsT + e Veto",
}

nonres_selection_regions = {}

for year in years:
    sr = deepcopy(nonres_selection_regions_year)

    for region in sr:
        for cuts in sr[region]:
            for i in range(2):
                if sr[region][cuts][i] == "HP":
                    sr[region][cuts][i] = txbb_wps[year]["HP"]

    nonres_selection_regions[year] = sr


res_selection_regions = {}

for year in years:
    sr = deepcopy(res_selection_regions_year)

    for region in sr:
        for cuts in sr[region]:
            for i in range(2):
                if sr[region][cuts][i] == "HP":
                    sr[region][cuts][i] = txbb_wps[year]["HP"]

    res_selection_regions[year] = sr


del year  # creates bugs later

nonres_scan_regions = {}

for bdtcut in np.arange(0.97, 1, 0.002):
    for bbcut in np.arange(0.97, 1, 0.002):
        cutstr = f"bdtcut_{bdtcut}_bbcut_{bbcut}"
        nonres_scan_regions[cutstr] = {
            "passCat1": {
                "BDTScore": [bdtcut, CUT_MAX_VAL],
                "bbFatJetParticleNetMD_Txbb": [bbcut, CUT_MAX_VAL],
            },
            "fail": {
                "bbFatJetParticleNetMD_Txbb": [0.8, bbcut],
            },
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


# TODO: check which of these applies to resonant as well
weight_shifts = {
    "pileup": nonres_sig_keys + res_sig_keys + bg_keys,
    "PDFalphaS": nonres_sig_keys,
    "ISRPartonShower": nonres_sig_keys + ["V+Jets"],
    "FSRPartonShower": nonres_sig_keys + ["V+Jets"],
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

    if args.resonant:
        shape_vars = res_shape_vars
        selection_regions = res_selection_regions
    else:
        shape_vars = nonres_shape_vars
        selection_regions = nonres_selection_regions
        scan_regions = nonres_scan_regions

    sig_keys, sig_samples, bg_keys, samples = _process_samples(args)

    print("Sig keys: ", sig_keys)
    print("Sig samples: ", sig_samples)
    print("BG keys: ", bg_keys)
    print("Samples: ", samples)

    all_samples = sig_samples | samples

    # make plot, template dirs if needed
    _make_dirs(args)

    systs_file = f"{args.template_dir}/systematics.json"

    # save cutflow as pandas table
    cutflow = pd.DataFrame(index=list(all_samples.keys()))
    # load pre-calculated systematics and those for different years if saved already
    systematics = _check_load_systematics(systs_file)

    # utils.remove_empty_parquets(samples_dir, year)
    filters = old_filters if args.old_processor else new_filters
    events_dict = utils.load_samples(args.signal_data_dir, sig_samples, args.year, filters)
    if len(samples):
        events_dict |= utils.load_samples(args.data_dir, samples, args.year, filters)
    utils.add_to_cutflow(events_dict, "Pre-selection", "weight", cutflow)

    print("")
    # print weighted sample yields
    wkey = "finalWeight" if "finalWeight" in list(events_dict.values())[0] else "weight"
    for sample in events_dict:
        tot_weight = np.sum(events_dict[sample][wkey].values)
        print(f"Pre-selection {sample} yield: {tot_weight:.2f}")

    bb_masks = bb_VV_assignment(events_dict)

    if "finalWeight_noTrigEffs" not in events_dict[sample]:
        apply_weights(events_dict, args.year, cutflow)
        derive_variables(events_dict)

    if args.plot_dir != "":
        cutflow.to_csv(f"{args.plot_dir}/cutflows/bdt_cutflow.csv")

    print("\nCutflow\n", cutflow)

    bdt_preds_dir = None
    if not args.resonant:
        print("\nLoading BDT predictions")
        bdt_preds_dir = f"{args.data_dir}/inferences/" if args.bdt_preds == "" else args.bdt_preds
        load_bdt_preds(
            events_dict,
            args.year,
            bdt_preds_dir,
            list(all_samples.keys()),
            jec_jmsr_shifts=True,
        )
        print("Loaded BDT preds\n")

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

    # LP SF
    for sig_key in sig_keys:
        sf_table = OrderedDict()  # format SFs for each sig key in a table
        if sig_key not in systematics or "lp_sf" not in systematics[sig_key]:
            print(f"\nGetting LP SFs for {sig_key}")

            systematics[sig_key] = {}

            if args.lp_sf_all_years:
                lp_sf, unc, uncs = get_lpsf_all_years(
                    events_dict,
                    sig_key,
                    args.signal_data_dir,
                    sig_samples,
                    selection_regions,
                    bdt_preds_dir,
                    list(all_samples.keys()),
                )
            else:
                raise RuntimeWarning(f"LP SF only calculated from {args.year} samples")
                # calculate only for current year
                events_dict[sig_key] = postprocess_lpsfs(events_dict[sig_key])
                sel, cf = utils.make_selection(
                    selection_regions[args.year]["lpsf"],
                    events_dict,
                    bb_masks,
                    prev_cutflow=cutflow,
                )
                lp_sf, unc, uncs = get_lpsf(events_dict[sig_key], sel[sig_key])

            print(f"BDT LP Scale Factor for {sig_key}: {lp_sf:.2f} ± {unc:.2f}")

            systematics[sig_key]["lp_sf"] = lp_sf
            systematics[sig_key]["lp_sf_unc"] = unc / lp_sf

            sf_table[sig_key] = {"SF": f"{lp_sf:.2f} ± {unc:.2f}", **uncs}

        if len(sf_table):
            sf_df = pd.DataFrame(index=sig_keys)
            for key in sf_table[sig_key]:
                sf_df[key] = [sf_table[skey][key] for skey in sig_keys]

            sf_df.to_csv(f"{args.template_dir}/lpsfs.csv")

    with open(systs_file, "w") as f:
        json.dump(systematics, f)

    # scale signal by LP SF
    for wkey in ["finalWeight", "finalWeight_noTrigEffs"]:
        for sig_key in sig_keys:
            events_dict[sig_key][wkey] *= systematics[sig_key]["lp_sf"]

    utils.add_to_cutflow(events_dict, "LP SF", "finalWeight", cutflow)

    # Check for 0 weights - would be an issue for weight shifts
    check_weights(events_dict)

    if args.templates:
        print("\nMaking templates")

        templates = {}

        for jshift in [""] + jec_shifts + jmsr_shifts:
            print(jshift)

            if jshift != "":
                if args.plot_shifts:
                    plot_dir = f"{args.plot_dir}/jshifts/"
                else:
                    plot_dir = ""
            else:
                plot_dir = args.plot_dir

            ttemps, tsyst = get_templates(
                events_dict,
                bb_masks,
                args.year,
                sig_keys,
                selection_regions[args.year],
                shape_vars,
                # bg_keys=["QCD", "TT", "V+Jets"],
                plot_dir=plot_dir,
                prev_cutflow=cutflow,
                # sig_splits=sig_splits,
                weight_shifts=weight_shifts,
                jshift=jshift,
                blind_pass=True if args.resonant else False,
                show=False,
                plot_shifts=args.plot_shifts,
            )

            templates = {**templates, **ttemps}
            if jshift == "":
                systematics[args.year] = tsyst

        print("\nSaving templates")
        save_templates(
            templates, f"{args.template_dir}/{args.year}_templates.pkl", args.resonant, shape_vars
        )

        with open(systs_file, "w") as f:
            json.dump(systematics, f)

    if args.scan:
        os.system(f"mkdir -p {args.template_file}")

        templates = {}

        for cutstr, region in scan_regions.items():
            print(cutstr)
            templates[cutstr] = get_templates(
                events_dict,
                bb_masks,
                args.year,
                region,
                shape_vars,
                plot_dir=args.plot_dir,
                prev_cutflow=cutflow,
                cutstr=cutstr,
            )
            save_templates(
                templates[cutstr], f"{args.template_dir}/{cutstr}.pkl", args.resonant, shape_vars
            )


def _process_samples(args):
    if args.resonant:
        sig_keys = res_sig_keys
        sig_samples = res_samples
    else:
        sig_keys = nonres_sig_keys
        sig_samples = nonres_samples

    if args.read_sig_samples:
        read_year = args.year if args.year != "all" else "2017"
        read_samples = os.listdir(f"{args.signal_data_dir}/{args.year}")
        sig_samples = OrderedDict()
        for sample in read_samples:
            if sample.startswith("NMSSM_XToYHTo2W2BTo4Q2B_MX-"):
                mY = int(sample.split("-")[-1])
                mX = int(sample.split("NMSSM_XToYHTo2W2BTo4Q2B_MX-")[1].split("_")[0])

                sig_samples[f"X[{mX}]->H(bb)Y[{mY}](VV)"] = sample

        sig_keys = list(sig_samples.keys())

    print(sig_samples, sig_keys)

    if len(args.sig_samples):
        for sig_key, sample in list(sig_samples.items()):
            if sample not in args.sig_samples:
                del sig_samples[sig_key]
                sig_keys.remove(sig_key)

    if len(args.bg_keys):
        for bg_key, sample in list(samples.items()):
            if bg_key not in args.bg_keys and bg_key != data_key:
                del samples[bg_key]
                bg_keys.remove(bg_key)

    if not args.data:
        del samples[data_key]

    return sig_keys, sig_samples, bg_keys, samples


def _make_dirs(args):
    if args.plot_dir != "":
        args.plot_dir = f"{args.plot_dir}/{args.year}/"
        os.system(f"mkdir -p {args.plot_dir}/cutflows/")

        if args.control_plots:
            os.system(f"mkdir -p {args.plot_dir}/control_plots/")

        os.system(f"mkdir -p {args.plot_dir}/templates/")

        if args.plot_shifts:
            os.system(f"mkdir -p {args.plot_dir}/templates/wshifts")
            os.system(f"mkdir -p {args.plot_dir}/templates/jshifts")

        if args.resonant:
            os.system(f"mkdir -p {args.plot_dir}/templates/hists2d")

    if args.template_dir != "":
        os.system(f"mkdir -p {args.template_dir}")


def _check_load_systematics(systs_file: str):
    if os.path.exists(systs_file):
        print("Loading systematics")
        with open(systs_file, "r") as f:
            systematics = json.load(f)
    else:
        systematics = {}

    return systematics


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
        h4qvst = (events["ak8FatJetParTMD_probHWW3q"] + events["ak8FatJetParTMD_probHWW4q"]) / (
            events["ak8FatJetParTMD_probHWW3q"]
            + events["ak8FatJetParTMD_probHWW4q"]
            + events["ak8FatJetParTMD_probQCD"]
            + events["ak8FatJetParTMD_probT"]
        )

        events_dict[sample] = pd.concat(
            [events, pd.concat([h4qvst], axis=1, keys=["ak8FatJetParTMD_THWWvsT"])], axis=1
        )


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

    # for jet in ["bb", "VV"]:
    for jet in ["VV"]:
        # temp dict
        td = {}

        # defaults of 1 for jets which aren't matched to anything - i.e. no SF
        for key in ["lp_sf_nom", "lp_sf_sys_down", "lp_sf_sys_up"]:
            td[key] = np.ones(len(events))

        td["lp_sf_toys"] = np.ones((len(events), num_lp_sf_toys))

        # defaults of 0 - i.e. don't contribute to unc.
        for key in ["lp_sf_double_matched_event", "lp_sf_unmatched_quarks", "lp_sf_num_sjpt_gt350"]:
            td[key] = np.zeros(len(events))

        # lp sfs saved for both jets
        if events["lp_sf_sys_up"].shape[1] == 2:
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
                    "lp_sf_num_sjpt_gt350",
                ]:
                    td[key][jet_match[j]] = events[key][jet_match[j]][j]

        # lp sfs saved only for hvv jet
        elif events["lp_sf_sys_up"].shape[1] == 1:
            # ignore rare case (~0.002%) where two jets are matched to same gen Higgs
            jet_match = np.sum(events[f"ak8FatJetH{jet}"], axis=1) == 1

            # fill values from matched jets
            td["lp_sf_nom"][jet_match] = events["lp_sf_lnN"][jet_match][0]
            td["lp_sf_toys"][jet_match] = events["lp_sf_lnN"][jet_match].loc[:, 1:]

            for key in [
                "lp_sf_sys_down",
                "lp_sf_sys_up",
                "lp_sf_double_matched_event",
                "lp_sf_unmatched_quarks",
                "lp_sf_num_sjpt_gt350",
            ]:
                td[key][jet_match] = events[key][jet_match].squeeze()

        else:
            raise ValueError("LP SF shapes are invalid")

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

    weight = events[weight_key].values
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


def get_lpsf_all_years(
    full_events_dict: Dict[str, pd.DataFrame],
    sig_key: str,
    data_dir: str,
    samples: Dict,
    selection_regions: Dict,
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
            i = 0
            for sample in bdt_sample_order:
                if sample != sig_key:
                    i += len(full_events_dict[sample])
                    continue
                else:
                    events = events_dict[sample]
                    num_events = len(events)
                    events["BDTScore"] = bdt_preds[i : i + num_events]
                    break

        sel, _ = utils.make_selection(selection_regions[year]["lpsf"], events_dict, bb_masks)

        events_all.append(events_dict[sig_key])
        sels_all.append(sel[sig_key])

    events = pd.concat(events_all, axis=0)
    sel = np.concatenate(sels_all, axis=0)

    return get_lpsf(events, sel)


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

    if jec_jmsr_shifts:
        shift_preds = {
            jshift: np.load(f"{bdt_preds_dir}/{year}/preds_{jshift}.npy")
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
    sig_keys: List[str],
    control_plot_vars: Dict[str, Tuple],
    plot_dir: str,
    year: str,
    weight_key: str = "finalWeight",
    hists: Dict = {},
    cutstr: str = "",
    sig_splits: List[List[str]] = None,
    show: bool = False,
):
    """
    Makes and plots histograms of each variable in ``control_plot_vars``.

    Args:
        control_plot_vars (Dict[str, Tuple]): Dictionary of variables to plot, formatted as
          {var1: ([num bins, min, max], label), var2...}.
        sig_splits: split up signals into different plots (in case there are too many for one)

    """

    print(control_plot_vars)

    from PyPDF2 import PdfMerger

    # sig_scale_dict = utils.getSignalPlotScaleFactor(events_dict, sig_keys)
    sig_scale_dict = {sig_key: 5e3 for sig_key in sig_keys}
    sig_scale_dict["HHbbVV"] = 2e5
    # print(f"{sig_scale_dict = }")

    for var, (bins, label) in control_plot_vars.items():
        # print(var)
        if var not in hists:
            hists[var] = utils.singleVarHist(
                events_dict, var, bins, label, bb_masks, weight_key=weight_key
            )

    with open(f"{plot_dir}/hists.pkl", "wb") as f:
        pickle.dump(hists, f)

    if sig_splits is None:
        sig_splits = [sig_keys]

    for i, plot_sig_keys in enumerate(sig_splits):
        tplot_dir = plot_dir if len(sig_splits) == 1 else f"{plot_dir}/sigs{i}/"
        tsig_scale_dict = {key: sig_scale_dict[key] for key in plot_sig_keys}

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

        merger_control_plots.write(f"{tplot_dir}/{year}_{cutstr}ControlPlots.pdf")
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
    selection_regions: Dict[str, Dict],
    shape_vars: List[ShapeVar],
    bg_keys: List[str] = bg_keys,
    plot_dir: str = "",
    prev_cutflow: pd.DataFrame = None,
    weight_key: str = "finalWeight",
    cutstr: str = "",
    sig_splits: List[List[str]] = None,
    weight_shifts: Dict = {},
    jshift: str = "",
    selection_regions_label: Dict = selection_regions_label,
    plot_shifts: bool = True,
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
        selection_region (Dict[str, Dict]): Dictionary of cuts for each region
          formatted as {region1: {cutvar1: [min, max], ...}, ...}.
        bg_keys (list[str]): background keys to plot.
        cutstr (str): optional string to add to plot file names.

    Returns:
        Dict[str, Hist]: dictionary of templates, saved as hist.Hist objects.

    """
    do_jshift = jshift != ""
    jlabel = "" if not do_jshift else "_" + jshift
    templates, systematics = {}, {}

    for label, region in selection_regions.items():
        pass_region = label.startswith("pass")

        if label == "lpsf":
            continue

        if not do_jshift:
            print(label)

        # make selection, taking JEC/JMC variations into account
        sel, cf = utils.make_selection(
            region, events_dict, bb_masks, prev_cutflow=prev_cutflow, jshift=jshift
        )
        # if not do_jshift:
        #     print("\nCutflow:\n", cf)

        if plot_dir != "":
            cf.to_csv(
                f"{plot_dir}{'/cutflows/' if jshift == '' else '/'}{label}_cutflow{jlabel}.csv"
            )

        if not do_jshift:
            systematics[label] = {}

            total, total_err = corrections.get_uncorr_trig_eff_unc(events_dict, bb_masks, year, sel)

            systematics[label]["trig_total"] = total
            systematics[label]["trig_total_err"] = total_err

            print(f"Trigger SF Unc.: {total_err / total:.3f}\n")

        # ParticleNetMD Txbb SFs
        sig_events = {}
        for sig_key in sig_keys:
            sig_events[sig_key] = deepcopy(events_dict[sig_key][sel[sig_key]])
            sig_bb_mask = bb_masks[sig_key][sel[sig_key]]
            if pass_region:
                corrections.apply_txbb_sfs(sig_events[sig_key], sig_bb_mask, year, weight_key)

        # set up samples
        hist_samples = list(events_dict.keys())

        if not do_jshift:
            # set up weight-based variations
            for shift in ["down", "up"]:
                if pass_region:
                    for sig_key in sig_keys:
                        hist_samples.append(f"{sig_key}_txbb_{shift}")

                for wshift, wsamples in weight_shifts.items():
                    for wsample in wsamples:
                        if wsample in events_dict:
                            hist_samples.append(f"{wsample}_{wshift}_{shift}")

        h = Hist(
            hist.axis.StrCategory(hist_samples, name="Sample"),
            *[shape_var.axis for shape_var in shape_vars],
            storage="weight",
        )

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
            # blind signal mass windows in pass region in data
            for i, shape_var in enumerate(shape_vars):
                if shape_var.blind_window is not None:
                    utils.blindBins(h, shape_var.blind_window, data_key, axis=i)

        if pass_region and not do_jshift:
            for sig_key in sig_keys:
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

        templates[label + jlabel] = h

        # plot templates incl variations
        if plot_dir != "":
            if pass_region:
                sig_scale_dict = {"HHbbVV": 1, **{skey: 1 for skey in res_sig_keys}}

            if not do_jshift:
                title = f"{selection_regions_label[label]} Region Pre-Fit Shapes"
            else:
                title = f"{selection_regions_label[label]} Region {jshift} Shapes"

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
                        "plot_data": not (label == "pass" and blind_pass),
                    }

                    plot_name = f"{plot_dir}/templates/{'jshifts/' if do_jshift else ''}{split_str}{cutstr}{label}_region_{shape_var.var}"

                    plotting.ratioHistPlot(
                        **plot_params,
                        title=title,
                        name=f"{plot_name}{jlabel}.pdf",
                    )

                    if not do_jshift and plot_shifts:
                        plot_name = f"{plot_dir}/templates/wshifts/{split_str}{cutstr}{label}_region_{shape_var.var}"

                        for wshift, wsamples in weight_shifts.items():
                            wlabel = weight_labels[wshift]

                            if wsamples == [sig_key]:
                                plotting.ratioHistPlot(
                                    **plot_params,
                                    sig_err=wshift,
                                    title=f"{selection_regions_label[label]} Region {wlabel} Unc. Shapes",
                                    name=f"{plot_name}_{wshift}.pdf",
                                )
                            else:
                                for skey, shift in [("Down", "down"), ("Up", "up")]:
                                    plotting.ratioHistPlot(
                                        **plot_params,
                                        variation=(wshift, shift, wsamples),
                                        title=f"{selection_regions_label[label]} Region {wlabel} Unc. {skey} Shapes",
                                        name=f"{plot_name}_{wshift}_{shift}.pdf",
                                    )

                        if pass_region:
                            plotting.ratioHistPlot(
                                **plot_params,
                                sig_err="txbb",
                                title=rf"{selection_regions_label[label]} Region $T_{{Xbb}}$ Shapes",
                                name=f"{plot_name}_txbb.pdf",
                            )

    return templates, systematics


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
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="../../../../data/skimmer/Feb24/",
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

    utils.add_bool_arg(parser, "resonant", "for resonant or nonresonant", default=False)
    utils.add_bool_arg(parser, "control-plots", "make control plots", default=False)
    utils.add_bool_arg(parser, "templates", "save m_bb templates using bdt cut", default=False)
    utils.add_bool_arg(
        parser, "overwrite-template", "if template file already exists, overwrite it", default=False
    )
    utils.add_bool_arg(parser, "scan", "Scan BDT + Txbb cuts and save templates", default=False)
    utils.add_bool_arg(parser, "plot-shifts", "Plot systematic variations as well", default=False)

    utils.add_bool_arg(parser, "lp-sf-all-years", "Calculate one LP SF for all run 2", default=True)

    parser.add_argument(
        "--sig-samples",
        help="specify signal samples",
        nargs="*",
        default=[],
        type=str,
    )

    parser.add_argument(
        "--bg-keys",
        help="specify background samples",
        nargs="*",
        default=[],
        type=str,
    )

    utils.add_bool_arg(
        parser, "read-sig-samples", "read signal samples from directory", default=False
    )

    utils.add_bool_arg(parser, "data", "include data", default=True)

    utils.add_bool_arg(parser, "old-processor", "temp arg for old processed samples", default=False)

    args = parser.parse_args()

    if args.template_dir == "":
        print("Need to set --template-dir. Exiting.")
        sys.exit()

    if args.signal_data_dir == "":
        args.signal_data_dir = args.data_dir

    main(args)
