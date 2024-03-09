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

from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import sys
import warnings
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import corrections

# from pandas.errors import SettingWithCopyWarning
import hist
import numpy as np
import pandas as pd
import plotting
import utils
from corrections import get_lpsf, postprocess_lpsfs
from hist import Hist
from utils import CUT_MAX_VAL, ShapeVar

from HHbbVV import hh_vars
from HHbbVV.hh_vars import (
    bg_keys,
    data_key,
    jec_shifts,
    jmsr_shifts,
    nonres_samples,
    nonres_sig_keys,
    norm_preserving_weights,
    qcd_key,
    res_samples,
    res_sig_keys,
    samples,
    txbb_wps,
    years,
)
from HHbbVV.run_utils import add_bool_arg

# ignore these because they don't seem to apply
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


@dataclass
class Syst:
    samples: list[str] = None
    years: list[str] = field(default_factory=lambda: years)
    label: str = None


@dataclass
class Region:
    cuts: dict = None
    label: str = None


# Both Jet's Regressed Mass above 50
load_filters = [
    [
        ("('ak8FatJetParticleNetMass', '0')", ">=", 50),
        ("('ak8FatJetParticleNetMass', '1')", ">=", 50),
    ],
]

# {var: (bins, label)}
control_plot_vars = [
    ShapeVar(var="MET_pt", label=r"$p^{miss}_T$ (GeV)", bins=[20, 0, 300]),
    # ShapeVar(var="DijetEta", label=r"$\eta^{jj}$", bins=[20, -8, 8]),
    ShapeVar(var="DijetPt", label=r"$p_T^{jj}$ (GeV)", bins=[20, 0, 750]),
    ShapeVar(var="DijetMass", label=r"$m^{jj}$ (GeV)", bins=[20, 600, 4000]),
    ShapeVar(var="bbFatJetEta", label=r"$\eta^{bb}$", bins=[20, -2.4, 2.4]),
    ShapeVar(
        var="bbFatJetPt", label=r"$p^{bb}_T$ (GeV)", bins=[20, 300, 2300], significance_dir="right"
    ),
    ShapeVar(
        var="bbFatJetParticleNetMass",
        label=r"$m^{bb}_{reg}$ (GeV)",
        bins=[20, 50, 250],
        significance_dir="bin",
    ),
    ShapeVar(var="bbFatJetMsd", label=r"$m^{bb}_{msd}$ (GeV)", bins=[20, 50, 250]),
    ShapeVar(var="bbFatJetParticleNetMD_Txbb", label=r"$T^{bb}_{Xbb}$", bins=[20, 0.8, 1]),
    ShapeVar(var="VVFatJetEta", label=r"$\eta^{VV}$", bins=[20, -2.4, 2.4]),
    ShapeVar(var="VVFatJetPt", label=r"$p^{VV}_T$ (GeV)", bins=[20, 300, 2300]),
    ShapeVar(var="VVFatJetParticleNetMass", label=r"$m^{VV}_{reg}$ (GeV)", bins=[20, 50, 250]),
    ShapeVar(var="VVFatJetMsd", label=r"$m^{VV}_{msd}$ (GeV)", bins=[20, 50, 250]),
    # ShapeVar(
    #     var="VVFatJetParticleNet_Th4q",
    #     label=r"Prob($H \to 4q$) vs Prob(QCD) (Non-MD)",
    #     bins=[20, 0, 1],
    # ),
    # ShapeVar(
    #     var="VVFatJetParTMD_THWW4q",
    #     label=r"Prob($H \to VV \to 4q$) vs Prob(QCD) (Mass-Decorrelated)",
    #     bins=[20, 0, 1],
    # ),
    # ShapeVar(var="VVFatJetParTMD_probT", label=r"Prob(Top) (Mass-Decorrelated)", bins=[20, 0, 1]),
    ShapeVar(var="VVFatJetParTMD_THWWvsT", label=r"$T^{VV}_{HWW}$", bins=[20, 0, 1]),
    # ShapeVar(var="bbFatJetPtOverDijetPt", label=r"$p^{bb}_T / p_T^{jj}$", bins=[20, 0, 40]),
    ShapeVar(var="VVFatJetPtOverDijetPt", label=r"$p^{VV}_T / p_T^{jj}$", bins=[20, 0, 40]),
    ShapeVar(var="VVFatJetPtOverbbFatJetPt", label=r"$p^{VV}_T / p^{bb}_T$", bins=[20, 0.4, 2.0]),
    ShapeVar(var="nGoodMuonsHbb", label=r"# of Muons", bins=[3, 0, 3]),
    ShapeVar(var="nGoodMuonsHH", label=r"# of Muons", bins=[3, 0, 3]),
    ShapeVar(var="nGoodElectronsHbb", label=r"# of Electrons", bins=[3, 0, 3]),
    ShapeVar(var="nGoodElectronsHH", label=r"# of Electrons", bins=[3, 0, 3]),
    # removed if not ggF nonresonant - needs to be the last variable!
    ShapeVar(var="BDTScore", label=r"BDT Score", bins=[50, 0, 1]),
]


# for msd vs mreg comparison plots only
mass_plot_vars = [
    ShapeVar(var="bbFatJetParticleNetMass", label=r"$m^{bb}_{reg}$ (GeV)", bins=[30, 0, 300]),
    ShapeVar(var="bbFatJetMsd", label=r"$m^{bb}_{msd}$ (GeV)", bins=[30, 0, 300]),
    ShapeVar(var="VVFatJetParticleNetMass", label=r"$m^{VV}_{reg}$ (GeV)", bins=[30, 0, 300]),
    ShapeVar(var="VVFatJetMsd", label=r"$m^{VV}_{msd}$ (GeV)", bins=[30, 0, 300]),
]


def get_nonres_selection_regions(
    year: str,
    txbb_wp: str = "MP",
    bdt_wp: float = 0.998,
    lepton_veto="None",
):
    pt_cuts = [300, CUT_MAX_VAL]
    txbb_cut = txbb_wps[year][txbb_wp]

    if lepton_veto == "None":
        lepton_cuts = {}
    elif lepton_veto == "Hbb":
        lepton_cuts = {
            "nGoodElectronsHbb": [0, 0.9],
            "nGoodMuonsHbb": [0, 0.9],
        }
    elif lepton_veto == "HH":
        lepton_cuts = {
            "nGoodElectronsHH": [0, 0.9],
            "nGoodMuonsHH": [0, 0.9],
        }
    else:
        raise ValueError(f"Invalid lepton veto: {lepton_veto}")

    return {
        # {label: {cutvar: [min, max], ...}, ...}
        "pass": Region(
            cuts={
                "bbFatJetPt": pt_cuts,
                "VVFatJetPt": pt_cuts,
                "BDTScore": [bdt_wp, CUT_MAX_VAL],
                "bbFatJetParticleNetMD_Txbb": [txbb_cut, CUT_MAX_VAL],
                **lepton_cuts,
            },
            label="Pass",
        ),
        "fail": Region(
            cuts={
                "bbFatJetPt": pt_cuts,
                "VVFatJetPt": pt_cuts,
                "bbFatJetParticleNetMD_Txbb": [0.8, txbb_cut],
                **lepton_cuts,
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


def get_nonres_vbf_selection_regions(
    year: str,
    txbb_wp: str = "HP",
    thww_wp: float = 0.6,
):
    # edit
    pt_cuts = [300, CUT_MAX_VAL]
    txbb_cut = txbb_wps[year][txbb_wp]

    return {
        # {label: {cutvar: [min, max], ...}, ...}
        "pass": Region(
            cuts={
                "bbFatJetPt": pt_cuts,
                "VVFatJetPt": pt_cuts,
                "bbFatJetParticleNetMD_Txbb": [txbb_cut, CUT_MAX_VAL],
                "VVFatJetParTMD_THWWvsT": [thww_wp, CUT_MAX_VAL],
                "vbf_Mass_jj": [500, 10000],
                "vbf_dEta_jj": [4, 10000],
                "ak8FatJetEta0": [-2.4, 2.4],
                "ak8FatJetEta1": [-2.4, 2.4],
                "DijetdEta": [0, 2.0],
                "DijetdPhi": [2.6, 10000],
                "bbFatJetParticleNetMass": [50, 250],
                "nGoodElectronsHbb": [0, 0.9],
                "nGoodMuonsHbb": [0, 0.9],
            },
            label="Pass",
        ),
        "fail": Region(
            cuts={
                "bbFatJetPt": pt_cuts,
                "VVFatJetPt": pt_cuts,
                "bbFatJetParticleNetMD_Txbb": [0.8, txbb_cut],
                "VVFatJetParTMD_THWWvsT": [thww_wp, CUT_MAX_VAL],
                "vbf_Mass_jj": [500, 10000],
                "vbf_dEta_jj": [4, 10000],
                "ak8FatJetEta0": [-2.4, 2.4],
                "ak8FatJetEta1": [-2.4, 2.4],
                "DijetdEta": [0, 2.0],
                "DijetdPhi": [2.6, 10000],
                "bbFatJetParticleNetMass": [50, 250],
                "nGoodElectronsHbb": [0, 0.9],
                "nGoodMuonsHbb": [0, 0.9],
            },
            label="Fail",
        ),
        "lpsf": Region(
            cuts={  # cut for which LP SF is calculated
                "bbFatJetPt": pt_cuts,
                "VVFatJetPt": pt_cuts,
                "VVFatJetParTMD_THWWvsT": [thww_wp, CUT_MAX_VAL],
            },
            label="LP SF Cut",
        ),
    }


def get_res_selection_regions(
    year: str,
    mass_window: list[float] = None,
    txbb_wp: str = "HP",
    thww_wp: float = 0.6,
    leadingpt_wp: float = 400,
    subleadingpt_wp: float = 300,
):
    if mass_window is None:
        mass_window = [110, 145]
    mwsize = mass_window[1] - mass_window[0]
    mw_sidebands = [
        [mass_window[0] - mwsize / 2, mass_window[0]],
        [mass_window[1], mass_window[1] + mwsize / 2],
    ]
    txbb_cut = txbb_wps[year][txbb_wp]

    # first define without pT cuts
    regions = {
        # "unblinded" regions:
        "pass": Region(
            cuts={
                "bbFatJetParticleNetMass": mass_window,
                "bbFatJetParticleNetMD_Txbb": [txbb_cut, CUT_MAX_VAL],
                "VVFatJetParTMD_THWWvsT": [thww_wp, CUT_MAX_VAL],
            },
            label="Pass",
        ),
        "fail": Region(
            cuts={
                "bbFatJetParticleNetMass": mass_window,
                "bbFatJetParticleNetMD_Txbb": [0.8, txbb_cut],
                "VVFatJetParTMD_THWWvsT": [-CUT_MAX_VAL, thww_wp],
            },
            label="Fail",
        ),
        # "blinded" validation regions:
        "passBlinded": Region(
            cuts={
                "bbFatJetParticleNetMass": mw_sidebands,
                "bbFatJetParticleNetMD_Txbb": [txbb_cut, CUT_MAX_VAL],
                "VVFatJetParTMD_THWWvsT": [thww_wp, CUT_MAX_VAL],
            },
            label="Validation Pass",
        ),
        "failBlinded": Region(
            cuts={
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

    # add pT cuts
    leading_pt_cut = [leadingpt_wp, CUT_MAX_VAL]
    subleading_pt_cut = [subleadingpt_wp, CUT_MAX_VAL]

    for _key, region in regions.items():
        cuts = {
            "bbFatJetPt": subleading_pt_cut,
            "VVFatJetPt": subleading_pt_cut,
            # '+' means OR
            "bbFatJetPt+VVFatJetPt": leading_pt_cut,
            **region.cuts,
        }
        region.cuts = cuts

    return regions


# fitting on bb regressed mass for nonresonant
nonres_shape_vars = [
    ShapeVar(
        "bbFatJetParticleNetMass",
        r"$m^{bb}_\mathrm{Reg}$ (GeV)",
        [20, 50, 250],
        reg=True,
        blind_window=[100, 150],
    )
]


# templates saved in bb regressed mass for nonresonant VBF
nonres_vbf_shape_vars = [
    ShapeVar(
        "bbFatJetParticleNetMass",
        r"$m^{bb}_\mathrm{Reg}$ (GeV)",
        [20, 50, 250],
        reg=True,
        blind_window=[100, 150],
    )
]


# fitting on VV regressed mass + dijet mass for resonant
res_shape_vars = [
    ShapeVar(
        "VVFatJetParticleNetMass",
        r"$m^{VV}_\mathrm{Reg}$ (GeV)",
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
nonres_vbf_scan_cuts = ["txbb", "thww"]  # TODO: add more cuts being scanned over
res_scan_cuts = ["txbb", "thww", "leadingpt", "subleadingpt"]

nonres_sig_keys_ggf = [
    "HHbbVV",
    "ggHH_kl_2p45_kt_1_HHbbVV",
    "ggHH_kl_5_kt_1_HHbbVV",
    "ggHH_kl_0_kt_1_HHbbVV",
]

fit_bgs = ["TT", "ST", "W+Jets", "Z+Jets"]  # only the BG MC samples that are used in the fits
fit_mcs = nonres_sig_keys + res_sig_keys + fit_bgs

weight_shifts = {
    "pileup": Syst(samples=fit_mcs, label="Pileup"),
    "pileupID": Syst(samples=fit_mcs, label="Pileup ID"),
    "ISRPartonShower": Syst(samples=fit_mcs, label="ISR Parton Shower"),
    "FSRPartonShower": Syst(samples=fit_mcs, label="FSR Parton Shower"),
    "L1EcalPrefiring": Syst(
        samples=fit_mcs,
        years=["2016APV", "2016", "2017"],
        label="L1 ECal Prefiring",
    ),
    "electron_id": Syst(samples=fit_mcs, label="Electron ID"),
    "muon_id": Syst(samples=fit_mcs, label="Muon ID"),
    # TODO: check which of these applies to resonant as well
    "scale": Syst(samples=nonres_sig_keys + ["TT"], label="QCDScaleAcc"),
    "pdf": Syst(samples=nonres_sig_keys, label="PDFAcc"),
    # "top_pt": ["TT"],
}

plot_sig_keys_nonres = [
    "HHbbVV",
    "VBFHHbbVV",
    "qqHH_CV_1_C2V_0_kl_1_HHbbVV",
    "qqHH_CV_1_C2V_2_kl_1_HHbbVV",
]


def main(args):
    global control_plot_vars  # noqa: PLW0603

    shape_vars, scan, scan_cuts, scan_wps = _init(args)
    sig_keys, sig_samples, bg_keys, bg_samples = _process_samples(args)
    all_samples = sig_keys + bg_keys
    _make_dirs(args, scan, scan_cuts, scan_wps)  # make plot, template dirs if needed
    cutflow = pd.DataFrame(index=all_samples)  # save cutflow as pandas table

    # only need to worry about variations if making templates
    events_dict = _load_samples(args, bg_samples, sig_samples, cutflow, variations=args.templates)
    bb_masks = bb_VV_assignment(events_dict)

    # QCD xsec normalization for plots
    qcd_sf(events_dict, cutflow)

    # THWW score vs Top (if not already from processor)
    derive_variables(
        events_dict,
        bb_masks,
        nonres_vars=args.vbf or args.control_plots,
        vbf_vars=args.vbf,
        do_jshifts=False,  # only need shifts for BDT pre-processing
    )

    # args has attr if --control-plots arg was set
    if hasattr(args, "control_plots_dir"):
        cutflow.to_csv(args.control_plots_dir / "preselection_cutflow.csv")

    print("\nCutflow", cutflow)

    # Load BDT Scores
    if not args.resonant and not args.vbf:
        print("\nLoading BDT predictions")
        load_bdt_preds(
            events_dict,
            args.year,
            args.bdt_preds_dir,
            jec_jmsr_shifts=True,
        )
        print("Loaded BDT preds\n")
    else:
        if control_plot_vars[-1].var == "BDTScore":
            control_plot_vars = control_plot_vars[:-1]

    # Control plots
    if args.control_plots:
        print("\nMaking control plots\n")
        plot_vars = mass_plot_vars if args.mass_plots else control_plot_vars
        if len(args.control_plot_vars):
            for var in plot_vars.copy():
                if var.var not in args.control_plot_vars:
                    plot_vars.remove(var)

        print("Plotting: ", [var.var for var in plot_vars])
        if args.resonant:
            p_sig_keys = sig_keys
            sig_scale_dict = {"HHbbVV": 1e5, "VBFHHbbVV": 2e6} | {key: 2e4 for key in res_sig_keys}
        else:
            p_sig_keys = plot_sig_keys_nonres
            sig_scale_dict = {
                "HHbbVV": 1e5,
                "VBFHHbbVV": 2e5,
                "qqHH_CV_1_C2V_0_kl_1_HHbbVV": 2e3,
                "qqHH_CV_1_C2V_2_kl_1_HHbbVV": 2e3,
            }

        control_plots(
            events_dict,
            bb_masks,
            p_sig_keys,
            plot_vars,
            args.control_plots_dir,
            args.year,
            bg_keys=args.bg_keys,
            sig_scale_dict=sig_scale_dict,
            # sig_splits=sig_splits,
            HEM2d=args.HEM2d,
            same_ylim=args.mass_plots,
            show=False,
        )

    if args.bdt_plots and not args.resonant and not args.vbf:
        print("\nMaking BDT sculpting plots\n")

        plot_bdt_sculpting(
            events_dict, bb_masks, args.bdt_sculpting_plots_dir, args.year, show=False
        )

    if args.templates:
        if args.resonant:
            sig_scale_dict = None
        else:
            sig_scale_dict = {
                "HHbbVV": 50,
                "VBFHHbbVV": 1000,
                "qqHH_CV_1_C2V_0_kl_1_HHbbVV": 10,
                "qqHH_CV_1_C2V_2_kl_1_HHbbVV": 10,
            }

        for wps in scan_wps:  # if not scanning, this will just be a single WP
            cutstr = "_".join([f"{cut}_{wp}" for cut, wp in zip(scan_cuts, wps)]) if scan else ""
            template_dir = args.template_dir / cutstr / args.templates_name

            cutargs = {f"{cut}_wp": wp for cut, wp in zip(scan_cuts, wps)}
            if args.resonant:
                selection_regions = get_res_selection_regions(args.year, **cutargs)
            elif args.vbf:
                selection_regions = get_nonres_vbf_selection_regions(args.year, **cutargs)
            else:
                selection_regions = get_nonres_selection_regions(args.year, **cutargs)

            # load pre-calculated systematics and those for different years if saved already
            systs_file = template_dir / "systematics.json"
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
                args.bdt_preds_dir,
                template_dir,
                systs_file,
                args.signal_data_dirs[0],
            )

            print("\nMaking templates")
            templates = {}

            jshifts = [""] + jec_shifts + jmsr_shifts if args.do_jshifts else [""]
            for jshift in jshifts:
                print(jshift)
                plot_dir = args.templates_plots_dir / cutstr if args.plot_dir != "" else ""
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
                    plot_sig_keys=plot_sig_keys_nonres if not args.resonant else sig_keys,
                    plot_dir=plot_dir,
                    prev_cutflow=cutflow,
                    # sig_splits=sig_splits,
                    sig_scale_dict=sig_scale_dict,
                    weight_shifts=weight_shifts,
                    jshift=jshift,
                    blind_pass=bool(args.resonant),
                    show=False,
                    plot_shifts=args.plot_shifts,
                )
                templates = {**templates, **temps}

            print("\nSaving templates")
            save_templates(
                templates, template_dir / f"{args.year}_templates.pkl", args.resonant, shape_vars
            )

            with systs_file.open("w") as f:
                json.dump(systematics, f)


def _init(args):
    if not (args.control_plots or args.bdt_plots or args.templates):
        print("You need to pass at least one of --control-plots, --bdt-plots, or --templates")
        return None

    if args.resonant:
        scan = (
            len(args.res_txbb_wp) > 1
            or len(args.res_thww_wp) > 1
            or len(args.res_leading_pt) > 1
            or len(args.res_subleading_pt) > 1
        )
        scan_wps = list(
            itertools.product(
                args.res_txbb_wp, args.res_thww_wp, args.res_leading_pt, args.res_subleading_pt
            )
        )
        # remove WPs where subleading pT > leading pT
        scan_wps = [wp for wp in scan_wps if wp[3] <= wp[2]]
        scan_cuts = res_scan_cuts
        shape_vars = res_shape_vars
    elif not args.vbf:
        scan = (
            len(args.nonres_txbb_wp) > 1 or len(args.nonres_bdt_wp) > 1 or len(args.lepton_veto) > 1
        )
        scan_wps = list(
            itertools.product(args.nonres_txbb_wp, args.nonres_bdt_wp, args.lepton_veto)
        )
        scan_cuts = nonres_scan_cuts
        shape_vars = nonres_shape_vars
    else:
        scan = len(args.nonres_vbf_txbb_wp) > 1 or len(args.nonres_vbf_thww_wp) > 1
        scan_wps = list(itertools.product(args.nonres_vbf_txbb_wp, args.nonres_vbf_thww_wp))
        scan_cuts = nonres_vbf_scan_cuts
        shape_vars = nonres_vbf_shape_vars

    return shape_vars, scan, scan_cuts, scan_wps


# adds all necessary columns to dataframes from events_dict
def _add_nonres_columns(df, bb_mask, vbf_vars=False, ptlabel="", mlabel=""):
    """Variables needed for ggF and/or VBF BDTs"""

    bbJet = utils.make_vector(df, "bbFatJet", bb_mask, ptlabel=ptlabel, mlabel=mlabel)
    VVJet = utils.make_vector(df, "VVFatJet", bb_mask, ptlabel=ptlabel, mlabel=mlabel)
    Dijet = bbJet + VVJet

    if f"DijetPt{ptlabel}{mlabel}" not in df.columns:
        df[f"DijetMass{ptlabel}{mlabel}"] = Dijet.mass
    df[f"DijetPt{ptlabel}{mlabel}"] = Dijet.pt
    df[f"VVFatJetPtOverbbFatJetPt{ptlabel}{mlabel}"] = VVJet.pt / bbJet.pt
    df[f"VVFatJetPtOverDijetPt{ptlabel}{mlabel}"] = VVJet.pt / df[f"DijetPt{ptlabel}{mlabel}"]

    if not vbf_vars:
        return

    import vector

    vbf1 = vector.array(
        {
            "pt": df[(f"VBFJetPt{ptlabel}", 0)],
            "phi": df[("VBFJetPhi", 0)],
            "eta": df[("VBFJetEta", 0)],
            "M": df[("VBFJetMass", 0)],
        }
    )

    vbf2 = vector.array(
        {
            "pt": df[(f"VBFJetPt{ptlabel}", 1)],
            "phi": df[("VBFJetPhi", 1)],
            "eta": df[("VBFJetEta", 1)],
            "M": df[("VBFJetMass", 1)],
        }
    )

    jj = vbf1 + vbf2

    # Adapted from HIG-20-005 ggF_Killer 6.2.2
    # https://coffeateam.github.io/coffea/api/coffea.nanoevents.methods.vector.PtEtaPhiMLorentzVector.html
    # https://coffeateam.github.io/coffea/api/coffea.nanoevents.methods.vector.LorentzVector.html
    # Adding variables defined in HIG-20-005 that show strong differentiation for VBF signal events and background

    # separation between both ak8 higgs jets
    if "vbf_dR_HH" not in df.columns:
        df["vbf_dR_HH"] = VVJet.deltaR(bbJet)
    if "vbf_dR_j0_HVV" not in df.columns:
        df["vbf_dR_j0_HVV"] = vbf1.deltaR(VVJet)
    if "vbf_dR_j1_HVV" not in df.columns:
        df["vbf_dR_j1_HVV"] = vbf2.deltaR(VVJet)
    if "vbf_dR_j0_Hbb" not in df.columns:
        df["vbf_dR_j0_Hbb"] = vbf1.deltaR(bbJet)
    if "vbf_dR_j1_Hbb" not in df.columns:
        df["vbf_dR_j1_Hbb"] = vbf2.deltaR(bbJet)
    if "vbf_dR_jj" not in df.columns:
        df["vbf_dR_jj"] = vbf1.deltaR(vbf2)
    if "vbf_Mass_jj{ptlabel}" not in df.columns:
        df[f"vbf_Mass_jj{ptlabel}"] = jj.M
    if "vbf_dEta_jj" not in df.columns:
        df["vbf_dEta_jj"] = np.abs(vbf1.eta - vbf2.eta)

    if "DijetdEta" not in df.columns:
        df["DijetdEta"] = np.abs(bbJet.eta - VVJet.eta)
    if "DijetdPhi" not in df.columns:
        df["DijetdPhi"] = np.abs(bbJet.phi - VVJet.phi)

    # Subleading VBF-jet cos(θ) in the HH+2j center of mass frame:
    # https://github.com/scikit-hep/vector/blob/main/src/vector/_methods.py#L916
    system_4vec = vbf1 + vbf2 + VVJet + bbJet
    j1_CMF = vbf1.boostCM_of_p4(system_4vec)

    # Leading VBF-jet cos(θ) in the HH+2j center of mass frame:
    thetab1 = 2 * np.arctan(np.exp(-j1_CMF.eta))
    thetab1 = np.cos(thetab1)  # 12

    if f"vbf_cos_j1{ptlabel}{mlabel}" not in df.columns:
        df[f"vbf_cos_j1{ptlabel}{mlabel}"] = np.abs(thetab1)

    # Subleading VBF-jet cos(θ) in the HH+2j center of mass frame:
    j2_CMF = vbf2.boostCM_of_p4(system_4vec)
    thetab2 = 2 * np.arctan(np.exp(-j2_CMF.eta))
    thetab2 = np.cos(thetab2)
    if f"vbf_cos_j2{ptlabel}{mlabel}" not in df.columns:
        df[f"vbf_cos_j2{ptlabel}{mlabel}"] = np.abs(thetab2)

    if "vbf_prod_centrality" not in df.columns:
        # H1-centrality * H2-centrality:
        delta_eta = vbf1.eta - vbf2.eta
        avg_eta = (vbf1.eta + vbf2.eta) / 2
        prod_centrality = np.exp(
            -np.power((VVJet.eta - avg_eta) / delta_eta, 2)
            - np.power((bbJet.eta - avg_eta) / delta_eta, 2)
        )
        df["vbf_prod_centrality"] = prod_centrality


def _process_samples(args, BDT_sample_order: list[str] = None):
    sig_samples = res_samples if args.resonant else nonres_samples
    sig_samples = deepcopy(sig_samples)

    if not args.resonant and BDT_sample_order is None and not args.vbf:
        with (args.bdt_preds_dir / f"{args.year}/sample_order.txt").open() as f:
            BDT_sample_order = list(eval(f.read()).keys())

    if args.read_sig_samples:
        # read all signal samples in directory
        read_samples = os.listdir(f"{args.signal_data_dirs[0]}/{args.year}")
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

        # can load in nonres samples for control plots
        for sig_key in args.sig_samples:
            if sig_key in nonres_samples:
                sig_samples[sig_key] = nonres_samples[sig_key]

        if len(sig_samples):
            # re-order according to input ordering
            tsig_samples = OrderedDict()
            for sample in args.sig_samples:
                if sample in sig_samples:
                    # if sample is a key, get it directly
                    tsig_samples[sample] = sig_samples[sample]
                else:
                    # else if it is a value, have to find corresponding key
                    key = next(key for key, value in sig_samples.items() if value == sample)
                    tsig_samples[key] = sample

            sig_samples = tsig_samples

    bg_samples = deepcopy(samples)
    for bg_key in list(bg_samples.keys()):
        if bg_key not in args.bg_keys and bg_key != data_key:
            del bg_samples[bg_key]

    if not args.resonant and not args.vbf:
        for key in sig_samples.copy():
            if key not in BDT_sample_order:
                del sig_samples[key]

        for key in bg_samples.copy():
            if key not in BDT_sample_order:
                del bg_samples[key]

    if not args.data:
        try:
            del bg_samples[data_key]
        except:
            print(f"no key {data_key}")

    sig_keys = list(sig_samples.keys())
    bg_keys = list(bg_samples.keys())

    print("BDT Sample Order: ", BDT_sample_order)
    print("Sig keys: ", sig_keys)
    # print("Sig samples: ", sig_samples)
    print("BG keys: ", bg_keys)
    # print("BG Samples: ", bg_samples)

    return sig_keys, sig_samples, bg_keys, bg_samples


def _make_dirs(args, scan, scan_cuts, scan_wps):
    if args.plot_dir != "":
        args.plot_dir = Path(args.plot_dir)
        args.plot_dir.mkdir(parents=True, exist_ok=True)

        if args.control_plots:
            args.control_plots_dir = args.plot_dir / "ControlPlots" / args.year
            args.control_plots_dir.mkdir(parents=True, exist_ok=True)

        if args.bdt_plots and not args.resonant and not args.vbf:
            args.bdt_sculpting_plots_dir = args.plot_dir / "BDTSculpting"
            args.bdt_sculpting_plots_dir.mkdir(parents=True, exist_ok=True)

        if args.templates:
            args.templates_plots_dir = args.plot_dir / "Templates" / args.year
            args.templates_plots_dir.mkdir(parents=True, exist_ok=True)

            if scan:
                for wps in scan_wps:
                    cutstr = "_".join([f"{cut}_{wp}" for cut, wp in zip(scan_cuts, wps)])
                    (args.templates_plots_dir / f"{cutstr}/wshifts").mkdir(
                        parents=True, exist_ok=True
                    )
                    (args.templates_plots_dir / f"{cutstr}/jshifts").mkdir(
                        parents=True, exist_ok=True
                    )
            else:
                (args.templates_plots_dir / "wshifts").mkdir(parents=True, exist_ok=True)
                (args.templates_plots_dir / "jshifts").mkdir(parents=True, exist_ok=True)
                if args.resonant:
                    (args.templates_plots_dir / "hists2d").mkdir(parents=True, exist_ok=True)

    elif args.control_plots or args.bdt_plots:
        print(
            "You need to pass --plot-dir if you want to make control plots or BDT plots. Exiting."
        )
        sys.exit()

    if args.template_dir != "":
        args.template_dir = Path(args.template_dir)
        if scan:
            for wps in scan_wps:
                cutstr = "_".join([f"{cut}_{wp}" for cut, wp in zip(scan_cuts, wps)])
                Path(f"{args.template_dir}/{args.templates_name}/cutflows/{args.year}").mkdir(
                    parents=True, exist_ok=True
                )
        else:
            Path(f"{args.template_dir}/{args.templates_name}/cutflows/{args.year}").mkdir(
                parents=True, exist_ok=True
            )


def _normalize_weights(
    events: pd.DataFrame, totals: dict, sample: str, isData: bool, variations: bool = True
):
    """Normalize weights and all the variations"""
    # don't need any reweighting for data
    if isData:
        events["finalWeight"] = events["weight"]
        return

    # check weights are scaled
    if "weight_noxsec" in events and np.all(events["weight"] == events["weight_noxsec"]):
        warnings.warn(f"{sample} has not been scaled by its xsec and lumi!", stacklevel=1)

    # checking that trigger efficiencies have been applied
    if "weight_noTrigEffs" in events and not np.all(
        np.isclose(events["weight"], events["weight_noTrigEffs"], rtol=1e-5)
    ):
        # normalize weights with and without trigger efficiencies
        events["finalWeight"] = events["weight"] / totals["np_nominal"]
        events["weight_noTrigEffs"] /= totals["np_nominal"]
    else:
        events["weight"] /= totals["np_nominal"]

    if not variations:
        return

    # normalize all the variations
    for wvar in weight_shifts:
        if f"weight_{wvar}Up" not in events:
            continue

        for shift in ["Up", "Down"]:
            wlabel = wvar + shift
            if wvar in norm_preserving_weights:
                # normalize by their totals
                events[f"weight_{wlabel}"] /= totals[f"np_{wlabel}"]
            else:
                # normalize by the nominal
                events[f"weight_{wlabel}"] /= totals["np_nominal"]

    # create lepton weights - using Hbb weights for now
    # TODO after finalizing lepton vetoes:
    # 1) choose the right (Hbb vs HH) lepton id weights
    # 2) multiply all weights by nominal lepton id weights
    if "single_weight_electron_hbb_id_Loose" in events:
        for new_key, old_key in [
            ("electron_id", "electron_hbb_id_Loose"),
            ("muon_id", "muon_hbb_id_Loose"),
        ]:
            for shift in ["Up", "Down"]:
                events[f"weight_{new_key}{shift}"] = (
                    events["finalWeight"] * events[f"single_weight_{old_key}{shift}"][0]
                )

    # normalize scale and PDF weights
    for wkey in ["scale_weights", "pdf_weights"]:
        if wkey in events:
            # .to_numpy() makes it way faster
            events[wkey] = events[wkey].to_numpy() / totals[f"np_{wkey}"]


def load_samples(
    data_dir: Path,
    samples: dict[str, str],
    year: str,
    filters: list = None,
    columns: list = None,
    hem_cleaning: bool = True,
    variations: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Loads events with an optional filter.
    Divides MC samples by the total pre-skimming, to take the acceptance into account.

    Args:
        data_dir (str): path to data directory.
        samples (Dict[str, str]): dictionary of samples and selectors to load.
        year (str): year.
        filters (List): Optional filters when loading data.
        columns (List): Optional columns to load.
        hem_cleaning (bool): Whether to apply HEM cleaning to 2018 data.
        variations (bool): Normalize variations as well (saves time to not do so). Defaults to True.

    Returns:
        Dict[str, pd.DataFrame]: ``events_dict`` dictionary of events dataframe for each sample.

    """
    data_dir = Path(data_dir) / year
    full_samples_list = os.listdir(data_dir)  # get all directories in data_dir
    events_dict = {}

    # label - key of sample in events_dict
    # selector - string used to select directories to load in for this sample
    for label, selector in samples.items():
        events_dict[label] = []  # list of directories we load in for this sample
        for sample in full_samples_list:
            # check if this directory passes our selector string
            if not utils.check_selector(sample, selector):
                continue

            sample_path = data_dir / sample
            parquet_path, pickles_path = sample_path / "parquet", sample_path / "pickles"

            # no parquet directory?
            if not parquet_path.exists():
                if not (
                    (year == "2016" and sample.endswith("HIPM"))
                    or (year == "2016APV" and not sample.endswith("HIPM"))
                ):  # don't complain about 2016/HIPM mismatch
                    warnings.warn(f"No parquet directory for {sample}!", stacklevel=1)
                continue

            # print(f"Loading {sample}")
            events = pd.read_parquet(parquet_path, filters=filters, columns=columns)

            # no events?
            if not len(events):
                warnings.warn(f"No events for {sample}!", stacklevel=1)
                continue

            # normalize by total events
            totals = utils.get_pickles(pickles_path, year, sample)["totals"]
            _normalize_weights(
                events, totals, sample, isData=label == data_key, variations=variations
            )

            if year == "2018" and hem_cleaning:
                events = utils._hem_cleaning(sample, events)

            events_dict[label].append(events)
            print(f"Loaded {sample: <50}: {len(events)} entries")

        if len(events_dict[label]):
            events_dict[label] = pd.concat(events_dict[label])
        else:
            del events_dict[label]

    return events_dict


def _check_load_systematics(systs_file: str, year: str):
    if systs_file.exists():
        print("Loading systematics")
        with systs_file.open() as f:
            systematics = json.load(f)
    else:
        systematics = {}

    if year not in systematics:
        systematics[year] = {}

    return systematics


def _load_samples(args, samples, sig_samples, cutflow, variations=True):
    """Wrapper for load_samples function"""
    filters = load_filters if args.filters else None

    events_dict = {}
    for d in args.signal_data_dirs:
        events_dict = {
            **events_dict,
            **load_samples(
                d,
                sig_samples,
                args.year,
                filters,
                hem_cleaning=args.hem_cleaning,
                variations=variations,
            ),
        }

    if args.data_dir:
        events_dict = {
            **events_dict,
            **load_samples(
                args.data_dir,
                samples,
                args.year,
                filters,
                hem_cleaning=args.hem_cleaning,
                variations=variations,
            ),
        }

    utils.add_to_cutflow(events_dict, "Pre-selection", "finalWeight", cutflow)
    print(cutflow)

    return events_dict


def apply_trigger_weights(events_dict: dict[str, pd.DataFrame], year: str, cutflow: pd.DataFrame):
    """Applies trigger scale factors to the events."""
    from coffea.lookup_tools.dense_lookup import dense_lookup

    with Path(f"../corrections/trigEffs/{year}_combined.pkl").open("rb") as filehandler:
        combined = pickle.load(filehandler)

    # sum over TH4q bins
    effs_txbb = combined["num"][:, sum, :, :] / combined["den"][:, sum, :, :]

    ak8TrigEffsLookup = dense_lookup(
        np.nan_to_num(effs_txbb.view(flow=False), 0), np.squeeze(effs_txbb.axes.edges)
    )

    weight_key = "finalWeight"

    for sample in events_dict:
        events = events_dict[sample]
        if sample == data_key:
            if weight_key not in events:
                events[weight_key] = events["weight"]
        elif f"{weight_key}_noTrigEffs" not in events:
            fj_trigeffs = ak8TrigEffsLookup(
                events["ak8FatJetParticleNetMD_Txbb"].to_numpy(),
                events["ak8FatJetPt"].to_numpy(),
                events["ak8FatJetMsd"].to_numpy(),
            )
            # combined eff = 1 - (1 - fj1_eff) * (1 - fj2_eff)
            combined_trigEffs = 1 - np.prod(1 - fj_trigeffs, axis=1, keepdims=True)
            events[f"{weight_key}_noTrigEffs"] = events["weight"]
            events[weight_key] = events["weight"] * combined_trigEffs

    if cutflow is not None:
        utils.add_to_cutflow(events_dict, "TriggerEffs", weight_key, cutflow)


def qcd_sf(events_dict: dict[str, pd.DataFrame], cutflow: pd.DataFrame):
    """Applies a QCD scale factor."""
    trig_yields = cutflow.iloc[:, -1]
    non_qcd_bgs_yield = np.sum(
        [
            trig_yields[sample]
            for sample in events_dict
            if sample not in {*nonres_sig_keys, qcd_key, data_key, *res_sig_keys}
        ]
    )
    QCD_SCALE_FACTOR = (trig_yields[data_key] - non_qcd_bgs_yield) / trig_yields[qcd_key]
    events_dict[qcd_key]["finalWeight"] *= QCD_SCALE_FACTOR

    print(f"\n{QCD_SCALE_FACTOR = }")

    if cutflow is not None:
        utils.add_to_cutflow(events_dict, "QCD SF", "finalWeight", cutflow)


def apply_weights(
    events_dict: dict[str, pd.DataFrame],
    year: str,
    cutflow: pd.DataFrame = None,
    trigger_effs: bool = True,
    do_qcd_sf: bool = True,
):
    """
    Applies (1) 2D trigger scale factors, (2) QCD scale facotr.

    Args:
        cutflow (pd.DataFrame): cutflow to which to add yields after scale factors.
        weight_key (str): column in which to store scaled weights in. Defaults to "finalWeight".

    """
    if trigger_effs:
        apply_trigger_weights(events_dict, year, cutflow)

    if do_qcd_sf:
        qcd_sf(events_dict, cutflow)


def bb_VV_assignment(events_dict: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
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


def derive_variables(
    events_dict: dict[str, pd.DataFrame],
    bb_masks: dict[str, pd.DataFrame],
    nonres_vars: bool = True,
    vbf_vars: bool = False,
    do_jshifts: bool = True,
):
    """Add Dijet variables"""
    for sample, events in events_dict.items():
        if not nonres_vars:
            continue

        bb_mask = bb_masks[sample]
        _add_nonres_columns(events, bb_mask, vbf_vars=vbf_vars)

        if sample == data_key or not do_jshifts:
            continue

        for var in jec_shifts:
            _add_nonres_columns(events, bb_mask, vbf_vars=vbf_vars, ptlabel=f"_{var}")

        for var in jmsr_shifts:
            _add_nonres_columns(events, bb_mask, vbf_vars=vbf_vars, mlabel=f"_{var}")


def load_bdt_preds(
    events_dict: dict[str, pd.DataFrame],
    year: str,
    bdt_preds_dir: Path,
    jec_jmsr_shifts: bool = False,
):
    """
    Loads the BDT scores for each event and saves in the dataframe in the "BDTScore" column.
    If ``jec_jmsr_shifts``, also loads BDT preds for every JEC / JMSR shift in MC.

    Args:
        bdt_preds (str): Path to the bdt_preds .npy file.
        bdt_sample_order (List[str]): Order of samples in the predictions file.

    """
    with (bdt_preds_dir / year / "sample_order.txt").open() as f:
        sample_order_dict = eval(f.read())

    bdt_preds = np.load(f"{bdt_preds_dir}/{year}/preds.npy")

    multiclass = len(bdt_preds.shape) > 1

    if jec_jmsr_shifts:
        shift_preds = {
            jshift: np.load(f"{bdt_preds_dir}/{year}/preds_{jshift}.npy")
            for jshift in jec_shifts + jmsr_shifts
        }

    i = 0
    for sample, num_events in sample_order_dict.items():
        if sample in events_dict:
            events = events_dict[sample]
            assert num_events == len(
                events
            ), f"# of BDT predictions does not match # of events for sample {sample}"

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
    sig_key: str,
    data_dir: Path,
    samples: dict,
    lp_region: Region,
    bdt_preds_dir: Path = None,
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
    ]

    load_columns += hh_vars.lp_sf_vars

    # reformat into ("column name", "idx") format for reading multiindex columns
    column_labels = []
    for key, num_columns in load_columns:
        for i in range(num_columns):
            column_labels.append(f"('{key}', '{i}')")

    for year in years:
        events_dict = load_samples(
            data_dir, {sig_key: samples[sig_key]}, year, load_filters, column_labels
        )

        # print weighted sample yields
        wkey = "finalWeight"
        print(np.sum(events_dict[sig_key][wkey].to_numpy()))

        bb_masks = bb_VV_assignment(events_dict)
        derive_variables(events_dict, bb_masks, nonres_vars=False, do_jshifts=False)
        events_dict[sig_key] = postprocess_lpsfs(events_dict[sig_key])

        if bdt_preds_dir is not None:
            with (bdt_preds_dir / year / "sample_order.txt").open() as f:
                sample_order_dict = eval(f.read())

            # load bdt preds for sig only
            bdt_preds = np.load(f"{bdt_preds_dir}/{year}/preds.npy")
            multiclass = len(bdt_preds.shape) > 1
            i = 0
            for sample, num_events in sample_order_dict.items():
                if sample != sig_key:
                    i += num_events
                    continue

                events = events_dict[sample]
                assert num_events == len(
                    events
                ), f"# of BDT predictions does not match # of events for sample {sample}"
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
    events_dict: dict[str, pd.DataFrame],
    bb_masks: dict[str, pd.DataFrame],
    sig_keys: list[str],
    sig_samples: dict[str, str],
    cutflow: pd.DataFrame,
    lp_selection_region: Region,
    systematics: dict,
    all_years: bool = False,
    bdt_preds_dir: Path = None,
    template_dir: Path = None,
    systs_file: Path = None,
    data_dir: Path = None,
):
    """
    1) Calculates LP SFs for each signal, if not already in ``systematics``
        - Does it for all years once if args.lp_sf_all_years or just for the given year
    2) Saves them to ``systs_file`` and CSV for posterity
    """
    if not all_years:
        warnings.warn(
            "LP SF only calculated from single year's samples", RuntimeWarning, stacklevel=1
        )

    for sig_key in sig_keys:
        if sig_key not in systematics or "lp_sf" not in systematics[sig_key]:
            print(f"\nGetting LP SFs for {sig_key}")

            if sig_key not in systematics:
                systematics[sig_key] = {}

            # SFs are correlated across all years so needs to be calculated with full dataset
            if all_years:
                lp_sf, unc, uncs = get_lpsf_all_years(
                    sig_key,
                    data_dir,
                    sig_samples,
                    lp_selection_region,
                    bdt_preds_dir,
                )
            # Only for testing, can do just for a single year
            else:
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
            systematics[sig_key]["lp_sf_uncs"] = uncs

            if systs_file is not None:
                with systs_file.open("w") as f:
                    json.dump(systematics, f)

    if template_dir is not None:
        sf_table = OrderedDict()  # format SFs for each sig key in a table
        for sig_key in sig_keys:
            systs = systematics[sig_key]
            sf_table[sig_key] = {
                "SF": f"{systs['lp_sf']:.2f} ± {systs['lp_sf'] * systs['lp_sf_unc']:.2f}",
                **systs["lp_sf_uncs"],
            }

        print("\nLP Scale Factors:\n", pd.DataFrame(sf_table).T)
        pd.DataFrame(sf_table).T.to_csv(f"{template_dir}/lpsfs.csv")

    utils.add_to_cutflow(events_dict, "LP SF", "finalWeight", cutflow)


def hists_HEM2d(
    events_dict: dict[str, pd.DataFrame],
    bb_masks: dict[str, pd.DataFrame],
    weight_key: str = "finalWeight",
    selection: dict[str, np.ndarray] = None,
):
    """Create 2D hists of FatJet phi vs eta for bb and VV jets as a check for HEM cleaning."""
    HEM2d_vars = [
        {
            f"{jet}FatJetPhi": ([40, -3.5, 3.5], rf"$\varphi^{{{jet}}}$"),
            f"{jet}FatJetEta": ([40, -3, 3], rf"$\eta^{{{jet}}}$"),
        }
        for jet in ["bb", "VV"]
    ]

    samples = list(events_dict.keys())
    hists2d = []

    for vars2d in HEM2d_vars:
        h = Hist(
            hist.axis.StrCategory(samples, name="Sample"),
            *[
                hist.axis.Regular(*bins, name=var, label=label)
                for var, (bins, label) in vars2d.items()
            ],
            storage=hist.storage.Weight(),
        )

        for sample in samples:
            events = events_dict[sample]

            fill_data = {var: utils.get_feat(events, var, bb_masks[sample]) for var in vars2d}
            weight = events[weight_key].to_numpy().squeeze()

            if selection is not None:
                sel = selection[sample]
                for var in fill_data:
                    fill_data[var] = fill_data[var][sel]

                weight = weight[sel]

            if len(weight):
                h.fill(Sample=sample, **fill_data, weight=weight)

        hists2d.append(h)

    return hists2d


def control_plots(
    events_dict: dict[str, pd.DataFrame],
    bb_masks: dict[str, pd.DataFrame],
    sig_keys: list[str],
    control_plot_vars: list[ShapeVar],
    plot_dir: Path,
    year: str,
    weight_key: str = "finalWeight",
    hists: dict = None,
    cutstr: str = "",
    sig_splits: list[list[str]] = None,
    bg_keys: list[str] = bg_keys,
    selection: dict[str, np.ndarray] = None,
    sig_scale_dict: dict[str, float] = None,
    combine_pdf: bool = True,
    HEM2d: bool = False,
    plot_significance: bool = False,
    same_ylim: bool = False,
    show: bool = False,
    log: tuple[bool, str] = "both",
):
    """
    Makes and plots histograms of each variable in ``control_plot_vars``.

    Args:
        control_plot_vars (Dict[str, Tuple]): Dictionary of variables to plot, formatted as
          {var1: ([num bins, min, max], label), var2...}.
        sig_splits: split up signals into different plots (in case there are too many for one)
        HEM2d: whether to plot 2D hists of FatJet phi vs eta for bb and VV jets as a check for HEM cleaning.
        plot_significance: whether to plot the significance as well as the ratio plot.
        same_ylim: whether to use the same y-axis limits for all plots.
        log: True or False if plot on log scale or not - or "both" if both.
    """

    from PyPDF2 import PdfMerger

    # sig_scale_dict = utils.getSignalPlotScaleFactor(events_dict, sig_keys)
    # sig_scale_dict = {sig_key: 5e3 for sig_key in sig_keys}
    # sig_scale_dict["HHbbVV"] = 2e5

    if hists is None:
        hists = {}
    if sig_scale_dict is None:
        sig_scale_dict = {sig_key: 2e5 for sig_key in sig_keys}

    print(control_plot_vars)
    print(selection)
    print(list(events_dict.keys()))

    for shape_var in control_plot_vars:
        if shape_var.var not in hists:
            hists[shape_var.var] = utils.singleVarHist(
                events_dict, shape_var, bb_masks, weight_key=weight_key, selection=selection
            )

    ylim = np.max([h.values() for h in hists.values()]) * 1.05 if same_ylim else None

    if HEM2d and year == "2018":
        hists["HEM2d"] = hists_HEM2d(events_dict, bb_masks, weight_key, selection)

    with (plot_dir / "hists.pkl").open("wb") as f:
        pickle.dump(hists, f)

    if sig_splits is None:
        sig_splits = [sig_keys]

    do_log = [True, False] if log == "both" else [log]

    for log, logstr in [(False, ""), (True, "_log")]:
        if log not in do_log:
            continue

        for i, plot_sig_keys in enumerate(sig_splits):
            tplot_dir = plot_dir if len(sig_splits) == 1 else f"{plot_dir}/sigs{i}/"
            tsig_scale_dict = {key: sig_scale_dict.get(key, 1) for key in plot_sig_keys}

            merger_control_plots = PdfMerger()

            for shape_var in control_plot_vars:
                name = f"{tplot_dir}/{cutstr}{shape_var.var}{logstr}.pdf"
                plotting.ratioHistPlot(
                    hists[shape_var.var],
                    year,
                    plot_sig_keys,
                    bg_keys,
                    name=name,
                    sig_scale_dict=tsig_scale_dict if not log else None,
                    plot_significance=plot_significance,
                    significance_dir=shape_var.significance_dir,
                    show=show,
                    log=log,
                    ylim=ylim if not log else 1e15,
                )
                merger_control_plots.append(name)

            if combine_pdf:
                merger_control_plots.write(f"{tplot_dir}/{cutstr}{year}{logstr}_ControlPlots.pdf")

            merger_control_plots.close()

    if HEM2d and year == "2018":
        # TODO: change plot keys?
        name = f"{plot_dir}/HEM2d.pdf"
        # plot keys
        plotting.plot_HEM2d(
            hists["HEM2d"], ["Data", "QCD", "TT", "HHbbVV", "X[900]->H(bb)Y[80](VV)"], year, name
        )

    return hists


def plot_bdt_sculpting(
    events_dict: dict[str, pd.DataFrame],
    bb_masks: dict[str, pd.DataFrame],
    plot_dir: Path,
    year: str,
    weight_key: str = "finalWeight",
    show: bool = False,
):
    """Plot bb jet mass for different BDT score cuts."""
    cuts = [0, 0.1, 0.5, 0.9, 0.95]
    # bdtvars = ["", "TT", "VJets"]
    bdtvars = [""]
    plot_keys = ["QCD", "HHbbVV"]

    shape_var = ShapeVar(
        var="bbFatJetParticleNetMass", label=r"$m^{bb}_{reg}$ (GeV)", bins=[20, 50, 250]
    )

    for var in bdtvars:
        for key in plot_keys:
            ed_key = {key: events_dict[key]}
            bbm_key = {key: bb_masks[key]}

            plotting.cutsLinePlot(
                ed_key,
                bbm_key,
                shape_var,
                key,
                f"BDTScore{var}",
                cuts,
                year,
                weight_key,
                plot_dir,
                f"{year}_BDT{var}Cuts_{shape_var.var}_{key}",
                show=show,
            )


def _get_fill_data(
    events: pd.DataFrame, bb_mask: pd.DataFrame, shape_vars: list[ShapeVar], jshift: str = ""
):
    return {
        shape_var.var: utils.get_feat(
            events,
            shape_var.var if jshift == "" else utils.check_get_jec_var(shape_var.var, jshift),
            bb_mask,
        )
        for shape_var in shape_vars
    }


def _get_qcdvar_hists(
    events: pd.DataFrame, shape_vars: list[ShapeVar], fill_data: dict, wshift: str
):
    """Get histograms for QCD scale and PDF variations"""
    wkey = f"{wshift}_weights"
    cols = list(events[wkey].columns)
    h = Hist(
        hist.axis.StrCategory([str(i) for i in cols], name="Sample"),
        *[shape_var.axis for shape_var in shape_vars],
        storage="weight",
    )

    for i in cols:
        h.fill(
            Sample=str(i),
            **fill_data,
            weight=events[wkey][i],
        )
    return h


def get_templates(
    events_dict: dict[str, pd.DataFrame],
    bb_masks: dict[str, pd.DataFrame],
    year: str,
    sig_keys: list[str],
    selection_regions: dict[str, Region],
    shape_vars: list[ShapeVar],
    systematics: dict,
    template_dir: Path = "",
    bg_keys: list[str] = bg_keys,
    plot_dir: Path = "",
    prev_cutflow: pd.DataFrame = None,
    weight_key: str = "finalWeight",
    plot_sig_keys: list[str] = None,
    sig_splits: list[list[str]] = None,
    sig_scale_dict: dict = None,
    weight_shifts: dict = None,
    jshift: str = "",
    lpsfs: bool = True,
    plot_shifts: bool = False,
    pass_ylim: int = None,
    fail_ylim: int = None,
    blind_pass: bool = False,
    show: bool = False,
) -> dict[str, Hist]:
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
    if weight_shifts is None:
        weight_shifts = {}
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
            region.cuts,
            events_dict,
            bb_masks,
            prev_cutflow=prev_cutflow,
            jshift=jshift,
            weight_key=weight_key,
        )

        if template_dir != "":
            cf.to_csv(f"{template_dir}/cutflows/{year}/{rname}_cutflow{jlabel}.csv")

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
                # scale all signal weights by LP SF
                if lpsfs:
                    for wkey in utils.get_all_weights(sig_events[sig_key]):
                        sig_events[sig_key][wkey] *= systematics[sig_key]["lp_sf"]

                corrections.apply_txbb_sfs(sig_events[sig_key], sig_bb_mask, year, weight_key)

        # if not do_jshift:
        #     print("\nCutflow:\n", cf)

        # set up samples
        hist_samples = list(events_dict.keys())

        if not do_jshift:
            # add all weight-based variations to histogram axis
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
            weight = events[weight_key].to_numpy().squeeze()
            h.fill(Sample=sample, **fill_data, weight=weight)

            if not do_jshift:
                # add weight variations
                for wshift, wsyst in weight_shifts.items():
                    if sample in wsyst.samples and year in wsyst.years:
                        if wshift not in ["scale", "pdf"]:
                            # fill histogram with weight variations
                            for skey, shift in [("Down", "down"), ("Up", "up")]:
                                h.fill(
                                    Sample=f"{sample}_{wshift}_{shift}",
                                    **fill_data,
                                    weight=events[f"weight_{wshift}{skey}"].to_numpy().squeeze(),
                                )
                        else:
                            # get histograms for all QCD scale and PDF variations
                            whists = _get_qcdvar_hists(events, shape_vars, fill_data, wshift)

                            if wshift == "scale":
                                # renormalization / factorization scale uncertainty is the max/min envelope of the variations
                                shape_up = np.max(whists.values(), axis=0)
                                shape_down = np.min(whists.values(), axis=0)
                            else:
                                # pdf uncertainty is the norm of each variation (corresponding to 103 eigenvectors) - nominal
                                nom_vals = h[sample, :].values()
                                abs_unc = np.linalg.norm(
                                    (whists.values() - nom_vals), axis=0
                                )  # / np.sqrt(103)
                                # cap at 100% uncertainty
                                rel_unc = np.clip(abs_unc / nom_vals, 0, 1)
                                shape_up = nom_vals * (1 + rel_unc)
                                shape_down = nom_vals * (1 - rel_unc)

                            h.values()[
                                utils.get_key_index(h, f"{sample}_{wshift}_up"), :
                            ] = shape_up
                            h.values()[
                                utils.get_key_index(h, f"{sample}_{wshift}_down"), :
                            ] = shape_down

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

        ################################
        # Plot templates incl variations
        ################################

        if plot_dir != "" and (not do_jshift or plot_shifts):
            if plot_sig_keys is None:
                plot_sig_keys = sig_keys

            if sig_scale_dict is None:
                sig_scale_dict = {
                    **{skey: 1 for skey in nonres_sig_keys if skey in plot_sig_keys},
                    **{skey: 1 for skey in res_sig_keys if skey in plot_sig_keys},
                }

            title = (
                f"{region.label} Region Pre-Fit Shapes"
                if not do_jshift
                else f"{region.label} Region {jshift} Shapes"
            )

            if sig_splits is None:
                sig_splits = [plot_sig_keys]

            # don't plot qcd in the pass region
            if pass_region:
                p_bg_keys = [key for key in bg_keys if key != qcd_key]
            else:
                p_bg_keys = bg_keys

            for i, shape_var in enumerate(shape_vars):
                for j, p_sig_keys in enumerate(sig_splits):
                    split_str = "" if len(sig_splits) == 1 else f"sigs{j}_"
                    plot_params = {
                        "hists": h.project(0, i + 1),
                        "sig_keys": p_sig_keys,
                        "sig_scale_dict": (
                            {key: sig_scale_dict[key] for key in p_sig_keys}
                            if pass_region
                            else None
                        ),
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
                        bg_keys=bg_keys,
                        title=title,
                        name=f"{plot_name}{jlabel}.pdf",
                    )

                    if not do_jshift and plot_shifts:
                        plot_name = (
                            f"{plot_dir}/wshifts/" f"{split_str}{rname}_region_{shape_var.var}"
                        )

                        for wshift, wsyst in weight_shifts.items():
                            plotting.ratioHistPlot(
                                **plot_params,
                                bg_keys=bg_keys,
                                syst=(wshift, wsyst.samples),
                                title=f"{region.label} Region {wsyst.label} Unc.",
                                name=f"{plot_name}_{wshift}.pdf",
                            )

                            for skey, shift in [("Down", "down"), ("Up", "up")]:
                                plotting.ratioHistPlot(
                                    **plot_params,
                                    bg_keys=p_bg_keys,  # don't plot QCD
                                    syst=(wshift, wsyst.samples),
                                    variation=shift,
                                    title=f"{region.label} Region {wsyst.label} Unc. {skey} Shapes",
                                    name=f"{plot_name}_{wshift}_{shift}.pdf",
                                    plot_ratio=False,
                                )

                        if pass_region:
                            plotting.ratioHistPlot(
                                **plot_params,
                                bg_keys=bg_keys,
                                sig_err="txbb",
                                title=rf"{region.label} Region $T_{{Xbb}}$ Shapes",
                                name=f"{plot_name}_txbb.pdf",
                            )

    return templates


def save_templates(
    templates: dict[str, Hist], template_file: Path, resonant: bool, shape_vars: list[ShapeVar]
):
    """Creates blinded copies of each region's templates and saves a pickle of the templates"""

    if not resonant:
        from copy import deepcopy

        blind_window = shape_vars[0].blind_window

        for label, template in list(templates.items()):
            blinded_template = deepcopy(template)
            utils.blindBins(blinded_template, blind_window)
            templates[f"{label}Blinded"] = blinded_template

    with template_file.open("wb") as f:
        pickle.dump(templates, f)

    print("Saved templates to", template_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=None,
        help="path to skimmed parquet",
        type=str,
    )

    parser.add_argument(
        "--signal-data-dirs",
        default=[],
        help="path to skimmed signal parquets, if different from other data",
        nargs="*",
        type=str,
    )

    parser.add_argument(
        "--year",
        default="2017",
        choices=["2016", "2016APV", "2017", "2018"],
        type=str,
    )

    parser.add_argument(
        "--bdt-preds-dir",
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

    add_bool_arg(parser, "resonant", "for resonant or nonresonant", default=False)
    add_bool_arg(parser, "vbf", "non-resonant VBF or inclusive", default=False)
    add_bool_arg(parser, "control-plots", "make control plots", default=False)
    add_bool_arg(
        parser,
        "mass-plots",
        "make mass comparison plots (filters will automatically be turned off)",
        default=False,
    )
    add_bool_arg(parser, "bdt-plots", "make bdt sculpting plots", default=False)
    add_bool_arg(parser, "templates", "save m_bb templates using bdt cut", default=False)
    add_bool_arg(
        parser, "overwrite-template", "if template file already exists, overwrite it", default=False
    )
    add_bool_arg(parser, "do-jshifts", "Do JEC/JMC variations", default=True)
    add_bool_arg(parser, "plot-shifts", "Plot systematic variations as well", default=False)
    add_bool_arg(parser, "lp-sf-all-years", "Calculate one LP SF for all run 2", default=True)

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
        default=bg_keys,
        type=str,
    )

    add_bool_arg(parser, "read-sig-samples", "read signal samples from directory", default=False)

    add_bool_arg(parser, "data", "include data", default=True)
    add_bool_arg(parser, "hem-cleaning", "perform hem cleaning for 2018", default=False)
    add_bool_arg(parser, "HEM2d", "fatjet phi v eta plots to check HEM cleaning", default=False)
    add_bool_arg(parser, "filters", "apply filters", default=True)

    parser.add_argument(
        "--control-plot-vars",
        help="Specify control plot variables to plot. By default plots all.",
        default=[],
        nargs="*",
        type=str,
    )

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
        "--nonres-vbf-txbb-wp",
        help="Txbb WP for signal region. If multiple arguments, will make templates for each.",
        default=["HP"],
        choices=["LP", "MP", "HP"],
        nargs="*",
        type=str,
    )

    parser.add_argument(
        "--nonres-vbf-thww-wp",
        help="THWWvsT WP for signal region. If multiple arguments, will make templates for each.",
        default=[0.6],
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
        default=[0.8],
        nargs="*",
        type=float,
    )

    parser.add_argument(
        "--res-leading-pt",
        help="pT cut for leading AK8 jet (resonant only)",
        default=[300],
        nargs="*",
        type=float,
    )

    parser.add_argument(
        "--res-subleading-pt",
        help="pT cut for sub-leading AK8 jet (resonant only)",
        default=[300],
        nargs="*",
        type=float,
    )

    parser.add_argument(
        "--lepton-veto",
        help="lepton vetoes: None, Hbb, or HH",
        default=["None"],
        nargs="*",
        type=str,
    )

    args = parser.parse_args()

    if args.templates and args.template_dir == "":
        print("Need to set --template-dir if making templates. Exiting.")
        sys.exit()

    if not args.signal_data_dirs:
        args.signal_data_dirs = [args.data_dir]

    if args.bdt_preds_dir != "" and args.bdt_preds_dir is not None:
        args.bdt_preds_dir = Path(args.bdt_preds_dir)

    if args.bdt_preds_dir == "" and not args.resonant:
        args.bdt_preds_dir = f"{args.data_dir}/inferences/"
    elif args.resonant:
        args.bdt_preds_dir = None

    if args.hem_cleaning is None:
        # can't do HEM cleaning for non-resonant until BDT is re-inferenced
        args.hem_cleaning = bool(args.resonant or args.vbf)

    if args.mass_plots:
        args.control_plots = True
        args.filters = False

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
