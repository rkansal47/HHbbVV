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
from regions import (
    Region,
    get_nonres_selection_regions,
    get_nonres_vbf_selection_regions,
    get_res_selection_regions,
)
from utils import ShapeVar

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
    years,
)
from HHbbVV.run_utils import add_bool_arg

# ignore these because they don't seem to apply
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# from memory_profiler import profile


@dataclass
class Syst:
    samples: list[str] = None
    years: list[str] = field(default_factory=lambda: years)
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
    # ShapeVar(var="MET_pt", label=r"$p^{miss}_T$ [GeV]", bins=[20, 0, 300]),
    # ShapeVar(var="DijetEta", label=r"$\eta^{jj}$", bins=[20, -8, 8]),
    # ShapeVar(var="DijetPt", label=r"$p_T^{jj}$ [GeV]", bins=[20, 0, 750]),
    # ShapeVar(var="DijetMass", label=r"$m^{jj}$ [GeV]", bins=[20, 600, 4000]),
    # ShapeVar(var="bbFatJetEta", label=r"$\eta^{bb}$", bins=[20, -2.4, 2.4]),
    # ShapeVar(var="bbFatJetPhi", label=r"$\varphi^{bb}$", bins=[20, -3, 3]),
    # ShapeVar(
    #     var="bbFatJetPt", label=r"$p^{bb}_T$ [GeV]", bins=[20, 300, 2300], significance_dir="right"
    # ),
    # ShapeVar(
    #     var="bbFatJetParticleNetMass",
    #     label=r"$m^{bb}_{reg}$ [GeV]",
    #     bins=[20, 50, 250],
    #     significance_dir="bin",
    # ),
    # ShapeVar(var="bbFatJetMsd", label=r"$m^{bb}_{msd}$ [GeV]", bins=[20, 50, 250]),
    # ShapeVar(var="bbFatJetParticleNetMD_Txbb", label=r"$T^{bb}_{Xbb}$", bins=[20, 0.8, 1]),
    # ShapeVar(var="VVFatJetEta", label=r"$\eta^{VV}$", bins=[20, -2.4, 2.4]),
    # ShapeVar(var="VVFatJetPhi", label=r"$\varphi^{VV}$", bins=[20, -3, 3]),
    # ShapeVar(var="VVFatJetPt", label=r"$p^{VV}_T$ [GeV]", bins=[20, 300, 2300]),
    # ShapeVar(var="VVFatJetParticleNetMass", label=r"$m^{VV}_{reg}$ [GeV]", bins=[20, 50, 250]),
    # ShapeVar(var="VVFatJetMsd", label=r"$m^{VV}_{msd}$ [GeV]", bins=[20, 50, 250]),
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
    # ShapeVar(var="VVFatJetParTMD_THWWvsT", label=r"$T^{VV}_{HWW}$", bins=[20, 0, 1]),
    # ShapeVar(var="bbFatJetPtOverDijetPt", label=r"$p^{bb}_T / p_T^{jj}$", bins=[20, 0, 40]),
    # ShapeVar(var="VVFatJetPtOverDijetPt", label=r"$p^{VV}_T / p_T^{jj}$", bins=[20, 0, 40]),
    # ShapeVar(var="VVFatJetPtOverbbFatJetPt", label=r"$p^{VV}_T / p^{bb}_T$", bins=[20, 0.4, 2.0]),
    # ShapeVar(var="nGoodMuonsHbb", label=r"# of Muons", bins=[3, 0, 3]),
    # ShapeVar(var="nGoodMuonsHH", label=r"# of Muons", bins=[3, 0, 3]),
    # ShapeVar(var="nGoodElectronsHbb", label=r"# of Electrons", bins=[3, 0, 3]),
    # ShapeVar(var="nGoodElectronsHH", label=r"# of Electrons", bins=[3, 0, 3]),
    # ShapeVar(var="DijetdEta", label=r"$|\Delta\eta^{jj}|$", bins=[16, 0, 4]),
    # ShapeVar(var="DijetdPhi", label=r"$|\Delta\varphi^{jj}|$", bins=[16, 0, 3.2]),
    # ShapeVar(var="VBFJetPt0", label=r"Leading VBF-tagged Jet $p_T$", bins=[20, 20, 300]),
    # ShapeVar(var="VBFJetPt1", label=r"Sub-leading VBF-tagged Jet $p_T$", bins=[20, 20, 300]),
    # ShapeVar(var="VBFJetEta0", label=r"Leading VBF-tagged Jet $\eta$", bins=[9, -4.5, 4.5]),
    # ShapeVar(var="VBFJetEta1", label=r"Sub-leading VBF-tagged Jet $\eta$", bins=[9, -4.5, 4.5]),
    # ShapeVar(var="VBFJetPhi0", label=r"Leading VBF-tagged Jet $\varphi$", bins=[10, -3, 3]),
    # ShapeVar(var="VBFJetPhi1", label=r"Sub-leading VBF-tagged Jet $\varphi$", bins=[10, -3, 3]),
    # ShapeVar(var="vbf_Mass_jj", label=r"$m_{jj}^{VBF}$", bins=[20, 0, 1000]),
    # ShapeVar(var="vbf_dEta_jj", label=r"$|\Delta\eta_{jj}^{VBF}|$", bins=[20, 0, 6]),
    # removed if not ggF nonresonant
    ShapeVar(var="BDTScore", label=r"$BDT_{ggF}$", bins=[20, 0, 1]),
    ShapeVar(var="BDTScoreVBF", label=r"$BDT_{VBF}$", bins=[20, 0, 1]),
]


# for msd vs mreg comparison plots only
mass_plot_vars = [
    ShapeVar(var="bbFatJetParticleNetMass", label=r"$m^{bb}_{reg}$ [GeV]", bins=[30, 0, 300]),
    ShapeVar(var="bbFatJetMsd", label=r"$m^{bb}_{msd}$ [GeV]", bins=[30, 0, 300]),
    ShapeVar(var="VVFatJetParticleNetMass", label=r"$m^{VV}_{reg}$ [GeV]", bins=[30, 0, 300]),
    ShapeVar(var="VVFatJetMsd", label=r"$m^{VV}_{msd}$ [GeV]", bins=[30, 0, 300]),
]


# fitting on bb regressed mass for nonresonant
nonres_shape_vars = [
    ShapeVar(
        "bbFatJetParticleNetMass",
        r"$m^{bb}_\mathrm{Reg}$ [GeV]",
        [20, 50, 250],
        reg=True,
        blind_window=[100, 150],
    )
]


# templates saved in bb regressed mass for nonresonant VBF
nonres_vbf_shape_vars = [
    ShapeVar(
        "bbFatJetParticleNetMass",
        r"$m^{bb}_\mathrm{Reg}$ [GeV]",
        [20, 50, 250],
        reg=True,
        blind_window=[100, 150],
    )
]


# fitting on VV regressed mass + dijet mass for resonant
res_shape_vars = [
    ShapeVar(
        "VVFatJetParticleNetMass",
        r"$m^{VV}_\mathrm{Reg}$ [GeV]",
        list(range(50, 110, 10)) + list(range(110, 200, 15)) + [200, 220, 250],
        reg=False,
    ),
    ShapeVar(
        "DijetMass",
        r"$m^{jj}$ [GeV]",
        list(range(800, 1400, 100)) + [1400, 1600, 2000, 3000, 4400],
        reg=False,
    ),
]

nonres_scan_cuts = ["ggf_txbb", "ggf_bdt", "vbf_txbb", "vbf_bdt", "lepton_veto"]
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
    shape_vars, scan, scan_cuts, scan_wps, filters = _init(args)
    sig_keys, sig_samples, bg_keys, bg_samples = _process_samples(args)
    all_samples = sig_keys + bg_keys
    _make_dirs(args, scan, scan_cuts, scan_wps)  # make plot, template dirs if needed
    cutflow = pd.DataFrame(index=all_samples)  # save cutflow as pandas table

    if args.lpsfs or args.templates:
        if len(sig_keys):
            _lpsfs(args, filters, scan, scan_cuts, scan_wps, sig_keys, sig_samples)
        if not (args.templates or args.bdt_plots or args.control_plots):
            return

    # only need to worry about variations if making templates
    events_dict = _load_samples(args, bg_samples, sig_samples, cutflow, variations=args.templates)
    bb_masks = bb_VV_assignment(events_dict)
    # QCD xsec normalization for plots
    qcd_sf(events_dict, cutflow, weight_key="finalWeight")

    derive_variables(
        events_dict,
        bb_masks,
        resonant=args.resonant,
        nonres_vars=args.vbf or (args.control_plots and not args.mass_plots),
        # nonres_vars=args.vbf,
        vbf_vars=args.vbf,
        do_jshifts=args.vbf,  # only need shifts for BDT pre-processing
    )

    # args has attr if --control-plots arg was set
    if hasattr(args, "control_plots_dir"):
        cutflow.to_csv(args.control_plots_dir / "preselection_cutflow.csv")

    print("\nCutflow", cutflow)

    # Load BDT Scores
    if not args.resonant and not args.vbf and not args.mass_plots:
        print("\nLoading BDT predictions")
        load_bdt_preds(
            events_dict,
            args.year,
            args.bdt_preds_dir,
            jec_jmsr_shifts=args.templates and args.do_jshifts,
        )
    else:
        for var in control_plot_vars.copy():
            if var.var.startswith("BDTScore"):
                control_plot_vars.remove(var)

    # Control plots
    if args.control_plots:
        print("\nMaking control plots\n")
        plot_vars = mass_plot_vars if args.mass_plots else control_plot_vars
        if len(args.control_plot_vars):
            for var in plot_vars.copy():
                if var.var not in args.control_plot_vars:
                    print("Removing: ", var.var)
                    plot_vars.remove(var)

        print("Plotting: ", [var.var for var in plot_vars])
        if args.resonant:
            p_sig_keys = sig_keys
            sig_scale_dict = {"HHbbVV": 1e5, "VBFHHbbVV": 2e6} | {key: 2e4 for key in res_sig_keys}
        else:
            p_sig_keys = plot_sig_keys_nonres
            sig_scale_dict = {
                "HHbbVV": 3e5,
                "VBFHHbbVV": 6e6,
                "qqHH_CV_1_C2V_0_kl_1_HHbbVV": 2e4,
                "qqHH_CV_1_C2V_2_kl_1_HHbbVV": 2e4,
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
            plot_ratio=not args.mass_plots,  # don't need data/MC ratio for mreg vs msd comparison
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
            cutstr, template_dir, selection_regions = _get_scan_regions(args, scan, scan_cuts, wps)

            # load pre-calculated systematics and those for different years if saved already
            systs_file = template_dir / "systematics.json"
            systematics = _check_load_systematics(systs_file, args.year, args.override_systs)

            print(systematics)

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
                    plot_sig_keys=(
                        list(set(plot_sig_keys_nonres).intersection(sig_keys))
                        if not args.resonant
                        else sig_keys
                    ),
                    plot_dir=plot_dir,
                    prev_cutflow=cutflow,
                    # sig_splits=sig_splits,
                    sig_scale_dict=sig_scale_dict,
                    weight_shifts=weight_shifts,
                    jshift=jshift,
                    blind=args.blinded,
                    blind_pass=bool(args.resonant),
                    show=False,
                    plot_shifts=args.plot_shifts,
                )
                templates = {**templates, **temps}

            print("\nSaving templates")
            save_templates(
                templates,
                template_dir / f"{args.year}_templates.pkl",
                args.blinded,
                args.resonant,
                shape_vars,
            )

            with systs_file.open("w") as f:
                json.dump(systematics, f, indent=4)


def _init(args):
    if not (args.control_plots or args.bdt_plots or args.templates or args.lpsfs):
        print(
            "You need to pass at least one of --control-plots, --bdt-plots, --templates, or --lpsfs"
        )
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
            len(args.nonres_ggf_txbb_wp) > 1
            or len(args.nonres_ggf_bdt_wp) > 1
            or len(args.nonres_vbf_txbb_wp) > 1
            or len(args.nonres_vbf_bdt_wp) > 1
            or len(args.lepton_veto) > 1
        )
        scan_wps = list(
            itertools.product(
                args.nonres_ggf_txbb_wp,
                args.nonres_ggf_bdt_wp,
                args.nonres_vbf_txbb_wp,
                args.nonres_vbf_bdt_wp,
                args.lepton_veto,
            )
        )
        scan_cuts = nonres_scan_cuts
        shape_vars = nonres_shape_vars
    else:
        scan = len(args.nonres_vbf_txbb_wp) > 1 or len(args.nonres_vbf_thww_wp) > 1
        scan_wps = list(itertools.product(args.nonres_vbf_txbb_wp, args.nonres_vbf_thww_wp))
        scan_cuts = nonres_vbf_scan_cuts
        shape_vars = nonres_vbf_shape_vars

    filters = load_filters if args.filters else None

    return shape_vars, scan, scan_cuts, scan_wps, filters


def _get_scan_regions(args, scan, scan_cuts, wps):
    cutstr = "_".join([f"{cut}_{wp}" for cut, wp in zip(scan_cuts, wps)]) if scan else ""
    template_dir = args.template_dir / cutstr / args.templates_name
    (template_dir / "cutflows" / args.year).mkdir(parents=True, exist_ok=True)

    cutargs = {f"{cut}_wp": wp for cut, wp in zip(scan_cuts, wps)}
    if args.resonant:
        selection_regions = get_res_selection_regions(args.year, **cutargs)
    elif args.vbf:
        selection_regions = get_nonres_vbf_selection_regions(args.year, **cutargs)
    else:
        selection_regions = get_nonres_selection_regions(args.year, args.nonres_regions, **cutargs)

    return cutstr, template_dir, selection_regions


# adds all necessary columns to dataframes from events_dict
def _add_nonres_columns(df, bb_mask, vbf_vars=False, ptlabel="", mlabel=""):
    """Variables needed for ggF and/or VBF BDTs"""
    # import time

    import vector

    # start = time.time()

    bbJet = utils.make_vector(df, "bbFatJet", bb_mask, ptlabel=ptlabel, mlabel=mlabel)
    VVJet = utils.make_vector(df, "VVFatJet", bb_mask, ptlabel=ptlabel, mlabel=mlabel)
    Dijet = bbJet + VVJet

    # print(f"Time to make vectors: {time.time() - start:.2f}")

    if f"DijetMass{ptlabel}{mlabel}" not in df.columns:
        df[f"DijetMass{ptlabel}{mlabel}"] = Dijet.mass
    df[f"DijetPt{ptlabel}{mlabel}"] = Dijet.pt
    df[f"VVFatJetPtOverbbFatJetPt{ptlabel}{mlabel}"] = VVJet.pt / bbJet.pt
    df[f"VVFatJetPtOverDijetPt{ptlabel}{mlabel}"] = VVJet.pt / df[f"DijetPt{ptlabel}{mlabel}"]

    # print(f"AK8 jet vars: {time.time() - start:.2f}")

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

    if "DijetdEta" not in df.columns:
        df["DijetdEta"] = np.abs(bbJet.eta - VVJet.eta)
    if "DijetdPhi" not in df.columns:
        df["DijetdPhi"] = np.abs(bbJet.deltaphi(VVJet))

    if f"vbf_Mass_jj{ptlabel}" not in df.columns:
        df[f"vbf_Mass_jj{ptlabel}"] = jj.M
        # df[f"vbf_Mass_jj{ptlabel}"] = np.nan_to_num(jj.M)
    if "vbf_dEta_jj" not in df.columns:
        df["vbf_dEta_jj"] = np.abs(vbf1.eta - vbf2.eta)
        # df["vbf_dEta_jj"] = np.nan_to_num(np.abs(vbf1.eta - vbf2.eta))

    # print(f"VBF jet vars: {time.time() - start:.2f}")

    if not vbf_vars:
        return

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

    if "DijetdEta" not in df.columns:
        df["DijetdEta"] = np.abs(bbJet.eta - VVJet.eta)
    if "DijetdPhi" not in df.columns:
        df["DijetdPhi"] = np.abs(bbJet.deltaphi(VVJet))

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


def _add_res_columns(df, bb_mask, ptlabel="", mlabel=""):

    bbJet = utils.make_vector(df, "bbFatJet", bb_mask, ptlabel=ptlabel, mlabel=mlabel)
    VVJet = utils.make_vector(df, "VVFatJet", bb_mask, ptlabel=ptlabel, mlabel=mlabel)
    Dijet = bbJet + VVJet

    if f"DijetMass{ptlabel}{mlabel}" not in df.columns:
        df[f"DijetMass{ptlabel}{mlabel}"] = Dijet.mass


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

    # print("BDT Sample Order: ", BDT_sample_order)
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
    hem_cleaning: bool = False,
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
    # remove empty parquets, otherwise read_parquet fails
    utils.remove_empty_parquets(data_dir)
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


def _check_load_systematics(systs_file: str, year: str, override_systs: bool):
    if systs_file.exists() and not override_systs:
        # print("Loading existing systematics")
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
        print(f"Loading signals from {d}")
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

    bg_samples = deepcopy(samples)

    if data_key in samples:
        bg_samples.pop(data_key)
        data_samples = {data_key: samples[data_key]}

    for d in args.bg_data_dirs:
        print(f"Loading backgrounds from {d}")
        events_dict = {
            **events_dict,
            **load_samples(
                d,
                bg_samples,
                args.year,
                filters,
                hem_cleaning=args.hem_cleaning,
                variations=variations,
            ),
        }

    if args.data_dir and data_key in samples:
        print(f"Loading data from {args.data_dir}")
        events_dict = {
            **events_dict,
            **load_samples(
                args.data_dir,
                data_samples,
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

    for sample, events in events_dict.items():
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


def qcd_sf(
    events_dict: dict[str, pd.DataFrame], cutflow: pd.DataFrame, weight_key: str = "finalWeight"
):
    """Applies a QCD scale factor."""
    if qcd_key not in events_dict or data_key not in events_dict:
        return

    trig_yields = cutflow.iloc[:, -1]
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


def apply_weights(
    events_dict: dict[str, pd.DataFrame],
    year: str,
    cutflow: pd.DataFrame = None,
    trigger_effs: bool = True,
    do_qcd_sf: bool = True,
    weight_key: str = "finalWeight",
):
    """
    Applies (1) 2D trigger scale factors, (2) QCD scale facotr.

    Args:
        cutflow (pd.DataFrame): cutflow to which to add yields after scale factors.
        weight_key (str): column in which to store scaled weights in. Defaults to "finalWeight".

    """
    if trigger_effs:
        apply_trigger_weights(events_dict, year, cutflow, weight_key=weight_key)

    if do_qcd_sf:
        qcd_sf(events_dict, cutflow, weight_key=weight_key)


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
    resonant: bool = False,
    nonres_vars: bool = True,
    vbf_vars: bool = False,
    do_jshifts: bool = True,
):
    """Add Dijet variables"""
    import time

    start = time.time()

    for sample, events in events_dict.items():
        print(f"Deriving variables for {sample} {time.time() - start:.2f}s")

        bb_mask = bb_masks[sample]

        fargs = [events, bb_mask]
        if resonant:
            dfunc = _add_res_columns
        elif nonres_vars:
            dfunc = _add_nonres_columns
            fargs.append(vbf_vars)
        else:
            continue

        dfunc(*fargs)

        if sample == data_key or not do_jshifts:
            continue

        for var in jec_shifts:
            dfunc(*fargs, ptlabel=f"_{var}")

        for var in jmsr_shifts:
            dfunc(*fargs, mlabel=f"_{var}")


def _add_bdt_scores(
    events: pd.DataFrame,
    sample_bdt_preds: np.ndarray,
    multiclass: bool,
    multisig: bool,
    all_outs: bool = True,
    jshift: str = "",
):
    if jshift != "":
        jshift = "_" + jshift

    if not multiclass:
        events[f"BDTScore{jshift}"] = sample_bdt_preds
    else:
        if multisig:
            num_sigs = 2
            bg_tot = np.sum(sample_bdt_preds[:, num_sigs:], axis=1)
            ggf_score = sample_bdt_preds[:, 0]
            vbf_score = sample_bdt_preds[:, 1]

            events[f"BDTScore{jshift}"] = ggf_score / (ggf_score + bg_tot)
            events[f"BDTScoreVBF{jshift}"] = vbf_score / (vbf_score + bg_tot)

            if all_outs:
                events[f"BDTScoreQCD{jshift}"] = sample_bdt_preds[:, num_sigs]
                events[f"BDTScoreTT{jshift}"] = sample_bdt_preds[:, num_sigs + 1]
                events[f"BDTScoreZjets{jshift}"] = sample_bdt_preds[:, num_sigs + 2]
        else:
            events[f"BDTScore{jshift}"] = sample_bdt_preds[:, 0]
            if all_outs:
                events[f"BDTScoreQCD{jshift}"] = sample_bdt_preds[:, 1]
                events[f"BDTScoreTT{jshift}"] = sample_bdt_preds[:, 2]
                events[f"BDTScoreZJets{jshift}"] = 1 - np.sum(sample_bdt_preds, axis=1)


def load_bdt_preds(
    events_dict: dict[str, pd.DataFrame],
    year: str,
    bdt_preds_dir: Path,
    jec_jmsr_shifts: bool = False,
    all_outs: bool = False,
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
    multisig = multiclass and bdt_preds.shape[1] > 3

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

            sample_bdt_preds = bdt_preds[i : i + num_events]
            _add_bdt_scores(events, sample_bdt_preds, multiclass, multisig, all_outs)

            if jec_jmsr_shifts and sample != data_key:
                for jshift in jec_shifts + jmsr_shifts:
                    sample_bdt_preds = shift_preds[jshift][i : i + num_events]
                    _add_bdt_scores(
                        events, sample_bdt_preds, multiclass, multisig, all_outs, jshift=jshift
                    )

        i += num_events

    assert i == len(bdt_preds), f"# events {i} != # of BDT preds {len(bdt_preds)}"


def _lpsfs(args, filters, scan, scan_cuts, scan_wps, sig_keys, sig_samples):
    """Get LP SFs for all scanned WPs together so signals are only loaded once."""

    # load pre-calculated systematics and those for different years if saved already
    systs_file = args.template_dir / "systematics.json"
    systematics = _check_load_systematics(systs_file, args.year, args.override_systs)

    lpsf_regions = []
    prelpsf_region = None

    for wps in scan_wps:  # if not scanning, this will just be a single WP
        cutstr, _, selection_regions = _get_scan_regions(args, scan, scan_cuts, wps)
        for region in selection_regions.values():
            if region.lpsf:
                tregion = deepcopy(region)
                if scan:
                    tregion.lpsf_region += "_" + cutstr
                lpsf_regions.append(tregion)

            if region.prelpsf:
                prelpsf_region = region

    lpsfs(
        sig_keys,
        prelpsf_region,
        lpsf_regions,
        systematics,
        sig_samples=sig_samples,
        filters=filters,
        all_years=args.lp_sf_all_years,
        year=args.year,
        bdt_preds_dir=args.bdt_preds_dir,
        template_dir=args.template_dir,
        systs_file=systs_file,
        data_dir=args.signal_data_dirs[0],
    )

    # save the LP SFs in the systematics files for each scanned point
    if scan:
        for wps in scan_wps:
            cutstr, template_dir, selection_regions = _get_scan_regions(args, scan, scan_cuts, wps)
            systs_file = template_dir / "systematics.json"
            wsysts = _check_load_systematics(systs_file, args.year, args.override_systs)

            for region in selection_regions.values():
                if region.lpsf:
                    wsysts[region.lpsf_region] = systematics[region.lpsf_region + "_" + cutstr]

            with systs_file.open("w") as f:
                json.dump(wsysts, f, indent=4)


# @profile
def _get_signal_all_years(
    sig_key: str,
    data_dir: Path,
    samples: dict,
    filters: list,
    prelpsf_region: Region,
    bdt_preds_dir: Path = None,
    year: str = None,
):
    """Load signal samples for all years and combine them for LP SF measurements."""
    print(f"Loading {sig_key} for all years")
    events_all = []

    # load the minimum necessary columns to save time
    # (column name, number of subcolumns)
    load_columns = [
        ("weight", 1),
        ("weight_noTrigEffs", 1),
        ("ak8FatJetPt", 2),
        # ("ak8FatJetMsd", 2),
        ("ak8FatJetHVV", 2),
        ("ak8FatJetHVVNumProngs", 1),
        ("ak8FatJetParticleNetMD_Txbb", 2),
        ("VVFatJetParTMD_THWWvsT", 1),
        ("nGoodElectronsHbb", 1),
        ("nGoodMuonsHbb", 1),
    ]

    load_columns += hh_vars.lp_sf_vars

    # reformat into ("column name", "idx") format for reading multiindex columns
    column_labels = []
    for key, num_columns in load_columns:
        for i in range(num_columns):
            column_labels.append(f"('{key}', '{i}')")

    run_years = years if year is None else [year]

    for year in run_years:
        events_dict = load_samples(
            data_dir,
            {sig_key: samples[sig_key]},
            year,
            filters,
            column_labels,
            variations=False,
        )

        cutflow = pd.DataFrame(index=[sig_key])  # save cutflow as pandas table
        utils.add_to_cutflow(events_dict, "Pre-selection", "finalWeight", cutflow)
        # print weighted sample yields
        # print(f"{np.sum(events_dict[sig_key]['finalWeight'].to_numpy()):.2f} Events")

        bb_masks = bb_VV_assignment(events_dict)
        derive_variables(events_dict, bb_masks, nonres_vars=False, do_jshifts=False)
        # events_dict[sig_key] = postprocess_lpsfs(events_dict[sig_key])

        if bdt_preds_dir is not None:
            load_bdt_preds(events_dict, year, bdt_preds_dir, all_outs=False)

        sel, cutflow = utils.make_selection(
            prelpsf_region.cuts,
            events_dict,
            bb_masks,
            weight_key="finalWeight",
            prev_cutflow=cutflow,
        )
        print(year, cutflow)

        # drop columns for memory
        events_dict[sig_key] = events_dict[sig_key].drop(
            [
                ("nGoodElectronsHbb", 0),
                ("nGoodMuonsHbb", 0),
                ("ak8FatJetPt", 0),
                ("ak8FatJetPt", 1),
            ],
            axis=1,
            # inplace=True,
        )
        events_all.append(events_dict[sig_key][sel[sig_key]])
        # sels_all.append(sel[sig_key])

    events_dict = {sig_key: pd.concat(events_all, axis=0)}
    events_dict[sig_key] = postprocess_lpsfs(events_dict[sig_key])
    # sel = np.concatenate(sels_all, axis=0)

    return events_dict  # get_lpsf(events, sel)


def _check_measure_lpsfs(systematics, sig_key, lp_selection_regions):
    """Check if any of the LP SFs need to be measured for the given signal."""
    measure_lpsf = False
    for lp_region in lp_selection_regions:
        assert lp_region.lpsf, str(lp_region) + " Not an LP SF region"
        rlabel = lp_region.lpsf_region

        # check if LP SF is already in systematics
        if (
            rlabel not in systematics
            or sig_key not in systematics[rlabel]
            or "lp_sf" not in systematics[rlabel][sig_key]
        ):
            measure_lpsf = True
            break

    return measure_lpsf


def lpsfs(
    sig_keys: list[str],
    prelpsf_region: Region,
    lp_selection_regions: Region | list[Region],
    systematics: dict,
    sig_samples: dict[str, str] = None,
    filters: list = [],  # noqa: B006
    all_years: bool = False,
    year: str = None,
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

    if isinstance(lp_selection_regions, Region):
        lp_selection_regions = [lp_selection_regions]

    for sig_key in sig_keys:
        if not _check_measure_lpsfs(systematics, sig_key, lp_selection_regions):
            continue

        print(f"\nGetting LP SFs for {sig_key}")

        assert data_dir is not None, "Need data_dir to load signals"
        assert filters != [], "Need to specify filters"
        assert sig_samples is not None, "Need sig_samples to load signals"

        # SFs are correlated across all years so needs to be calculated with full dataset
        if all_years:
            events_dict = _get_signal_all_years(
                sig_key, data_dir, sig_samples, filters, prelpsf_region, bdt_preds_dir
            )
            bb_masks = bb_VV_assignment(events_dict)

        # ONLY FOR TESTING, can do just for a single year
        else:
            events_dict = _get_signal_all_years(
                sig_key, data_dir, sig_samples, filters, prelpsf_region, bdt_preds_dir, year=year
            )
            bb_masks = bb_VV_assignment(events_dict)

            # continue

        for lp_region in lp_selection_regions:
            rlabel = lp_region.lpsf_region
            print(rlabel)

            if rlabel not in systematics:
                systematics[rlabel] = {}

            rsysts = systematics[rlabel]

            if sig_key not in rsysts:
                rsysts[sig_key] = {}

            sel, _ = utils.make_selection(lp_region.cuts, events_dict, bb_masks)
            lp_sf, unc, uncs_sym, uncs_asym = get_lpsf(events_dict[sig_key], sel[sig_key])

            print(
                f"LP Scale Factor for {sig_key} in {rlabel} region: {lp_sf:.2f} +{unc[0]:.2f}-{unc[1]:.2f}"
            )

            rsysts[sig_key]["lp_sf"] = lp_sf
            rsysts[sig_key]["lp_sf_unc_up"] = unc[0] / lp_sf
            rsysts[sig_key]["lp_sf_unc_down"] = unc[1] / lp_sf
            rsysts[sig_key]["lp_sf_uncs_sym"] = uncs_sym
            rsysts[sig_key]["lp_sf_uncs_asym"] = uncs_asym

            if systs_file is not None:
                with systs_file.open("w") as f:
                    json.dump(systematics, f, indent=4)

    if template_dir is not None:
        sf_table = OrderedDict()  # format SFs for each sig key in a table
        for lp_region in lp_selection_regions:
            rlabel = lp_region.lpsf_region
            rsysts = systematics[rlabel]
            for sig_key in sig_keys:
                systs = rsysts[sig_key]
                sf_table[sig_key] = {
                    "SF": f"{systs['lp_sf']:.2f} +{systs['lp_sf'] * systs['lp_sf_unc_up']:.2f}-{systs['lp_sf'] * systs['lp_sf_unc_down']:.2f}",
                    **systs["lp_sf_uncs_sym"],
                }
                for key in systs["lp_sf_uncs_asym"]["up"]:
                    sf_table[sig_key][
                        key
                    ] = f"+{systs['lp_sf_uncs_asym']['up'][key]:.2f}-{systs['lp_sf_uncs_asym']['down'][key]:.2f}"

            print(f"\nLP Scale Factors in {rlabel}:\n", pd.DataFrame(sf_table).T)
            pd.DataFrame(sf_table).T.to_csv(f"{template_dir}/lpsfs_{rlabel}.csv")


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
    title: str = None,
    sig_splits: list[list[str]] = None,
    bg_keys: list[str] = bg_keys,
    selection: dict[str, np.ndarray] = None,
    sig_scale_dict: dict[str, float] = None,
    combine_pdf: bool = True,
    HEM2d: bool = False,
    plot_ratio: bool = True,
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
        plot_ratio: whether to plot the data/MC ratio.
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
                    title=title,
                    sig_scale_dict=tsig_scale_dict if not log else None,
                    plot_significance=plot_significance,
                    significance_dir=shape_var.significance_dir,
                    show=show,
                    log=log,
                    ylim=ylim if not log else 1e15,
                    plot_ratio=plot_ratio,
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
    bdtvars = ["", "VBF"]
    plot_keys = ["QCD", "TT", "Z+Jets", "HHbbVV", "qqHH_CV_1_C2V_0_kl_1_HHbbVV"]

    shape_var = ShapeVar(
        var="bbFatJetParticleNetMass", label=r"$m^{bb}_{reg}$ [GeV]", bins=[20, 50, 250]
    )

    for var in bdtvars:
        for key in plot_keys:
            ed_key = {key: events_dict[key]}
            bbm_key = {key: bb_masks[key]}

            plotting.cutsLinePlot(
                ed_key,
                shape_var,
                key,
                f"BDTScore{var}",
                r"$BDT_{ggF}$" if var == "" else r"$BDT_{VBF}$",
                cuts,
                year,
                weight_key,
                bb_masks=bbm_key,
                plot_dir=plot_dir,
                name=f"{year}_BDT{var}Cuts_{shape_var.var}_{key}",
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
    blind: bool = True,
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
    import time

    start = time.time()

    if weight_shifts is None:
        weight_shifts = {}
    do_jshift = jshift != ""
    jlabel = "" if not do_jshift else "_" + jshift
    templates = {}

    for rname, region in selection_regions.items():
        if region.lpsf:
            continue

        pass_region = rname.startswith("pass")

        print(f"{rname} Region: {time.time() - start:.2f}")

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
        print(f"Selection: {time.time() - start:.2f}")

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

            if region.signal:
                # scale all signal weights by LP SF (if not doing a j shift)
                if lpsfs:
                    scale_wkeys = (
                        utils.get_all_weights(sig_events[sig_key])
                        if not do_jshift
                        else [weight_key]
                    )
                    for wkey in scale_wkeys:
                        sig_events[sig_key][wkey] *= systematics[rname][sig_key]["lp_sf"]

                # print(f"LP SFs: {time.time() - start:.2f}")
                corrections.apply_txbb_sfs(
                    sig_events[sig_key], sig_bb_mask, year, weight_key, do_shifts=not do_jshift
                )

                # print(f"Txbb SFs: {time.time() - start:.2f}")

        print(f"Tagger SFs: {time.time() - start:.2f}")

        # if not do_jshift:
        #     print("\nCutflow:\n", cf)

        # set up samples
        hist_samples = list(events_dict.keys())

        if not do_jshift:
            # add all weight-based variations to histogram axis
            for shift in ["down", "up"]:
                if region.signal:
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

            # breakpoint()
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

        print(f"Histograms: {time.time() - start:.2f}")

        if region.signal and blind:
            # blind signal mass windows in pass region in data
            for i, shape_var in enumerate(shape_vars):
                if shape_var.blind_window is not None:
                    utils.blindBins(h, shape_var.blind_window, data_key, axis=i)

        if region.signal and not do_jshift:
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
            print(f"Plotting templates: {time.time() - start:.2f}")
            if plot_sig_keys is None:
                plot_sig_keys = sig_keys

            if sig_scale_dict is None:
                sig_scale_dict = {
                    **{skey: 1 for skey in nonres_sig_keys if skey in plot_sig_keys},
                    **{skey: 10 for skey in res_sig_keys if skey in plot_sig_keys},
                }

            title = (
                f"{region.label} Region Pre-Fit Shapes"
                if not do_jshift
                else f"{region.label} Region {jshift} Shapes"
            )

            if sig_splits is None:
                sig_splits = [plot_sig_keys]

            # don't plot qcd in the pass regions
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
                            if region.signal
                            else None
                        ),
                        "show": show,
                        "year": year,
                        "ylim": pass_ylim if pass_region else fail_ylim,
                        "plot_data": not (rname == "pass" and blind_pass),
                        "divide_bin_width": args.resonant,
                    }

                    plot_name = (
                        f"{plot_dir}/"
                        f"{'jshifts/' if do_jshift else ''}"
                        f"{split_str}{rname}_region_{shape_var.var}"
                    )

                    plotting.ratioHistPlot(
                        **plot_params,
                        bg_keys=p_bg_keys,
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
                                bg_keys=p_bg_keys,
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

                        if region.signal:
                            plotting.ratioHistPlot(
                                **plot_params,
                                bg_keys=p_bg_keys,
                                sig_err="txbb",
                                title=rf"{region.label} Region $T_{{Xbb}}$ Shapes",
                                name=f"{plot_name}_txbb.pdf",
                            )

    return templates


def save_templates(
    templates: dict[str, Hist],
    template_file: Path,
    blind: bool,
    resonant: bool,
    shape_vars: list[ShapeVar],
):
    """Creates blinded copies of each region's templates and saves a pickle of the templates"""

    if not resonant and blind:
        from copy import deepcopy

        blind_window = shape_vars[0].blind_window

        for label, template in list(templates.items()):
            blinded_template = deepcopy(template)
            utils.blindBins(blinded_template, blind_window)
            templates[f"{label}Blinded"] = blinded_template

    with template_file.open("wb") as f:
        pickle.dump(templates, f)

    print("Saved templates to", template_file)


def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default=None,
        help="path to skimmed parquet",
        type=str,
    )

    parser.add_argument(
        "--bg-data-dirs",
        default=[],
        help="path to skimmed background parquets, if different from other data",
        nargs="*",
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
        required=True,
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
    parser.add_argument(
        "--nonres-regions",
        help="Which nonresonant categories to make templates for.",
        default="all",
        choices=["all", "ggf", "ggf_no_vbf", "vbf"],
        type=str,
    )
    add_bool_arg(parser, "vbf", "old!! non-resonant VBF category", default=False)
    add_bool_arg(parser, "control-plots", "make control plots", default=False)
    add_bool_arg(
        parser,
        "mass-plots",
        "make mass comparison plots (filters will automatically be turned off)",
        default=False,
    )
    add_bool_arg(parser, "blinded", "blind the data in the Higgs mass window", default=True)
    add_bool_arg(parser, "bdt-plots", "make bdt sculpting plots", default=False)
    add_bool_arg(parser, "lpsfs", "measure LP SFs for given WPs", default=False)
    add_bool_arg(parser, "templates", "save m_bb templates using bdt cut", default=False)
    add_bool_arg(
        parser, "overwrite-template", "if template file already exists, overwrite it", default=False
    )
    add_bool_arg(parser, "do-jshifts", "Do JEC/JMC variations", default=True)
    add_bool_arg(parser, "plot-shifts", "Plot systematic variations as well", default=False)
    add_bool_arg(parser, "lp-sf-all-years", "Calculate one LP SF for all run 2", default=True)
    add_bool_arg(
        parser, "override-systs", "Override saved systematics file if it exists", default=False
    )

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
        "--nonres-ggf-txbb-wp",
        help="Txbb WP for ggF signal region. If multiple arguments, will make templates for each.",
        default=["MP"],
        choices=["LP", "MP", "HP"],
        nargs="*",
        type=str,
    )

    parser.add_argument(
        "--nonres-ggf-bdt-wp",
        help="BDT WP for ggF signal region. If multiple arguments, will make templates for each.",
        default=[0.995],
        nargs="*",
        type=float,
    )

    parser.add_argument(
        "--nonres-vbf-txbb-wp",
        help="Txbb WP for VBF signal region. If multiple arguments, will make templates for each.",
        default=["HP"],
        choices=["LP", "MP", "HP"],
        nargs="*",
        type=str,
    )

    parser.add_argument(
        "--nonres-vbf-bdt-wp",
        help="Txbb WP for VBFsignal region. If multiple arguments, will make templates for each.",
        default=[0.999],
        nargs="*",
        type=float,
    )

    parser.add_argument(
        "--nonres-vbf-thww-wp",
        help="THWWvsT WP for VBF signal region. If multiple arguments, will make templates for each.",
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
        default=[0.6],
        nargs="*",
        type=float,
    )

    parser.add_argument(
        "--res-leading-pt",
        help="pT cut for leading AK8 jet (resonant only)",
        default=[400],
        nargs="*",
        type=float,
    )

    parser.add_argument(
        "--res-subleading-pt",
        help="pT cut for sub-leading AK8 jet (resonant only)",
        default=[350],
        nargs="*",
        type=float,
    )

    parser.add_argument(
        "--lepton-veto",
        help="lepton vetoes: None, Hbb, or HH",
        default=["Hbb"],
        nargs="*",
        type=str,
    )

    args = parser.parse_args()

    if (args.templates or args.lpsfs) and args.template_dir == "":
        print("Need to set --template-dir if making templates or measuring LP SFs. Exiting.")
        sys.exit()

    if not args.signal_data_dirs and args.data_dir:
        args.signal_data_dirs = [args.data_dir]

    if not args.bg_data_dirs and args.data_dir:
        args.bg_data_dirs = [args.data_dir]

    if args.bdt_preds_dir != "" and args.bdt_preds_dir is not None:
        args.bdt_preds_dir = Path(args.bdt_preds_dir)

    if args.bdt_preds_dir == "" and not args.resonant:
        if args.data_dir is None:
            args.bdt_preds_dir = Path(f"{args.signal_data_dirs[0]}/inferences/")
        else:
            args.bdt_preds_dir = Path(f"{args.data_dir}/inferences/")
    elif args.resonant:
        args.bdt_preds_dir = None

    if args.hem_cleaning is None:
        # can't do HEM cleaning for non-resonant until BDT is re-inferenced
        args.hem_cleaning = bool(args.resonant or args.vbf)

    if args.mass_plots:
        args.control_plots = True
        args.filters = False

    print(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
