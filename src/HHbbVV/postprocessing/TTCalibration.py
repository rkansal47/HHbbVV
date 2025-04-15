"""
Plots for LP SF calibration on top jets. Adapted from TopAnalysis.ipynb.

Author: Raghav Kansal
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from copy import deepcopy
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import plotting
import postprocessing
import utils
from hist import Hist
from pandas.errors import SettingWithCopyWarning

# ignore these because they don't seem to apply
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

bg_samples = {
    "QCD": "QCD",
    "TTbar": ["TTTo2L2Nu", "TTToHadronic"],
    "W+Jets": "WJets",
    "Diboson": ["WW", "WZ", "ZZ"],
    "Data": "SingleMuon",
}

sig_samples = {"Top": ["TTToSemiLeptonic", "ST"]}

samples = {**bg_samples, **sig_samples}

top_matched_key, top_wmatched_key, top_unmatched_key = (
    "Top Matched",
    "Top W Matched",
    "Top Unmatched",
)


plot_samples = [
    "QCD",
    "Diboson",
    # "Single Top",
    "W+Jets",
    top_unmatched_key,
    top_wmatched_key,
    top_matched_key,
]

bg_colours = {
    "QCD": "lightblue",
    "Single Top": "darkblue",
    top_unmatched_key: "brown",
    top_wmatched_key: "red",
    top_matched_key: "orange",
    "W+Jets": "beige",
    "Diboson": "darkpurple",
}


# {var: (bins, label)}
plot_vars = {
    # "ak8FatJetMsd": ([20, 125, 225], r"$m_{SD}$ (GeV)"),
    # "ak8FatJetParticleNetMass": ([30, 50, 200], r"$m_{reg}$ (GeV)"),
    # "ak8FatJetPt": ([30, 0, 1200], r"$p_T$ (GeV)"),
    # "MET_pt": ([30, 0, 200], r"MET (GeV)"),
    # "ak8FatJetnPFCands": ([20, 0, 120], r"# of PF Candidates"),
    # "ak8FatJetParticleNet_Th4q": ([20, 0.6, 1], r"ParticleNet $T_{H4q}$ Non-MD"),
    "ak8FatJetParTMD_THWW4q": ([20, 0.2, 1], r"$ParT\ T^{No Top}_{HWW}$"),
    "ak8FatJetdeepTagMD_WHvsQCD": ([20, 0.2, 1], r"DeepAK8-MD score (No Top)"),
    # "ak8FatJetdeepTagMD_H4qvsQCD": ([20, 0.2, 1], r"DeepAK8 (QCD veto)"),
    # "ak8FatJetdeepTagMD_WvsQCD": ([20, 0.2, 1], r"DeepAK8 WvsQCD (QCD veto)"),
    # "tau21": ([20, 0.04, 0.8], r"$\tau_{21}$"),
    # "tau32": ([20, 0.2, 1], r"$\tau_{32}$"),
    # "tau43": ([20, 0.42, 1], r"$\tau_{43}$"),
    # "tau42": ([20, 0, 1], r"$\tau_{42}$"),
    # "tau41": ([20, 0, 1], r"$\tau_{41}$"),
}

CLIP = 5.0


def chisquare(mc, data):
    return np.sum(np.square(data - mc) / data)


def main(args):
    cutflow = pd.DataFrame(index=list(samples.keys()))
    events_dict = postprocessing.load_samples(
        args.data_dir, bg_samples, args.year, hem_cleaning=False
    )
    events_dict |= postprocessing.load_samples(
        args.signal_data_dirs[0], sig_samples, args.year, hem_cleaning=False
    )
    utils.add_to_cutflow(events_dict, "Selection", "weight", cutflow)

    derive_variables(events_dict)
    normalize_events(events_dict)
    utils.add_to_cutflow(events_dict, "Scale", "weight", cutflow)

    fatjet_selection(events_dict)
    utils.add_to_cutflow(events_dict, "FatJetSelection", "weight", cutflow)

    top_matching(events_dict)

    events = events_dict[top_matched_key]
    lp_sf_processing(events)
    lp_sf_normalization(events)

    # TODO: plotting and SF analysis


def derive_variables(events_dict):
    for _sample, events in events_dict.items():
        wq = events["ak8FatJetdeepTagMD_WvsQCD"].to_numpy()
        hq = events["ak8FatJetdeepTagMD_H4qvsQCD"].to_numpy()
        wdivq = wq / (1 - wq)
        hdivq = hq / (1 - hq)
        whvsq = (wdivq + hdivq) / (1 + wdivq + hdivq)
        events[("ak8FatJetdeepTagMD_WHvsQCD", 0)] = whvsq


def normalize_events(events_dict):
    # normalizations are off
    scale_samples = ["Top", "TTbar", "W+Jets", "QCD"]

    total_scale = 0
    total_noscale = 0
    for sample, events in events_dict.items():
        if sample in scale_samples:
            total_scale += events["weight"].sum().to_numpy()[0]
        elif sample != "Data":
            total_noscale += events["weight"].sum().to_numpy()[0]

    print(f"Total MC: {total_scale + total_noscale}")

    sf = (events_dict["Data"]["weight"].sum().to_numpy()[0] - total_noscale) / total_scale
    for sample, events in events_dict.items():
        if sample in scale_samples:
            events["weight"] *= sf

    total = 0
    for sample, events in events_dict.items():
        if sample != "Data":
            total += events["weight"].sum().to_numpy()[0]

    print(f"New Total MC: {total}")


def fatjet_selection(events_dict):
    # cuts to match Top region definition from JME-23-001 and its predecessors
    for key in events_dict:
        events_dict[key] = events_dict[key][events_dict[key]["ak8FatJetPt"][0] >= 500]
        events_dict[key] = events_dict[key][events_dict[key]["ak8FatJetMsd"][0] >= 150]
        events_dict[key] = events_dict[key][events_dict[key]["ak8FatJetMsd"][0] <= 225]


def top_matching(events_dict):
    events_dict[top_matched_key] = events_dict["Top"].loc[events_dict["Top"]["top_matched"][0] == 1]
    events_dict[top_wmatched_key] = events_dict["Top"].loc[events_dict["Top"]["w_matched"][0] == 1]
    events_dict[top_unmatched_key] = pd.concat(
        [
            events_dict["TTbar"],
            events_dict["Top"].loc[events_dict["Top"]["unmatched"][0] == 1],
        ]
    )


def lp_sf_processing(events):
    """Process LP SFs"""
    np_sfs = events["lp_sf_lnN"][0].to_numpy()
    dist_sfs = events["lp_sf_dist"][0].to_numpy()
    # remove low stats values
    dist_sfs[dist_sfs > CLIP] = 1.0
    dist_sfs[dist_sfs < 1.0 / CLIP] = 1.0
    events["lp_sf_dist_up"] = np_sfs * dist_sfs
    events["lp_sf_dist_down"] = np_sfs / dist_sfs

    # cases where we recluster with 1 more prong for matching unc.
    up_prong_rc = (
        (events["lp_sf_outside_boundary_quarks"][0] > 0)
        | (events["lp_sf_double_matched_event"][0] > 0)
        | (events["lp_sf_unmatched_quarks"][0] > 0)
    ).to_numpy()

    # cases where we recluster with 1 fewer prong for matching unc.
    down_prong_rc = (
        (events["lp_sf_inside_boundary_quarks"][0] > 0)
        | (events["lp_sf_double_matched_event"][0] > 0)
        | (events["lp_sf_unmatched_quarks"][0] > 0)
    ).to_numpy()

    # cases where there are still unmatched quarks after reclustering with +/- 1 prong
    rc_unmatched = events["lp_sf_rc_unmatched_quarks"][0] > 0

    for shift, prong_rc in [("up", up_prong_rc), ("down", down_prong_rc)]:
        np_sfs = events["lp_sf_lnN"][0].to_numpy()
        np_sfs[prong_rc] = events[f"lp_sf_np_{shift}"][0][prong_rc]
        events.loc[:, (f"lp_sf_np_{shift}", 0)] = np_sfs

        # np_sfs = np.nan_to_num(np.clip(np_sfs, 1.0 / CLIP, CLIP))
        # events.loc[:, (f"lp_sf_np_{shift}", 0)] = np_sfs / np.mean(np_sfs, axis=0)

    for shift in ["up", "down"]:
        np_sfs = events["lp_sf_lnN"][0].to_numpy()
        np_sfs[rc_unmatched] = CLIP if shift == "up" else 1.0 / CLIP
        events[f"lp_sf_unmatched_{shift}"] = np_sfs

        # np_sfs = np.nan_to_num(np.clip(np_sfs, 1.0 / CLIP, CLIP))
        # events[f"lp_sf_unmatched_{shift}"] = np_sfs / np.mean(np_sfs, axis=0)


def lp_sf_normalization(events):
    # normalize scale factors to average to 1
    for key in [
        # "lp_sf",
        "lp_sf_lnN",
        "lp_sf_sys_down",
        "lp_sf_sys_up",
        "lp_sf_dist_down",
        "lp_sf_dist_up",
        "lp_sf_np_up",
        "lp_sf_np_down",
        "lp_sf_unmatched_up",
        "lp_sf_unmatched_down",
        "lp_sf_pt_extrap_vars",
        "lp_sfs_bl_ratio",
    ]:
        # cut off at 5
        events.loc[:, key] = np.clip(events.loc[:, key].to_numpy(), 1.0 / 5.0, 5.0)

        if key == "lp_sfs_bl_ratio":
            mean_lp_sfs = np.mean(
                np.nan_to_num(events[key][0] * events["lp_sf_lnN"][0]),
                axis=0,
            )
        else:
            mean_lp_sfs = np.mean(np.nan_to_num(events[key]), axis=0)

        events.loc[:, key] = np.nan_to_num(events.loc[:, key]) / mean_lp_sfs


def plot_pre_hists(events_dict, plot_dir, year, show=False) -> dict[str, Hist]:
    pre_hists = {}

    for var, (bins, label) in plot_vars.items():
        if var not in pre_hists:
            pre_hists[var] = utils.singleVarHistNoMask(
                events_dict, var, bins, label, weight_key="weight"
            )

    for var, var_hist in pre_hists.items():
        name = f"{plot_dir}/pre_{var}.pdf"
        plotting.ratioLinePlot(
            var_hist,
            plot_samples,
            year,
            name=name,
            bg_colours=bg_colours,
            # bg_order=plot_samples,
            # ratio_ylims=[0.6, 1.3],
            show=show,
        )

    with Path(f"{plot_dir}/pre_hists.pkl").open("wb") as f:
        pickle.dump(pre_hists, f)

    return pre_hists


def plot_post_hists(events_dict, pre_hists, plot_dir, year, show=False) -> tuple[dict, dict, dict]:
    post_lnN_hists = {}
    post_lnN_hists_err = {}
    uncs_lnN_dict = {}

    events = events_dict[top_matched_key]

    for var, (bins, _label) in plot_vars.items():
        if var not in post_lnN_hists:
            toy_hists = []
            for i in range(events["lp_sf_lnN"].shape[1]):
                toy_hists.append(
                    np.histogram(
                        events[var][0].to_numpy().squeeze(),
                        np.linspace(*bins[1:], bins[0] + 1),
                        weights=events["weight"][0].to_numpy() * events["lp_sf_lnN"][i].to_numpy(),
                    )[0]
                )

            sys_up_down = []
            for key in ["lp_sf_sys_up", "lp_sf_sys_down"]:
                sys_up_down.append(
                    np.histogram(
                        events[var][0].to_numpy().squeeze(),
                        np.linspace(*bins[1:], bins[0] + 1),
                        weights=events["weight"][0].to_numpy() * events[key][0].to_numpy(),
                    )[0]
                )

            np_up_down = []
            for key in ["lp_sf_np_up", "lp_sf_np_down"]:
                np_up_down.append(
                    np.histogram(
                        events[var][0].to_numpy().squeeze(),
                        np.linspace(*bins[1:], bins[0] + 1),
                        weights=events["weight"][0].to_numpy()
                        * np.nan_to_num(events[key][0].to_numpy()),
                    )[0]
                )

            dist_up_down = []
            for key in ["lp_sf_dist_up", "lp_sf_dist_down"]:
                dist_up_down.append(
                    np.histogram(
                        events[var][0].to_numpy().squeeze(),
                        np.linspace(*bins[1:], bins[0] + 1),
                        weights=events["weight"][0].to_numpy()
                        * np.nan_to_num(events[key].to_numpy()),
                    )[0]
                )

            um_up_down = []
            for key in ["lp_sf_unmatched_up", "lp_sf_unmatched_down"]:
                um_up_down.append(
                    np.histogram(
                        events[var][0].to_numpy().squeeze(),
                        np.linspace(*bins[1:], bins[0] + 1),
                        weights=events["weight"][0].to_numpy()
                        * np.nan_to_num(events[key].to_numpy()),
                    )[0]
                )

            nom_vals = toy_hists[0]  # first column are nominal values

            pt_toy_hists = []
            for i in range(events["lp_sf_pt_extrap_vars"].shape[1]):
                pt_toy_hists.append(
                    np.histogram(
                        events[var][0].to_numpy().squeeze(),
                        np.linspace(*bins[1:], bins[0] + 1),
                        weights=events["weight"][0].to_numpy()
                        * events["lp_sf_pt_extrap_vars"][i].to_numpy(),
                    )[0]
                )

            b_ratio_hist = np.histogram(
                events[var][0].to_numpy().squeeze(),
                np.linspace(*bins[1:], bins[0] + 1),
                weights=events["weight"][0].to_numpy()
                * events["lp_sfs_bl_ratio"][0].to_numpy()
                * events["lp_sf_lnN"][0].to_numpy(),
            )[0]

            uncs = {
                "stat_unc": np.minimum(nom_vals, np.std(toy_hists[1:], axis=0)),  # cap at 100% unc
                "syst_rat_unc": np.minimum(nom_vals, (np.abs(sys_up_down[0] - sys_up_down[1])) / 2),
                "np_unc": np.minimum(nom_vals, (np.abs(np_up_down[0] - np_up_down[1])) / 2),
                "dist_unc": np.minimum(nom_vals, (np.abs(dist_up_down[0] - dist_up_down[1])) / 2),
                "um_unc": np.minimum(nom_vals, (np.abs(um_up_down[0] - um_up_down[1])) / 2),
                # "syst_sjm_unc": nom_vals * sj_matching_unc,
                "syst_sjpt_unc": np.minimum(nom_vals, np.std(pt_toy_hists, axis=0)),
                "syst_b_unc": np.abs(1 - (b_ratio_hist / nom_vals)) * nom_vals,
            }

            uncs_lnN_dict[var] = uncs

            unc = np.linalg.norm(list(uncs.values()), axis=0)

            t_hist = deepcopy(pre_hists[var])
            top_matched_key_index = np.where(np.array(list(t_hist.axes[0])) == top_matched_key)[0][
                0
            ]
            t_hist.view(flow=False)[top_matched_key_index, :].value = nom_vals
            post_lnN_hists[var] = t_hist

            post_lnN_hists_err[var] = unc

    for var, var_hist in post_lnN_hists.items():
        name = f"{plot_dir}/postlnN_{var}.pdf"
        plotting.ratioLinePlot(
            var_hist,
            plot_samples,
            year,
            bg_colours=bg_colours,
            bg_err=post_lnN_hists_err[var],
            name=name,
            show=show,
        )

    with Path(f"{plot_dir}/post_lnN_hists.pkl").open("wb") as f:
        pickle.dump(post_lnN_hists, f)

    with Path(f"{plot_dir}/uncs_lnN_dict.pkl").open("wb") as f:
        pickle.dump(uncs_lnN_dict, f)

    with Path(f"{plot_dir}/post_lnN_hists_err.pkl").open("wb") as f:
        pickle.dump(post_lnN_hists_err, f)

    return post_lnN_hists, uncs_lnN_dict, post_lnN_hists_err


def plot_prepost(pre_hists, post_lnN_hists, post_lnN_hists_err, chi2s, plot_dir, year, show=False):
    for var, var_hist in post_lnN_hists.items():
        for prelim, label in [(False, ""), (True, "prelim_")]:
            name = f"{plot_dir}/{label}PrePostlnN_{var}.pdf"
            plotting.ratioLinePlotPrePost(
                var_hist,
                pre_hists[var],
                plot_samples,
                year,
                bg_colours=bg_colours,
                bg_err=post_lnN_hists_err[var],
                chi2s=chi2s[var],
                name=name,
                preliminary=prelim,
                show=show and prelim,
            )


def bin_sf(pre_hists, post_lnN_hists, uncs_lnN_dict, post_lnN_hists_err, plot_dir, binn=-1):
    tvar = "ak8FatJetParTMD_THWW4q"
    top_matched_key_index = np.where(np.array(list(pre_hists[tvar].axes[0])) == top_matched_key)[0][
        0
    ]
    pre_vals = pre_hists[tvar].view(flow=False)[top_matched_key_index, :].value
    nom_vals = post_lnN_hists[tvar].view(flow=False)[top_matched_key_index, :].value
    unc = post_lnN_hists_err[tvar]
    uncs = {key: val[binn] / nom_vals[binn] * 100 for key, val in uncs_lnN_dict[tvar].items()}

    # print("SF: ", nom_vals[binn] / pre_vals[binn])
    # print("Uncs: ", uncs)
    # print("Combined: ", unc[binn] / nom_vals[binn] * 100)
    # print("Abs: ", unc[binn] / pre_vals[binn])

    # save all the text above in a .txt file in plotdir using pathlib
    with Path(f"{plot_dir}/lp_sf_bin{binn}.txt").open("w") as f:
        f.write(f"SF: {nom_vals[binn] / pre_vals[binn]}\n")
        f.write(f"Uncs: {uncs}\n")
        f.write(f"Combined: {unc[binn] / nom_vals[binn]}\n")
        f.write(f"Abs: {unc[binn] / pre_vals[binn]}\n")

    # print out contents of the .txt file
    print(Path(f"{plot_dir}/lp_sf_bin{binn}.txt").read_text())


def chisq_diff(pre_hists, post_lnN_hists, tvars: list[str], plot_dir, lb=20):
    """Check improvement in chi^2"""
    chi2s = {}
    for tvar in tvars:
        data_vals = pre_hists[tvar]["Data", ...].values()
        pre_MC_vals = (
            pre_hists[tvar][sum, :].values()
            - data_vals
            - pre_hists[tvar]["TTbar", :].values()  # remove repeated data
            - pre_hists[tvar]["Top", :].values()
            # - pre_hists[tvar]["SingleTop", :].values()
        )
        post_lnN_MC_vals = (
            post_lnN_hists[tvar][sum, :].values()
            - data_vals
            - post_lnN_hists[tvar]["TTbar", :].values()  # remove repeated data
            - post_lnN_hists[tvar]["Top", :].values()
            # - post_lnN_hists[tvar]["SingleTop", :].values()
        )

        pre_chi2 = chisquare(pre_MC_vals[-lb:], data_vals[-lb:])
        post_chi2 = chisquare(post_lnN_MC_vals[-lb:], data_vals[-lb:])

        # save all the text above in a .txt file
        with Path(f"{plot_dir}/chi2_{tvar}_lb{lb}.txt").open("w") as f:
            f.write(f"Pre chi2: {pre_chi2}\n")
            f.write(f"Post chi2: {post_chi2}\n")

        # print out contents of the .txt file
        # print(Path(f"{plot_dir}/chi2_{tvar}_lb{lb}.txt").read_text())

        chi2s[tvar] = [pre_chi2, post_chi2, lb]

    return chi2s


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
        "--plot-dir",
        help="If making control or template plots, path to directory to save them in",
        default="",
        type=str,
    )


if __name__ == "__main__":
    mpl.use("Agg")
    args = parse_args()
    main(args)
