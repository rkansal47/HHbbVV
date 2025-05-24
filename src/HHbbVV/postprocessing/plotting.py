"""
Common plotting functions.

Author(s): Raghav Kansal
"""

from __future__ import annotations

import pickle
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
import scipy
from hist import Hist
from hist.intervals import poisson_interval, ratio_uncertainty
from numpy.typing import ArrayLike
from pandas import DataFrame

from HHbbVV.hh_vars import LUMI, data_key, hbb_bg_keys, res_sig_keys, txbb_wps
from HHbbVV.postprocessing import utils
from HHbbVV.postprocessing.utils import CUT_MAX_VAL

plt.style.use(hep.style.CMS)
hep.style.use("CMS")
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))

# Data point styling parameters
DATA_STYLE = {
    "histtype": "errorbar",
    "color": "black",
    "markersize": 15,
    "elinewidth": 2,
    "capsize": 0,
}

# this is needed for some reason to update the font size for the first plot
# fig, ax = plt.subplots(1, 1, figsize=(12, 12))
# plt.rcParams.update({"font.size": 24})
# plt.close()


BG_UNC_LABEL = "Total Bkg. Uncertainty"

bg_order = ["Diboson", "HH", "HWW", "Hbb", "ST", "W+Jets", "Z+Jets", "Other", "TT", "QCD"]

sample_label_map = {
    "HHbbVV": r"ggF HH$\rightarrow b\overline{b}VV$",
    "VBFHHbbVV": r"VBF HH$\rightarrow b\overline{b}VV$",
    "qqHH_CV_1_C2V_1_kl_1_HHbbVV": r"VBF HH$\rightarrow b\overline{b}VV$",
    "qqHH_CV_1_C2V_0_kl_1_HHbbVV": r"VBF HH$\rightarrow b\overline{b}VV$ ($\kappa_{2V} = 0$)",
    "qqHH_CV_1_C2V_2_kl_1_HHbbVV": r"VBF HH$\rightarrow b\overline{b}VV$ ($\kappa_{2V} = 2$)",
    "ST": r"Single-$t$",
    "TT": r"$t\bar{t}$",
    "Hbb": r"H$\rightarrow b\overline{b}$",
    "X[900]->HY[80]": r"X[900]$\rightarrow$HY[80]",
    "Top Matched": r"Top matched",
    "Top W Matched": r"Top W matched",
    "Top Unmatched": r"Top unmatched",
    "Corrected Top Matched": r"Corrected top matched",
    "Uncorrected Top Matched": r"Uncorrected top matched",
}

COLOURS = {
    # CMS 10-colour-scheme from
    # https://cms-analysis.docs.cern.ch/guidelines/plotting/colors/#categorical-data-eg-1d-stackplots
    "darkblue": "#3f90da",
    "lightblue": "#92dadd",
    "orange": "#e76300",
    "red": "#bd1f01",
    "darkpurple": "#832db6",
    "brown": "#a96b59",
    "gray": "#717581",
    "beige": "#b9ac70",
    "yellow": "#ffa90e",
    "lightgray": "#94a4a2",
    # extra colours
    "darkred": "#A21315",
    "green": "#7CB518",
    "mantis": "#81C14B",
    "forestgreen": "#2E933C",
    "darkgreen": "#064635",
    "purple": "#9381FF",
    "deeppurple": "#36213E",
    "ashgrey": "#ACBFA4",
    "canary": "#FFE51F",
    "arylideyellow": "#E3C567",
    "earthyellow": "#D9AE61",
    "satinsheengold": "#C8963E",
    "flax": "#EDD382",
    "vanilla": "#F2F3AE",
    "dutchwhite": "#F5E5B8",
}

MARKERS = [
    "o",
    "^",
    "v",
    "<",
    ">",
    "s",
    "+",
    "x",
    "d",
    "1",
    "2",
    "3",
    "4",
    "h",
    "p",
    "|",
    "_",
    "D",
    "H",
]

# for more than 5, probably better to use different MARKERS
LINESTYLES = [
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 5, 1, 5, 1, 5)),
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 5, 1, 5, 1, 5)),
]


BG_COLOURS = {
    "QCD": "darkblue",
    "TT": "brown",
    "Other": "yellow",
    # "V+Jets": "gray",
    "W+Jets": "orange",
    "Z+Jets": "yellow",
    "ST": "lightblue",
    "Diboson": "lightgray",
    "Hbb": "beige",
    "HWW": "gray",
    # below not needed anymore
    "HH": "ashgrey",
    "HHbbVV": "red",
    "qqHH_CV_1_C2V_0_kl_1_HHbbVV": "darkpurple",
}

sig_colour = "red"

SIG_COLOURS = [
    "#bd1f01",
    "#ff5252",
    "#7F2CCB",
    "#ffbaba",
    # "#ff7b7b",
    "#885053",
    "#a70000",
    "#5A1807",
    "#3C0919",
    "#353535",
]

ROC_COLOURS = [
    "darkblue",
    "lightblue",
    "orange",
    "brown",
    "darkpurple",
    "red",
    "gray",
    "beige",
    "yellow",
]


def _combine_hbb_bgs(hists, bg_keys):
    """combine all hbb backgrounds into a single "Hbb" background for plotting"""

    # skip this if no hbb bg keys specified
    if len(set(bg_keys) & set(hbb_bg_keys)) == 0:
        return hists, bg_keys

    h = utils.combine_hbb_bgs(hists)

    bg_keys = [key for key in bg_keys if key not in hbb_bg_keys]

    if "Hbb" not in bg_keys:
        bg_keys.append("Hbb")

    return h, bg_keys


def _combine_other_bgs(hists, bg_keys, keep_keys: list[str] = None):
    """Combine all backgrounds except for ``keep_keys`` into a single "Other" background"""
    if keep_keys is None:
        keep_keys = ["QCD", "TT"]

    combine_keys = [key for key in bg_keys if key not in keep_keys]
    h = utils.combine_other_bgs(hists, combine_keys)
    bg_keys = keep_keys + ["Other"]
    return h, bg_keys


def _process_samples(sig_keys, bg_keys, bg_colours, sig_scale_dict, bg_order, syst, variation):
    # set up samples, colours and labels
    bg_keys = [key for key in bg_order if key in bg_keys]
    bg_colours = [COLOURS[bg_colours[sample]] for sample in bg_keys]
    bg_labels = [sample_label_map.get(bg_key, bg_key) for bg_key in bg_keys]

    if sig_scale_dict is None:
        sig_scale_dict = OrderedDict([(sig_key, 1.0) for sig_key in sig_keys])
    else:
        sig_scale_dict = {key: val for key, val in sig_scale_dict.items() if key in sig_keys}

    sig_labels = OrderedDict()
    for sig_key, sig_scale in sig_scale_dict.items():
        label = sample_label_map.get(sig_key, sig_key)

        if sig_scale != 1:
            if sig_scale <= 10000:
                label = f"{label} $\\times$ {sig_scale:.0f}"
            else:
                label = f"{label} $\\times$ {sig_scale:.1e}"

        sig_labels[sig_key] = label

    # set up systematic variations if needed
    if syst is not None and variation is not None:
        wshift, wsamples = syst
        shift = variation
        skey = {"up": " Up", "down": " Down"}[shift]

        for i, key in enumerate(bg_keys):
            if key in wsamples:
                bg_keys[i] += f"_{wshift}_{shift}"
                bg_labels[i] += skey

        for sig_key in list(sig_scale_dict.keys()):
            if sig_key in wsamples:
                new_key = f"{sig_key}_{wshift}_{shift}"
                sig_scale_dict[new_key] = sig_scale_dict[sig_key]
                sig_labels[new_key] = sig_labels[sig_key] + skey
                del sig_scale_dict[sig_key], sig_labels[sig_key]

    return bg_keys, bg_colours, bg_labels, sig_scale_dict, sig_labels


def _divide_bin_widths(hists, data_err, bg_tot, bg_err):
    """Divide histograms by bin widths"""
    edges = hists.axes[1].edges
    bin_widths = edges[1:] - edges[:-1]

    if data_err is None:
        data_err = (
            np.abs(poisson_interval(hists[data_key, ...].values()) - hists[data_key, ...].values())
            / bin_widths
        )

    if bg_err is not None:
        bg_err = bg_err / bin_widths

    bg_tot = bg_tot / bin_widths
    hists = hists / bin_widths[np.newaxis, :]
    return hists, data_err, bg_tot, bg_err


def _fill_error(ax, edges, down, up, scale=1):
    ax.fill_between(
        np.repeat(edges, 2)[1:-1],
        np.repeat(down, 2) * scale,
        np.repeat(up, 2) * scale,
        color="black",
        alpha=0.2,
        hatch="//",
        linewidth=0,
    )


def _asimov_significance(s, b):
    """Asimov estimate of discovery significance (with no systematic uncertainties).
    See e.g. https://www.pp.rhul.ac.uk/~cowan/atlas/cowan_atlas_15feb11.pdf.
    Or for more explanation: https://www.pp.rhul.ac.uk/~cowan/stat/cowan_munich16.pdf
    """
    return np.sqrt(2 * ((s + b) * np.log(1 + (s / b)) - s))


def add_cms_label(ax, year, data=True, label="Preliminary", loc=2, lumi=True):
    if year == "all":
        hep.cms.label(
            label,
            data=data,
            lumi=f"{np.sum(list(LUMI.values())) / 1e3:.0f}" if lumi else None,
            year=None,
            ax=ax,
            loc=loc,
        )
    else:
        hep.cms.label(
            label,
            data=data,
            lumi=f"{LUMI[year] / 1e3:.0f}" if lumi else None,
            year=year,
            ax=ax,
            loc=loc,
        )


def ratioHistPlot(
    hists: Hist,
    year: str,
    sig_keys: list[str],
    bg_keys: list[str],
    resonant: bool,
    sig_colours: list[str] = None,
    bg_colours: dict[str, str] = None,
    sig_err: ArrayLike | str = None,
    bg_err: ArrayLike = None,
    data_err: ArrayLike | bool | None = None,
    title: str = None,
    name: str = "",
    sig_scale_dict: OrderedDict[str, float] = None,
    ylim: int = None,
    show: bool = False,
    syst: tuple = None,
    variation: str = None,
    region_label: str = None,
    bg_err_type: str = "shaded",
    plot_signal: bool = True,
    plot_data: bool = True,
    bg_order: list[str] = bg_order,
    combine_other_bgs: bool = False,
    log: bool = False,
    ratio_ylims: list[float] = None,
    divide_bin_width: bool = False,
    plot_significance: bool = False,
    significance_dir: str = "right",
    plot_ratio: bool = True,
    plot_pulls: bool = False,
    pull_args: dict = None,
    axrax: tuple = None,
    leg_args: dict = None,
    reorder_legend: bool = True,
    cmslabel: str = None,
    cmsloc: int = 0,
):
    """
    Makes and saves a histogram plot, with backgrounds stacked, signal separate (and optionally
    scaled) with a data/mc ratio plot below

    Args:
        hists (Hist): input histograms per sample to plot
        year (str): datataking year
        sig_keys (List[str]): signal keys
        bg_keys (List[str]): background keys
        sig_colours (Dict[str, str], optional): dictionary of colours per signal. Defaults to sig_colours.
        bg_colours (Dict[str, str], optional): dictionary of colours per background. Defaults to bg_colours.
        sig_err (Union[ArrayLike, str], optional): plot error on signal.
          if string, will take up down shapes from the histograms (assuming they're saved as "{sig_key}_{sig_err}_{up/down}")
          if 1D Array, will take as error per bin
        bg_err (ArrayLike, optional): [bg_tot_down, bg_tot_up] to plot bg variations. Defaults to None.
        data_err (Union[ArrayLike, bool, None], optional): plot error on data.
          if True, will plot poisson error per bin
          if array, will plot given errors per bins
        title (str, optional): plot title. Defaults to None.
        name (str): name of file to save plot
        sig_scale_dict (Dict[str, float]): if scaling signals in the plot, dictionary of factors
          by which to scale each signal
        ylim (optional): y-limit on plot
        show (bool): show plots or not
        syst (Tuple): Tuple of (wshift: name of systematic e.g. pileup,  wsamples: list of samples which are affected by this),
          to plot variations of this systematic.
        variation (str): options:
          "up" or "down", to plot only one wshift variation (if syst is not None).
          Defaults to None i.e. plotting both variations.
        bg_err_type (str): "shaded" or "line".
        plot_data (bool): plot data
        bg_order (List[str]): order in which to plot backgrounds
        ratio_ylims (List[float]): y limits on the ratio plots
        divide_bin_width (bool): divide yields by the bin width (for resonant fit regions)
        plot_significance (bool): plot Asimov significance below ratio plot
        plot_pulls (bool): plot pulls instead of data /bkg. ratio
        significance_dir (str): "Direction" for significance. i.e. a > cut ("right"), a < cut ("left"), or per-bin ("bin").
        axrax (Tuple): optionally input ax and rax instead of creating new ones
        ncol (int): # of legend columns. By default, it is 2 for log-plots and 1 for non-log-plots.
    """

    if ratio_ylims is None:
        ratio_ylims = [0, 2]
    if bg_colours is None:
        bg_colours = BG_COLOURS
    if sig_colours is None:
        sig_colours = SIG_COLOURS
    if leg_args is None:
        leg_args = {"ncol": 2 if log else 1, "fontsize": 24}
    if pull_args is None:
        pull_args = {}
    pull_args["combined_sigma"] = pull_args.get("combined_sigma", False)

    # copy hists and bg_keys so input objects are not changed
    hists, bg_keys = deepcopy(hists), deepcopy(bg_keys)
    hists, bg_keys = _combine_hbb_bgs(hists, bg_keys)
    if combine_other_bgs:
        hists, bg_keys = _combine_other_bgs(hists, bg_keys)

    bg_keys, bg_colours, bg_labels, sig_scale_dict, sig_labels = _process_samples(
        sig_keys, bg_keys, bg_colours, sig_scale_dict, bg_order, syst, variation
    )

    bg_tot = np.maximum(sum([hists[sample, :] for sample in bg_keys]).values(), 0.0)

    if syst is not None and variation is None:
        # plot up/down variations
        wshift, wsamples = syst
        if sig_keys[0] in wsamples:
            sig_err = wshift  # will plot sig variations below
        bg_err = []
        for shift in ["down", "up"]:
            bg_sums = []
            for sample in bg_keys:
                if sample in wsamples and f"{sample}_{wshift}_{shift}" in hists.axes[0]:
                    bg_sums.append(hists[f"{sample}_{wshift}_{shift}", :].values())
                # elif sample != "Hbb":
                else:
                    bg_sums.append(hists[sample, :].values())
            bg_err.append(np.maximum(np.sum(bg_sums, axis=0), 0.0))
        bg_err = np.array(bg_err)

    pre_divide_hists = hists
    pre_divide_bg_tot = bg_tot
    pre_divide_bg_err = bg_err

    if divide_bin_width:
        hists, data_err, bg_tot, bg_err = _divide_bin_widths(
            hists, data_err if plot_data else [], bg_tot, bg_err
        )

    # set up plots
    if axrax is not None:
        if plot_significance:
            raise RuntimeError("Significance plots with input axes not implemented yet.")

        ax, rax = axrax
        ax.sharex(rax)
    elif plot_significance:
        fig, (ax, rax, sax) = plt.subplots(
            3,
            1,
            figsize=(12, 18),
            gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0},
            sharex=True,
        )
    elif plot_ratio:
        fig, (ax, rax) = plt.subplots(
            2,
            1,
            figsize=(12, 14),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1},
            sharex=True,
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 11))

    plt.rcParams.update({"font.size": 24})

    # plot histograms
    y_label = r"<Events / GeV>" if divide_bin_width else "Events / GeV"
    ax.set_ylabel(y_label)

    # background samples
    if len(bg_keys):
        hep.histplot(
            [hists[sample, :] for sample in bg_keys],
            ax=ax,
            histtype="fill",
            stack=True,
            label=bg_labels,
            color=bg_colours,
        )

    # signal samples
    if len(sig_scale_dict) and plot_signal:
        hep.histplot(
            [hists[sig_key, :] * sig_scale for sig_key, sig_scale in sig_scale_dict.items()],
            ax=ax,
            histtype="step",
            label=list(sig_labels.values()),
            color=sig_colours[: len(sig_keys)],
            linewidth=3,
        )

        # plot signal errors
        if isinstance(sig_err, str):
            for skey, shift in [("Up", "up"), ("Down", "down")]:
                hep.histplot(
                    [
                        hists[f"{sig_key}_{sig_err}_{shift}", :] * sig_scale
                        for sig_key, sig_scale in sig_scale_dict.items()
                    ],
                    yerr=0,
                    ax=ax,
                    histtype="step",
                    label=[f"{sig_label} {skey}" for sig_label in sig_labels.values()],
                    alpha=0.6,
                    color=sig_colours[: len(sig_keys)],
                )
        elif sig_err is not None:
            for sig_key, sig_scale in sig_scale_dict.items():
                _fill_error(
                    ax,
                    hists.axes[1].edges,
                    hists[sig_key, :].values() * (1 - sig_err),
                    hists[sig_key, :].values() * (1 + sig_err),
                    sig_scale,
                )

    if bg_err is not None:
        # if divide_bin_width:
        #     raise NotImplementedError("Background error for divide bin width not checked yet")

        if len(np.array(bg_err).shape) == 1:
            bg_err = [bg_tot - bg_err, bg_tot + bg_err]

        if bg_err_type == "shaded":
            ax.fill_between(
                np.repeat(hists.axes[1].edges, 2)[1:-1],
                np.repeat(bg_err[0], 2),
                np.repeat(bg_err[1], 2),
                color="black",
                alpha=0.2,
                hatch="//",
                linewidth=0,
                label=BG_UNC_LABEL,
            )
        else:
            ax.stairs(
                bg_tot,
                hists.axes[1].edges,
                color="black",
                linewidth=3,
                label="BG Total",
                baseline=bg_tot,
            )

            ax.stairs(
                bg_err[0],
                hists.axes[1].edges,
                color="red",
                linewidth=3,
                label="BG Down",
                baseline=bg_err[0],
            )

            ax.stairs(
                bg_err[1],
                hists.axes[1].edges,
                color="#7F2CCB",
                linewidth=3,
                label="BG Up",
                baseline=bg_err[1],
            )

    # plot data
    if plot_data:
        hep.histplot(
            hists[data_key, :],
            ax=ax,
            yerr=data_err,
            xerr=divide_bin_width,
            label=data_key,
            **DATA_STYLE,
        )

    if log:
        ax.set_yscale("log")
        # two column legend
        ax.legend(**leg_args)
    elif reorder_legend:
        if resonant:
            legend_order = [data_key] + list(sig_labels.values()) + bg_order[::-1] + [BG_UNC_LABEL]
        else:
            legend_order = [data_key] + bg_order[::-1] + list(sig_labels.values()) + [BG_UNC_LABEL]
        legend_order = [sample_label_map.get(k, k) for k in legend_order]

        handles, labels = ax.get_legend_handles_labels()
        ordered_handles = [
            handles[labels.index(label)] for label in legend_order if label in labels
        ]
        ordered_labels = [label for label in legend_order if label in labels]
        ax.legend(ordered_handles, ordered_labels, **leg_args)
    else:
        ax.legend(**leg_args)

    y_lowlim = 0 if not log else 1e-5
    if ylim is not None:
        ax.set_ylim([y_lowlim, ylim])
    else:
        ax.set_ylim(y_lowlim)

    ax.margins(x=0)

    # plot ratio below
    if plot_ratio and not plot_pulls:
        if plot_data:
            # new: plotting data errors (black lines) and background errors (shaded) separately
            yerr = np.nan_to_num(
                np.abs(
                    poisson_interval(pre_divide_hists[data_key, ...].values())
                    - pre_divide_hists[data_key, ...].values()
                )
                / (pre_divide_bg_tot + 1e-5)
            )

            hep.histplot(
                pre_divide_hists[data_key, :] / (pre_divide_bg_tot + 1e-5),
                yerr=yerr,
                xerr=divide_bin_width,
                ax=rax,
                **DATA_STYLE,
            )

            if bg_err is not None and bg_err_type == "shaded":
                # (bkg + err) / bkg
                rax.fill_between(
                    np.repeat(hists.axes[1].edges, 2)[1:-1],
                    np.repeat((bg_err[0]) / bg_tot, 2),
                    np.repeat((bg_err[1]) / bg_tot, 2),
                    color="black",
                    alpha=0.1,
                    hatch="//",
                    linewidth=0,
                )
        else:
            rax.set_xlabel(hists.axes[1].label)

        rax.set_ylabel("Data / Bkg.")
        rax.set_ylim(ratio_ylims)

        rax.grid()
        rax.margins(x=0)
        ax.set_xlabel(None)
        ax.set_xticklabels([])

    if plot_pulls:
        # pulls = (data - bkg) / σ
        # σ_data = np.sqrt(predicted_yield)
        sigma_data_sqrd = pre_divide_bg_tot
        if pull_args["combined_sigma"]:
            # σ = np.sqrt(σ_data_sqrd + σ_fit_sqrd)
            sigma = np.sqrt(sigma_data_sqrd + pre_divide_bg_err**2)
            slabel = r"$\sigma$"
            ylim = 3.5
        else:
            sigma = np.sqrt(sigma_data_sqrd)
            slabel = r"$\sigma_\mathrm{Stat}$"
            ylim = 7.5

        pulls = (pre_divide_hists[data_key, :] - pre_divide_bg_tot) / sigma

        hep.histplot(
            pulls,
            yerr=np.ones_like(pulls),
            xerr=divide_bin_width,
            ax=rax,
            **DATA_STYLE,
            label=r"(Data - Bkg.) / " + slabel,
        )

        if not pull_args["combined_sigma"]:
            rax.fill_between(
                np.repeat(hists.axes[1].edges, 2)[1:-1],
                np.repeat(-pre_divide_bg_err / sigma, 2),
                np.repeat(pre_divide_bg_err / sigma, 2),
                color="black",
                alpha=0.2,
                hatch="//",
                linewidth=0,
                label=r"$\sigma_\mathrm{Syst}$ / $\sigma_\mathrm{Stat}$",
            )

        # hep.histplot(
        #     pulls,
        #     ax=rax,
        #     color=COLOURS["gray"],
        #     label=r"(Data - Bkg.) / $\sigma$",
        # )

        if plot_signal:
            hep.histplot(
                [pre_divide_hists[sig_key, :] / sigma for sig_key in sig_scale_dict],
                ax=rax,
                color=sig_colours[: len(sig_keys)],
                label=[
                    sample_label_map.get(sig_key, sig_key) + " / " + slabel
                    for sig_key in sig_scale_dict
                ],
                linewidth=4,
            )

        # put signal label in the top right

        # Create two separate legends - one for signal in top right, one for others in lower right
        handles, labels = rax.get_legend_handles_labels()
        signal_handles = []
        signal_labels = []
        other_handles = []
        other_labels = []

        for handle, label in zip(handles, labels):
            if any(sample_label_map.get(sig_key, sig_key) in label for sig_key in sig_scale_dict):
                signal_handles.append(handle)
                signal_labels.append(label)
            else:
                other_handles.append(handle)
                other_labels.append(label)

        if signal_handles:
            # Add first legend for signal in upper right
            first_legend = rax.legend(signal_handles, signal_labels, ncol=1, loc="upper right")
            rax.add_artist(first_legend)  # Add the first legend to the plot
        if other_handles:
            # Add second legend for others in lower right
            rax.legend(other_handles, other_labels, ncol=2, loc="lower right")

        rax.set_ylabel("Pull")
        rax.set_ylim(-ylim, ylim)
        # rax.grid()
        rax.margins(x=0)
        rax.hlines(0, *rax.get_xlim(), color=COLOURS["gray"], linewidth=1)
        ax.set_xlabel(None)
        ax.tick_params(axis="x", labelbottom=False)

    if plot_significance:
        sigs = [pre_divide_hists[sig_key, :].values() for sig_key in sig_scale_dict]

        if significance_dir == "left":
            bg_tot = np.cumsum(bg_tot[::-1])[::-1]
            sigs = [np.cumsum(sig[::-1])[::-1] for sig in sigs]
            sax.set_ylabel(r"Asimov Sign. for $\leq$ Cuts")
        elif significance_dir == "right":
            bg_tot = np.cumsum(bg_tot)
            sigs = [np.cumsum(sig) for sig in sigs]
            sax.set_ylabel(r"Asimov Sign. for $\geq$ Cuts")
        elif significance_dir == "bin":
            sax.set_ylabel("Asimov Sign. per Bin")
        else:
            raise RuntimeError(
                'Invalid value for ``significance_dir``. Options are ["left", "right", "bin"].'
            )

        edges = pre_divide_hists.axes[1].edges
        hep.histplot(
            [(_asimov_significance(sig, bg_tot), edges) for sig in sigs],
            ax=sax,
            histtype="step",
            label=[sample_label_map.get(sig_key, sig_key) for sig_key in sig_scale_dict],
            color=sig_colours[: len(sig_keys)],
        )

        sax.legend(fontsize=12)
        sax.set_yscale("log")
        sax.set_ylim([1e-7, 10])
        sax.set_xlabel(hists.axes[1].label)

    if title is not None:
        ax.set_title(title, y=1.08)

    if region_label is not None:
        mline = "\n" in region_label
        xpos = 0.32 if not mline else 0.24
        ypos = 0.91 if not mline else 0.87
        xpos = 0.035 if not resonant else xpos
        ax.text(
            xpos,
            ypos,
            region_label,
            transform=ax.transAxes,
            fontsize=24,
            fontproperties="Tex Gyre Heros:bold",
        )

    add_cms_label(ax, year, label=cmslabel, loc=cmsloc)

    if axrax is None:
        if len(name):
            plt.savefig(name, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()


def ratioLinePlot(
    hists: Hist,
    bg_keys: list[str],
    year: str,
    bg_colours: dict[str, str] = None,
    sig_colour: str = sig_colour,  # noqa: ARG001
    bg_err: np.ndarray | str = None,
    data_err: ArrayLike | bool | None = None,
    title: str = None,
    pulls: bool = False,
    name: str = "",
    sig_scale: float = 1.0,  # noqa: ARG001
    show: bool = True,
):
    """
    Makes and saves a histogram plot, with backgrounds stacked, signal separate (and optionally
    scaled) with a data/mc ratio plot below
    """
    if bg_colours is None:
        bg_colours = BG_COLOURS

    plt.rcParams.update({"font.size": 24})

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1}, sharex=True
    )

    bg_tot = np.sum(hists[bg_keys, :].values(), axis=0)
    plot_hists = [hists[sample, :] for sample in bg_keys]

    ax.set_ylabel("Events")
    hep.histplot(
        plot_hists + [sum(plot_hists)],
        ax=ax,
        histtype="step",
        label=bg_keys + ["Total"],
        color=[COLOURS[bg_colours[sample]] for sample in bg_keys] + ["black"],
        yerr=False,
    )

    if bg_err is not None:
        ax.fill_between(
            np.repeat(hists.axes[1].edges, 2)[1:-1],
            np.repeat(bg_tot - bg_err, 2),
            np.repeat(bg_tot + bg_err, 2),
            color="black",
            alpha=0.2,
            hatch="//",
            linewidth=0,
            label="Lund Plane Uncertainty",
        )

    hep.histplot(
        hists[data_key, :], ax=ax, yerr=data_err, histtype="errorbar", label=data_key, color="black"
    )

    if bg_err is not None:
        # Switch order so that uncertainty label comes at the end
        handles, labels = ax.get_legend_handles_labels()
        # Reorder to put uncertainty at the end
        handles = handles[1:] + handles[:1]
        labels = labels[1:] + labels[:1]
        # Split into two columns with 5 and 6 items
        left_handles = handles[:5]
        left_labels = labels[:5]
        right_handles = handles[5:]
        right_labels = labels[5:]
        # Combine back with None as separator
        handles = left_handles + [None] + right_handles
        labels = left_labels + [""] + right_labels
        ax.legend(handles, labels, ncol=2, fontsize=24)
    else:
        ax.legend(ncol=2)

    ax.set_ylim(0, ax.get_ylim()[1] * 1.5)

    data_vals = hists[data_key, :].values()

    if not pulls:
        yerr = ratio_uncertainty(data_vals, bg_tot, "poisson")

        hep.histplot(
            hists[data_key, :] / (sum([hists[sample, :] for sample in bg_keys]).values() + 1e-5),
            yerr=yerr,
            ax=rax,
            **DATA_STYLE,
        )

        if bg_err is not None:
            # (bkg + err) / bkg
            rax.fill_between(
                np.repeat(hists.axes[1].edges, 2)[1:-1],
                np.repeat((bg_tot - bg_err) / bg_tot, 2),
                np.repeat((bg_tot + bg_err) / bg_tot, 2),
                color="black",
                alpha=0.1,
                hatch="//",
                linewidth=0,
            )

        rax.set_ylabel("Data / Sim.")
        rax.set_ylim(0.5, 1.5)
        rax.grid()
    else:
        bg_tot / (data_vals + 1e-5)
        yerr = bg_err / data_vals

        hep.histplot(
            (sum([hists[sample, :] for sample in bg_keys]) / (data_vals + 1e-5) - 1) * (-1),
            yerr=yerr,
            ax=rax,
            histtype="errorbar",
            **DATA_STYLE,
        )
        rax.set_ylabel("(Data - MC) / Data")
        rax.set_ylim(-0.5, 0.5)
        rax.grid()

    if title is not None:
        ax.set_title(title, y=1.08)

    add_cms_label(ax, year, loc=0)

    if len(name):
        plt.savefig(name, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def ratioLinePlotPrePost(
    hists: Hist,
    pre_hists: Hist,
    bg_keys: list[str],
    year: str,
    bg_colours: dict[str, str] = None,
    bg_err: np.ndarray | str = None,
    data_err: ArrayLike | bool | None = None,
    title: str = None,
    chi2s: list = None,  # [pre, post reweighting chi2s, dofs]
    pulls: bool = False,
    name: str = "",
    preliminary: bool = True,
    show: bool = True,
):
    """
    Makes and saves a histogram plot, with backgrounds stacked, signal separate (and optionally
    scaled) with a data/mc ratio plot below
    """
    if bg_colours is None:
        bg_colours = BG_COLOURS

    plt.rcParams.update({"font.size": 24})

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.1}, sharex=True
    )

    bg_tot = np.sum(hists[bg_keys, :].values(), axis=0)

    plot_hists = []
    labels = []
    colors = []
    linestyles = []
    alpha = []
    markers = []
    for i, k in enumerate(bg_keys):
        if k == "Top Matched":
            plot_hists.append(pre_hists[k, :])
            labels.append("Uncorrected top matched")
            colors.append(COLOURS[bg_colours[k]])
            linestyles.append("--")
            alpha.append(0.5)
            markers.append(None)

            plot_hists.append(hists[k, :])
            labels.append("Corrected top matched")
            colors.append(COLOURS[bg_colours[k]])
            linestyles.append("-")
            alpha.append(1)
            markers.append(None)
        else:
            plot_hists.append(hists[k, :])
            labels.append(sample_label_map.get(k, k))
            colors.append(COLOURS[bg_colours[k]])
            linestyles.append("-")
            markers.append(MARKERS[i])
            alpha.append(1)

    plot_hists = plot_hists + [
        sum([pre_hists[sample, :] for sample in bg_keys]),
        sum([hists[sample, :] for sample in bg_keys]),
    ]
    labels = labels + ["Uncorrected total", "Corrected total"]
    colors = colors + ["black", "black"]
    linestyles = linestyles + ["--", "-"]
    alpha = alpha + [0.5, 1]
    markers = markers + [None, None]

    hep.histplot(
        plot_hists,
        ax=ax,
        histtype="step",
        # label=labels,
        color=colors,
        linestyle=linestyles,
        # marker=markers,
        alpha=alpha,
        yerr=False,
        flow="none",
    )

    hep.histplot(
        plot_hists,
        ax=ax,
        histtype="errorbar",
        # label=labels,
        color=colors,
        # linestyle=linestyles,
        marker=markers,
        markerfacecolor="none",
        alpha=alpha,
        yerr=False,
        flow="none",
    )

    hep.histplot(
        [h * -1 for h in plot_hists],
        ax=ax,
        histtype="errorbar",
        label=labels,
        color=colors,
        linestyle=linestyles,
        marker=markers,
        markerfacecolor="none",
        alpha=alpha,
        yerr=False,
        flow="none",
    )

    if bg_err is not None:
        ax.fill_between(
            np.repeat(hists.axes[1].edges, 2)[1:-1],
            np.repeat(bg_tot - bg_err, 2),
            np.repeat(bg_tot + bg_err, 2),
            color="black",
            alpha=0.2,
            hatch="//",
            linewidth=0,
            label="LJP uncertainty",
        )

    hep.histplot(
        hists[data_key, :], ax=ax, yerr=data_err, label=data_key, **DATA_STYLE, flow="none"
    )

    if bg_err is not None:
        # Switch order so that uncertainty label comes at the end
        handles, labels = ax.get_legend_handles_labels()
        # Reorder to put uncertainty at the end
        handles = handles[-1:] + handles[1:-1] + handles[:1]
        labels = labels[-1:] + labels[1:-1] + labels[:1]
        ax.legend(handles, labels, ncol=2, fontsize=24)
    else:
        ax.legend(ncol=2, fontsize=24)

    ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
    ax.set_ylabel("Events / 0.04 units")
    ax.set_xlabel(None)
    ax.margins(x=0)

    data_vals = hists[data_key, :].values()

    if not pulls:
        yerr = ratio_uncertainty(data_vals, bg_tot, "poisson")

        hep.histplot(
            hists[data_key, :]
            / (sum([pre_hists[sample, :] for sample in bg_keys]).values() + 1e-5),
            # yerr=yerr,
            yerr=False,
            ax=rax,
            histtype="errorbar",
            color=COLOURS["red"],
            label="Uncorrected",
            markerfacecolor="none",
            marker="^",
            flow="none",
        )

        hep.histplot(
            hists[data_key, :] / (sum([hists[sample, :] for sample in bg_keys]).values() + 1e-5),
            yerr=yerr,
            ax=rax,
            label="Corrected",
            **DATA_STYLE,
            flow="none",
        )

        if bg_err is not None:
            # (bkg + err) / bkg
            rax.fill_between(
                np.repeat(hists.axes[1].edges, 2)[1:-1],
                np.repeat((bg_tot - bg_err) / bg_tot, 2),
                np.repeat((bg_tot + bg_err) / bg_tot, 2),
                color="black",
                alpha=0.1,
                hatch="//",
                linewidth=0,
            )

        rax.set_ylabel("Data / Sim.")
        rax.set_ylim(0.5, 1.5)
        # rax.grid()
        rax.legend()
    else:
        bg_tot / (data_vals + 1e-5)
        yerr = bg_err / data_vals

        hep.histplot(
            (sum([hists[sample, :] for sample in bg_keys]) / (data_vals + 1e-5) - 1) * (-1),
            yerr=yerr,
            ax=rax,
            histtype="errorbar",
            color="black",
            capsize=4,
            flow="none",
        )
        rax.set_ylabel("(Data - Sim.) / Data")
        rax.set_ylim(-0.5, 0.5)
        # rax.grid()

    rax.margins(x=0)
    rax.hlines(1, *rax.get_xlim(), color=COLOURS["gray"], linewidth=1)

    if title is not None:
        ax.set_title(title, y=1.08)

    if chi2s is not None:
        fs = 20
        rax.text(
            0.35,
            0.12,
            rf"$\chi^2$ / ndof = {chi2s[0]:.1f} / {chi2s[2]}",
            transform=rax.transAxes,
            fontsize=fs,
            color=COLOURS["red"],
        )
        rax.text(
            0.65,
            0.12,
            rf"$\chi^2$ / ndof = {chi2s[1]:.1f} / {chi2s[2]}",
            transform=rax.transAxes,
            fontsize=fs,
            color="black",
        )

    add_cms_label(ax, year, loc=0, label="Preliminary" if preliminary else None)

    if len(name):
        plt.savefig(name, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def hist2ds(
    hists: dict[str, Hist],
    plot_dir: str,
    regions: list[str] = None,
    region_labels: dict[str, str] = None,  # noqa: ARG001
    samples: list[str] = None,
    fail_zlim: float = None,
    pass_zlim: float = None,
    skip_blinded: bool = True,
    preliminary: bool = True,
    show: bool = False,
):
    """
    2D hists for each region and sample in ``hists``.

    Args:
        hists (Dict[str, Hist]): dictionary of hists per region.
        plot_dir (str): directory in which to save plots.
        regions (List[str], optional): regions to plot. Defaults to None i.e. plot all in hists.
        region_labels (Dict[str, str], optional): Optional labels for each region in hists.
        samples (List[str], optional): samples to plot. Defaults to None i.e. plot all in hists.
        fail_zlim (float, optional): fail region plots upper limit. Defaults to None.
        pass_zlim (float, optional): pass region plots upper limit. Defaults to None.
        show (bool, optional): show plot or close. Defaults to True.
    """
    if samples is None:
        samples = list(next(iter(hists.values())).axes[0])

    if regions is None:
        regions = list(hists.keys())

    for region in regions:
        h = utils.combine_hbb_bgs(hists[region])
        if region.endswith("Blinded") and skip_blinded:
            continue

        print(f"\t\t{region}")

        # region_label = region_labels[region] if region_labels is not None else region
        pass_region = region.startswith("pass")
        lim = pass_zlim if pass_region else fail_zlim

        bg_samps = [
            s
            for s in samples
            if ((s != data_key) and (s not in res_sig_keys) and ("->HY" not in s))
        ]
        print("\t\t\tBG samples: ", bg_samps)
        bg_hists = [h[sample, ...] for sample in bg_samps]
        bg_tot = sum(bg_hists)

        dbg_ratio = h[data_key, ...] / (bg_tot + 1e-6)

        phists = [h[sample, ...] for sample in samples] + [bg_tot, dbg_ratio]
        skeys = samples + ["Bkg.", "Data / Bkg."]

        for phist, skey in zip(phists, skeys):
            if "->HY" in skey and not pass_region:
                continue

            if skey != "Data / Bkg.":
                continue

            print(f"\t\t\t{skey}")
            slabel = sample_label_map.get(skey, skey)

            if lim is not None:
                norm = mpl.colors.LogNorm(vmin=lim[0], vmax=lim[1])
            else:
                norm = mpl.colors.LogNorm()

            fig, ax = plt.subplots(figsize=(12, 12))
            if skey == "Data / Bkg.":
                h2d = hep.hist2dplot(phist, cmap="turbo", cmin=0, cmax=2)
            else:
                h2d = hep.hist2dplot(phist, cmap="turbo", norm=norm)
            h2d.cbar.set_label(f"{slabel} Events")

            v, mYbins, mXbins = phist.to_numpy()
            for i in range(len(mYbins) - 1):
                for j in range(len(mXbins) - 1):
                    if not np.isnan(v[i, j]):
                        ax.text(
                            (mYbins[i] + mYbins[i + 1]) / 2,
                            (mXbins[j] + mXbins[j + 1]) / 2,
                            v[i, j].round(2),
                            color="black",
                            ha="center",
                            va="center",
                            fontsize=12,
                        )

            # plt.title(f"{sample} in {region_label} Region")
            add_cms_label(ax, "all", "Preliminary" if preliminary else None, loc=0)
            pkey = skey.replace(" / ", "_").replace(".", "")
            plt.savefig(f"{plot_dir}/{region}_{pkey}_2d.pdf", bbox_inches="tight")

            if show:
                plt.show()
            else:
                plt.close()


def hist2dPullPlot(
    hists: Hist,
    bg_err: np.ArrayLike,
    sig_key: str,
    bg_keys: list[str],
    region_label: str,
    # zlim: float = None,
    preliminary: bool = True,
    name: str = "",
    show: bool = False,
):
    """
    2D pull plots.

    Args:
        hists (Dict[str, Hist]): dictionary of hists per region.
        plot_dir (str): directory in which to save plots.
        regions (List[str], optional): regions to plot. Defaults to None i.e. plot all in hists.
        region_labels (Dict[str, str], optional): Optional labels for each region in hists.
        fail_zlim (float, optional): fail region plots upper limit. Defaults to None.
        pass_zlim (float, optional): pass region plots upper limit. Defaults to None.
        show (bool, optional): show plot or close. Defaults to True.
    """
    bg_tot = np.maximum(sum([hists[sample, ...] for sample in bg_keys]).values(), 0.0)
    sigma = np.sqrt(bg_tot + bg_err.T**2)
    pulls = (hists[data_key, ...] - bg_tot) / sigma

    fig, ax = plt.subplots(figsize=(12, 12))

    # 2D Pull plot
    h2d = hep.hist2dplot(
        pulls.values().T,
        hists.axes[2].edges,
        hists.axes[1].edges,
        cmap="viridis",
        cmin=-3.5,
        cmax=3.5,
        ax=ax,
    )
    h2d.cbar.set_label(r"(Data - Bkg.) / $\sigma$")
    h2d.pcolormesh.set_edgecolor("face")

    # Plot signal contours
    sig_hist = hists[sig_key, ...].values() / sigma
    levels = np.array([0.04, 0.5, 0.95]) * np.max(sig_hist)

    # Create interpolated grid with 4x more points
    x = hists.axes[1].centers
    y = hists.axes[2].centers
    x_interp = np.linspace(x.min(), x.max(), len(x) * 4)
    y_interp = np.linspace(y.min(), y.max(), len(y) * 4)

    # Interpolate signal histogram with increased smoothing
    sig_interp = scipy.interpolate.RectBivariateSpline(y, x, sig_hist.T)

    # Use edges instead of centers for interpolation range
    x_edges = hists.axes[1].edges
    y_edges = hists.axes[2].edges
    x_interp = np.linspace(x_edges[0], x_edges[-1], len(x) * 4)
    y_interp = np.linspace(y_edges[0], y_edges[-1], len(y) * 4)
    X, Y = np.meshgrid(x_interp, y_interp)
    Z = sig_interp(y_interp, x_interp)

    sig_colour = COLOURS["red"]

    cs = ax.contour(
        Y.T,
        X.T,
        Z.T,
        levels=levels,
        colors=sig_colour,
        # linestyles=["--", "-", "--"],
        linewidths=3,
    )
    ax.clabel(cs, cs.levels, inline=True, fmt="%.2f", fontsize=12)

    xticks = [800, 1200, 1600, 2000, 3000, 4400]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.0f}" for x in xticks], rotation=45)
    ax.set_ylabel(hists.axes[1].label)
    ax.set_xlabel(hists.axes[2].label)

    # Add legend for signal contours
    handles, labels = ax.get_legend_handles_labels()
    # Create proxy artist for contour lines
    contour_proxy = plt.Line2D([], [], color=sig_colour, linestyle="-", linewidth=3)
    handles.append(contour_proxy)
    labels.append(sample_label_map.get(sig_key, sig_key) + r" / $\sigma$")
    ax.legend(
        handles,
        labels,
        loc="upper right",
        # bbox_to_anchor=(1.0, 0.98),  # Moved down from default 1.0
        fontsize=28,
        frameon=False,
    )

    add_cms_label(ax, "all", data=True, label="Preliminary" if preliminary else None, loc=2)

    ax.text(
        0.3,
        0.92,
        region_label,
        transform=ax.transAxes,
        fontsize=28,
        fontproperties="Tex Gyre Heros:bold",
    )

    if len(name):
        plt.savefig(name, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return pulls


def sigErrRatioPlot(
    h: Hist,
    year: str,
    sig_key: str,
    wshift: str,
    title: str = None,
    plot_dir: str = None,
    name: str = None,
    show: bool = False,
):
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [3, 1], "hspace": 0}, sharex=True
    )

    nom = h[sig_key, :].values()
    hep.histplot(
        h[sig_key, :],
        histtype="step",
        label=sig_key,
        yerr=False,
        color=SIG_COLOURS[0],
        ax=ax,
        linewidth=2,
    )

    for skey, shift in [("Up", "up"), ("Down", "down")]:
        colour = {"up": "#81C14B", "down": "#1f78b4"}[shift]
        hep.histplot(
            h[f"{sig_key}_{wshift}_{shift}", :],
            histtype="step",
            yerr=False,
            label=f"{sig_key} {skey}",
            color=colour,
            ax=ax,
            linewidth=2,
        )

        hep.histplot(
            h[f"{sig_key}_{wshift}_{shift}", :] / nom,
            histtype="errorbar",
            # yerr=False,
            label=f"{sig_key} {skey}",
            color=colour,
            ax=rax,
        )

    ax.legend()
    ax.set_ylim(0)
    ax.set_ylabel("Events")
    add_cms_label(ax, year)
    ax.set_title(title, y=1.08)

    rax.set_ylim([0, 2])
    rax.set_xlabel(r"$m^{bb}_{reg}$ (GeV)")
    rax.legend()
    rax.set_ylabel("Variation / Nominal")
    rax.grid(axis="y")

    plt.savefig(f"{plot_dir}/{name}.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def rocCurve(
    fpr,
    tpr,
    auc=None,
    sig_eff_lines=None,
    # bg_eff_lines=[],
    title=None,
    xlim=None,
    ylim=None,
    plot_dir="",
    name="",
    log: bool = True,
    show: bool = False,
):
    """Plots a ROC curve"""
    if ylim is None:
        ylim = [1e-06, 1]
    if xlim is None:
        xlim = [0, 0.8]
    if sig_eff_lines is None:
        sig_eff_lines = []

    line_style = {"colors": "lightgrey", "linestyles": "dashed"}

    plt.figure(figsize=(12, 12))

    plt.plot(tpr, fpr, label=f"AUC: {auc:.2f}" if auc is not None else None)

    for sig_eff in sig_eff_lines:
        y = fpr[np.searchsorted(tpr, sig_eff)]
        plt.hlines(y=y, xmin=0, xmax=sig_eff, **line_style)
        plt.vlines(x=sig_eff, ymin=0, ymax=y, **line_style)

    if log:
        plt.yscale("log")

    plt.xlabel("Signal efficiency")
    plt.ylabel("Background efficiency")
    plt.title(title)
    plt.grid(which="major")

    if auc is not None:
        plt.legend()

    plt.xlim(*xlim)
    plt.ylim(*ylim)
    hep.cms.label(data=False, label="Preliminary", rlabel="(13 TeV)")

    if len(name):
        plt.savefig(plot_dir / f"{name}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def multiROCCurveGrey(
    rocs: dict,
    sig_effs: list[float],
    plot_dir: Path,
    xlim=None,
    ylim=None,
    name: str = "",
    log: bool = True,
    show: bool = False,
):
    """_summary_

    Args:
        rocs (dict): {label: {sig_key1: roc, sig_key2: roc, ...}, ...} where label is e.g Test or Train
        sig_effs (list[float]): plot signal efficiency lines
    """
    if ylim is None:
        ylim = [1e-06, 1]
    if xlim is None:
        xlim = [0, 1]
    line_style = {"colors": "lightgrey", "linestyles": "dashed"}

    plt.figure(figsize=(12, 12))
    for roc_sigs in rocs.values():
        for roc in roc_sigs.values():
            auc_label = f" (AUC: {roc['auc']:.2f})" if "auc" in roc else ""

            plt.plot(
                roc["tpr"],
                roc["fpr"],
                label=roc["label"] + auc_label,
                linewidth=2,
            )

            for sig_eff in sig_effs:
                y = roc["fpr"][np.searchsorted(roc["tpr"], sig_eff)]
                plt.hlines(y=y, xmin=0, xmax=sig_eff, **line_style)
                plt.vlines(x=sig_eff, ymin=0, ymax=y, **line_style)

    hep.cms.label(data=False, label="Preliminary", rlabel="(13 TeV)")
    if log:
        plt.yscale("log")
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background efficiency")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend(loc="upper left")
    plt.grid(which="major")

    if len(name):
        plt.savefig(plot_dir / f"{name}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


th_colours = [
    # "#36213E",
    # "#9381FF",
    "#1f78b4",
    # "#a6cee3",
    # "#32965D",
    "#7CB518",
    "#EDB458",
    # "#ff7f00",
    "#a70000",
]


def multiROCCurve(
    rocs: dict,
    thresholds=None,
    title=None,
    xlim=None,
    ylim=None,
    log=True,
    year="all",
    kin_label=None,
    plot_dir="",
    name="",
    prelim=True,
    show=False,
):
    if ylim is None:
        ylim = [1e-06, 1]
    if xlim is None:
        xlim = [0, 1]
    if thresholds is None:
        thresholds = [[0.9, 0.98, 0.995, 0.9965, 0.998], [0.99, 0.997, 0.998, 0.999, 0.9997]]

    plt.rcParams.update({"font.size": 32})

    fig, ax = plt.subplots(figsize=(12, 12))
    for i, roc_sigs in enumerate(rocs.values()):
        for j, roc in enumerate(roc_sigs.values()):
            if len(np.array(thresholds).shape) > 1:
                pthresholds = thresholds[j]
            else:
                pthresholds = thresholds

            ax.plot(
                roc["tpr"],
                roc["fpr"],
                label=roc["label"],
                linewidth=3,
                color=COLOURS[ROC_COLOURS[i * len(roc_sigs) + j]],
                linestyle=LINESTYLES[i * len(roc_sigs) + j],
            )

            pths = {th: [[], []] for th in pthresholds}
            for th in pthresholds:
                idx = _find_nearest(roc["thresholds"], th)
                pths[th][0].append(roc["tpr"][idx])
                pths[th][1].append(roc["fpr"][idx])
                # print(roc["tpr"][idx])

            for k, th in enumerate(pthresholds):
                ax.scatter(
                    *pths[th],
                    marker="o",
                    s=80,
                    label=(
                        f"Score > {th}" if i == len(rocs) - 1 and j == len(roc_sigs) - 1 else None
                    ),
                    color=th_colours[k],
                    zorder=100,
                )

                ax.vlines(
                    x=pths[th][0],
                    ymin=0,
                    ymax=pths[th][1],
                    color=th_colours[k],
                    linestyles="dashed",
                    alpha=0.5,
                )

                ax.hlines(
                    y=pths[th][1],
                    xmin=0,
                    xmax=pths[th][0],
                    color=th_colours[k],
                    linestyles="dashed",
                    alpha=0.5,
                )

    add_cms_label(ax, year, data=False, label="Preliminary" if prelim else None, loc=1, lumi=False)

    if log:
        plt.yscale("log")

    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background efficiency")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(loc="lower right", fontsize=24)

    if title:
        ax.text(
            0.05,
            0.83,
            title,
            transform=ax.transAxes,
            fontsize=24,
            fontproperties="Tex Gyre Heros:bold",
        )

    if kin_label:
        ax.text(
            0.05,
            0.72,
            kin_label,
            transform=ax.transAxes,
            fontsize=20,
            fontproperties="Tex Gyre Heros",
        )

    if len(name):
        # save ROC as pickle
        with Path(f"{plot_dir}/{name}.pkl").open("wb") as f:
            pickle.dump(rocs, f)

        plt.savefig(f"{plot_dir}/{name}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_HEM2d(hists2d: list[Hist], plot_keys: list[str], year: str, name: str, show: bool = False):
    fig, axs = plt.subplots(
        len(plot_keys),
        2,
        figsize=(20, 8 * len(plot_keys)),
        gridspec_kw={"wspace": 0.25, "hspace": 0.25},
    )

    for j, key in enumerate(plot_keys):
        for i in range(2):
            ax = axs[j][i]
            hep.hist2dplot(hists2d[i][key, ...], cmap="turbo", ax=ax)
            hep.cms.label("Preliminary", data=True, lumi=round(LUMI[year] * 1e-3), year=year, ax=ax)
            ax.set_title(key, y=1.07)
            ax._children[0].colorbar.set_label("Events")

    plt.savefig(name, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def ratioTestTrain(
    h: Hist,
    training_keys: list[str],
    shape_var: utils.ShapeVar,
    year: str,
    plot_dir="",
    name="",
    show=False,
):
    """Line and ratio plots comparing training and testing distributions

    Args:
        h (Hist): Histogram with ["Train", "Test"] x [sample] x [shape_var] axes
        training_keys (List[str]): List of training samples.
        shape_var (utils.ShapeVar): Variable being plotted.
        year (str): year.
    """
    plt.rcParams.update({"font.size": 24})

    style = {
        "Train": {"linestyle": "--"},
        "Test": {"alpha": 0.5},
    }

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [3, 1], "hspace": 0}, sharex=True
    )

    labels = [sample_label_map.get(key, key) for key in training_keys]
    for data in ["Train", "Test"]:
        plot_hists = [h[data, sample, :] for sample in training_keys]

        ax.set_ylabel("Events")
        hep.histplot(
            plot_hists,
            ax=ax,
            histtype="step",
            label=[data + " " + label for label in labels],
            color=[COLOURS[BG_COLOURS[sample]] for sample in training_keys],
            yerr=True,
            **style[data],
        )

    ax.set_xlim([shape_var.axis.edges[0], shape_var.axis.edges[-1]])
    ax.set_yscale("log")
    ax.legend(fontsize=20, ncol=2, loc="center left")

    plot_hists = [h["Train", sample, :] / h["Test", sample, :].values() for sample in training_keys]
    err = [
        np.sqrt(
            np.sum(
                [
                    h[data, sample, :].variances() / (h[data, sample, :].values() ** 2)
                    for data in ["Train", "Test"]
                ],
                axis=0,
            )
        )
        for sample in training_keys
    ]

    hep.histplot(
        plot_hists,
        ax=rax,
        histtype="errorbar",
        label=labels,
        color=[COLOURS[BG_COLOURS[sample]] for sample in training_keys],
        yerr=np.abs([err[i] * plot_hists[i].values() for i in range(len(plot_hists))]),
    )

    rax.set_ylim([0, 2])
    rax.set_xlabel(shape_var.label)
    rax.set_ylabel("Train / Test")
    rax.legend(fontsize=20, loc="upper left", ncol=3)
    rax.grid()

    hep.cms.label(data=False, year=year, ax=ax, lumi=f"{LUMI[year] / 1e3:.0f}")

    if len(name):
        plt.savefig(f"{plot_dir}/{name}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def cutsLinePlot(
    events_dict: dict[str, DataFrame],
    shape_var: utils.ShapeVar,
    plot_key: str,
    cut_var: str,
    cut_var_label: str,
    cuts: list[float],
    year: str,
    weight_key: str,
    bb_masks: dict[str, DataFrame] = None,
    plot_dir: str = "",
    name: str = "",
    ratio: bool = False,
    ax: plt.Axes = None,
    show: bool = False,
):
    """Plot line plots of ``shape_var`` for different cuts on ``cut_var``."""
    if ax is None:
        if ratio:
            assert cuts[0] == 0, "First cut must be 0 for ratio plots."
            fig, (ax, rax) = plt.subplots(
                2,
                1,
                figsize=(12, 14),
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0},
                sharex=True,
            )
        else:
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        in_ax = False
    else:
        if ratio:
            raise NotImplementedError("Ratio plots not implemented with input axes.")
        in_ax = True

    plt.rcParams.update({"font.size": 24})

    hists = OrderedDict()
    for cut in cuts:
        sel, _ = utils.make_selection({cut_var: [cut, CUT_MAX_VAL]}, events_dict, bb_masks)
        h = utils.singleVarHist(
            events_dict, shape_var, bb_masks, weight_key=weight_key, selection=sel
        )

        hists[cut] = h[plot_key, ...] / np.sum(h[plot_key, ...].values())

        hep.histplot(
            hists[cut],
            yerr=True,
            label=f"{cut_var_label} >= {cut}",
            ax=ax,
            linewidth=2,
            alpha=0.8,
        )

    ax.set_xlabel(shape_var.label)
    ax.set_ylabel("Fraction of Events")
    ax.legend()

    if ratio:
        rax.hlines(1, shape_var.axis.edges[0], shape_var.axis.edges[-1], linestyle="--", alpha=0.5)
        vals_nocut = hists[0].values()

        next(rax._get_lines.prop_cycler)  # skip first
        for cut in cuts[1:]:
            hep.histplot(
                hists[cut] / vals_nocut,
                yerr=True,
                label=f"BDTScore >= {cut}",
                ax=rax,
                histtype="errorbar",
            )

        rax.set_ylim([0.4, 2.2])
        rax.set_ylabel("Ratio to Inclusive Shape")
        # rax.legend()

    if year == "all":
        hep.cms.label(
            "Preliminary",
            data=True,
            lumi=f"{np.sum(list(LUMI.values())) / 1e3:.0f}",
            year=None,
            ax=ax,
        )
    else:
        hep.cms.label("Preliminary", data=True, lumi=f"{LUMI[year] / 1e3:.0f}", year=year, ax=ax)

    if in_ax:
        return

    if len(name):
        plt.savefig(f"{plot_dir}/{name}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plotMassSculpting(
    bbmass,
    vvmass,
    weights,
    tagger,
    taggercuts,
    mlabel,
    tlabel,
    year,
    name: Path = None,
    show: bool = False,
):
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(20, 12),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0},
        sharex=True,
    )

    for i, (jet, mass) in enumerate(zip(["bb", "VV"], [bbmass, vvmass])):
        ax, rax = axs[0][i], axs[1][i]
        ax.set_prop_cycle(plt.rcParamsDefault["axes.prop_cycle"])
        hists = []

        for cut in taggercuts:
            if isinstance(cut, float):
                sel = tagger > cut
                hlabel = rf"{tlabel} > {cut}" if cut > 0 else "Inclusive"
            elif cut in txbb_wps[year]:
                sel = tagger > txbb_wps[year][cut]
                hlabel = rf"{tlabel} > {cut}"

            h = Hist(hist.axis.Regular(20, 50, 250, name="mass", label="mass"), storage="weight")
            # h = np.histogram(mass[sel], np.arange(50, 250, 10))
            h.fill(mass[sel], weight=weights[sel])
            h = h / np.sum(h.values())
            hists.append(h)

            hep.histplot(
                h,
                label=hlabel,
                ax=ax,
                histtype="step",
                yerr=True,
                linewidth=2,
                alpha=0.8,
            )

        if name is not None:
            (name.parent / "pickles").mkdir(exist_ok=True)
            with Path(name.parent / "pickles" / f"{jet}_{name.stem}.pkl").open("wb") as f:
                pickle.dump(hists, f)

        add_cms_label(ax, year, "Preliminary", loc=0)

        ax.set_ylabel("Normalized Events [A.U.]")
        ax.legend(fontsize=16)

        # do ratios
        rax.set_prop_cycle(plt.rcParamsDefault["axes.prop_cycle"][1:])  # skip first colour
        rax.hlines(1, 50, 250, linestyle="--", alpha=0.5)
        rax.grid(True, which="both", axis="y", linestyle="--", alpha=0.5)
        vals_nocut = hists[0].values()

        # next(rax.prop_cycler)  # skip first
        for j, cut in enumerate(taggercuts[1:]):
            hep.histplot(
                hists[j + 1] / vals_nocut,
                yerr=True,
                label=rf"{tlabel} > {cut}",
                ax=rax,
                histtype="errorbar",
            )

        rax.set_ylim([0, 2.2])
        rax.set_ylabel("Cut / Inclusive")
        rax.set_xlabel(rf"$m_\mathrm{{{mlabel}}}^{{{jet}}}$ [GeV]")

    if name is not None:
        plt.savefig(name, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plotMassSculptingAllYears(
    hists,
    taggercuts,
    mlabel,
    tlabel,
    name: Path = None,
    show: bool = False,
):
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(20, 12),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0},
        sharex=True,
    )

    for i, jet in enumerate(["bb", "VV"]):
        ax, rax = axs[0][i], axs[1][i]
        ax.set_prop_cycle(plt.rcParamsDefault["axes.prop_cycle"])

        for j, cut in enumerate(taggercuts):
            if isinstance(cut, float):
                hlabel = rf"{tlabel} > {cut}" if cut > 0 else "Inclusive"
            else:
                hlabel = rf"{tlabel} > {cut}"

            hep.histplot(
                hists[jet][j],
                label=hlabel,
                ax=ax,
                histtype="step",
                yerr=True,
                linewidth=2,
                alpha=0.8,
            )

        if name is not None:
            (name.parent / "pickles").mkdir(exist_ok=True)
            with Path(name.parent / "pickles" / f"{jet}_{name.stem}.pkl").open("wb") as f:
                pickle.dump(hists, f)

        add_cms_label(ax, "all", "Preliminary", loc=0)

        ax.set_ylabel("Normalized Events [A.U.]")
        ax.legend(fontsize=16)

        # do ratios
        rax.set_prop_cycle(plt.rcParamsDefault["axes.prop_cycle"][1:])  # skip first colour
        rax.hlines(1, 50, 250, linestyle="--", alpha=0.5)
        rax.grid(True, which="both", axis="y", linestyle="--", alpha=0.5)
        vals_nocut = hists[jet][0].values()

        # next(rax.prop_cycler)  # skip first
        for j, cut in enumerate(taggercuts[1:]):
            hep.histplot(
                hists[jet][j + 1] / vals_nocut,
                yerr=True,
                label=rf"{tlabel} > {cut}",
                ax=rax,
                histtype="errorbar",
            )

        rax.set_ylim([0, 2.2])
        rax.set_ylabel("Cut / Inclusive")
        rax.set_xlabel(rf"$m_\mathrm{{{mlabel}}}^{{{jet}}}$ [GeV]")

    if name is not None:
        plt.savefig(name, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_lund_plane(
    h: np.ndarray,
    title: str = "",
    ax=None,
    fig=None,
    name: str = "",
    log: bool = False,
    show: bool = False,
):
    from matplotlib.colors import LogNorm

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        assert fig is not None, "Must provide fig if providing ax."

    extent = [-1, 8, -5, 7]
    im = ax.imshow(
        h.T, origin="lower", extent=extent, cmap="viridis", norm=LogNorm() if log else None
    )
    ax.set_aspect("auto")
    fig.colorbar(im, ax=ax)
    # cbar.set_label('Density')

    ax.set_xlabel(r"ln$(0.8/\Delta)$")
    ax.set_ylabel(r"ln$(k_T/GeV)$")

    if len(title):
        ax.set_title(title, fontsize=24)

    plt.tight_layout()

    if ax is None:
        if len(name):
            plt.savefig(name, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()


def plot_lund_plane_six(
    hists: np.ndarray,
    edges: np.ndarray = None,
    name: str = "",
    log: bool = False,
    show: bool = False,
):
    if isinstance(hists, Hist):
        hists = hists.values()

    fig, axs = plt.subplots(2, 3, figsize=(20, 12), sharex=True, sharey=True)
    for i, ax in enumerate(axs.flat):
        plot_lund_plane(
            hists[i],
            title=(
                rf"$p_T$: [{edges[0][i]:.0f}, {edges[0][i + 1]:.0f}] GeV"
                if edges is not None
                else ""
            ),
            ax=ax,
            fig=fig,
            log=log,
        )

    if len(name):
        plt.savefig(name, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def XHYscatter2d(arr, label: str = None, name: str = "", show: bool = False):
    """Scatter plot of (mX, mY) plane for resonant analysis"""
    arr = np.array(arr)
    colours = np.ones(arr.shape[0]) if arr.shape[1] == 2 else arr[:, 2]

    fig, ax = plt.subplots(figsize=(14, 12))
    mappable = plt.scatter(arr[:, 0], arr[:, 1], s=150, c=colours, cmap="turbo")
    # plt.title(title)
    plt.xlabel(r"$m_X$ (GeV)")
    plt.ylabel(r"$m_Y$ (GeV)")
    plt.colorbar(mappable, label=label)

    add_cms_label(ax, "all", loc=0)

    if str(name):
        plt.savefig(name, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def scatter2d_limits(
    arr, label: str = None, name: str = "", show: bool = False, preliminary: bool = True
):
    """Scatter plot of limits in the  (mX, mY) plane for resonant analysis"""
    fig, ax = plt.subplots(figsize=(14, 12))
    mappable = plt.scatter(
        arr[:, 0],
        arr[:, 1],
        s=150,
        c=arr[:, 2],
        cmap="viridis",
        norm=mpl.colors.LogNorm(vmin=0.01, vmax=100),
    )
    plt.xlabel(r"$m_X$ (GeV)")
    plt.ylabel(r"$m_Y$ (GeV)")
    plt.colorbar(mappable, label=label)
    plt.savefig(name, bbox_inches="tight")

    add_cms_label(ax, "all", "Preliminary" if preliminary else None, loc=0)

    if show:
        plt.show()
    else:
        plt.close()


def scatter2d_overlay(
    arr,
    overlay_arr,
    label: str = None,
    name: str = "",
    show: bool = False,
    preliminary: bool = True,
):
    fig, ax = plt.subplots(figsize=(14, 12))
    mappable = ax.scatter(
        arr[:, 0],
        arr[:, 1],
        s=150,
        c=arr[:, 2],
        cmap="viridis",
        norm=mpl.colors.LogNorm(vmin=0.01, vmax=100),
    )
    _ = ax.scatter(
        overlay_arr[:, 0],
        overlay_arr[:, 1],
        s=300,
        marker="s",
        alpha=overlay_arr[:, 2],
        c=np.ones(overlay_arr.shape[0]),
        vmax=1,
    )
    plt.xlabel(r"$m_X$ (GeV)")
    plt.ylabel(r"$m_Y$ (GeV)")
    plt.colorbar(mappable, label=label)
    plt.savefig(name, bbox_inches="tight")

    add_cms_label(ax, "all", "Preliminary" if preliminary else None, loc=0)

    if show:
        plt.show()
    else:
        plt.close()


def colormesh(
    xx,
    yy,
    lims,
    label: str = None,
    name: str = "",
    show: bool = False,
    preliminary: bool = True,
    vmin=0.05,
    vmax=1e4,
    log: bool = True,
    region_labels: bool = False,
    figsize=(12, 8),
):
    fig, ax = plt.subplots(figsize=figsize)

    if log:
        pmesh_args = {"norm": mpl.colors.LogNorm(vmin=vmin, vmax=vmax)}
    else:
        pmesh_args = {"vmin": vmin, "vmax": vmax}

    pcol = plt.pcolormesh(xx, yy, lims, cmap="viridis", **pmesh_args)
    pcol.set_edgecolor("face")

    # plt.title(title)
    plt.xlabel(r"$m_X$ (GeV)")
    plt.ylabel(r"$m_Y$ (GeV)")
    plt.colorbar(label=label)

    if yy.max() > 2750:
        plt.ylim(60, 2780)
        cmsloc = 2
    else:
        cmsloc = 0

    if region_labels:
        # Draw diagonal line for signal region
        x = np.array([900, 4000])  # Wide x range to ensure line spans plot
        y = 0.1285 * x + 134.5
        plt.plot(x, y, "--", color="white", alpha=0.8, linewidth=2)

        plt.text(
            3550,
            2300,
            "Semi-resolved",
            color="white",
            fontsize=18,
            ha="center",
            va="center",
            fontproperties="Tex Gyre Heros",
        )
        plt.text(
            3550,
            250,
            "Fully-merged",
            color="white",
            fontsize=18,
            ha="center",
            va="center",
            fontproperties="Tex Gyre Heros",
        )

    add_cms_label(ax, "all", data=True, label="Preliminary" if preliminary else "", loc=cmsloc)
    plt.savefig(name, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_tf(
    tf: Hist,
    label: str = None,
    vmax: float = None,
    plot_dir: Path = None,
    name: str = "",
    data: bool = True,
    prelim: bool = True,
    show: bool = False,
):
    fig, ax = plt.subplots(1, 1, figsize=(11, 10))

    # h2d = hep.hist2dplot(
    #     pulls.values().T,
    #     hists.axes[2].edges,
    #     hists.axes[1].edges,
    #     cmap="viridis",
    #     cmin=-3.5,
    #     cmax=3.5,
    #     ax=ax,
    # )

    h2d = hep.hist2dplot(
        tf.values().T,
        tf.axes[1].edges,
        tf.axes[0].edges,
        ax=ax,
        cmap="viridis",
        cmax=vmax,
        flow="none",
    )
    h2d.pcolormesh.set_edgecolor("face")
    h2d.cbar.set_label(label)
    h2d.cbar.formatter.set_scientific(True)
    h2d.cbar.formatter.set_powerlimits((0, 0))
    add_cms_label(ax, year="all", data=data, loc=0, label="Preliminary" if prelim else None)

    xticks = [800, 1200, 1600, 2000, 3000, 4400]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.0f}" for x in xticks], rotation=45)
    ax.set_xlabel(tf.axes[1].label)
    ax.set_ylabel(tf.axes[0].label)

    if name:
        with (plot_dir / f"{name}.pkl").open("wb") as f:
            pickle.dump(tf, f)

        plt.savefig(plot_dir / f"{name}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
