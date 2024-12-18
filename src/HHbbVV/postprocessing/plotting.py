"""
Common plotting functions.

Author(s): Raghav Kansal
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
from hist import Hist
from hist.intervals import poisson_interval, ratio_uncertainty
from numpy.typing import ArrayLike
from pandas import DataFrame

from HHbbVV.hh_vars import LUMI, data_key, hbb_bg_keys
from HHbbVV.postprocessing import utils
from HHbbVV.postprocessing.utils import CUT_MAX_VAL

plt.style.use(hep.style.CMS)
hep.style.use("CMS")
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))

# this is needed for some reason to update the font size for the first plot
# fig, ax = plt.subplots(1, 1, figsize=(12, 12))
# plt.rcParams.update({"font.size": 24})
# plt.close()


bg_order = ["Diboson", "HH", "HWW", "Hbb", "ST", "W+Jets", "Z+Jets", "TT", "QCD"]

sample_label_map = {
    "HHbbVV": "ggF HHbbVV",
    "VBFHHbbVV": "VBF HHbbVV",
    "qqHH_CV_1_C2V_1_kl_1_HHbbVV": "VBF HHbbVV",
    "qqHH_CV_1_C2V_0_kl_1_HHbbVV": r"VBF HHbbVV ($\kappa_{2V} = 0$)",
    "qqHH_CV_1_C2V_2_kl_1_HHbbVV": r"VBF HHbbVV ($\kappa_{2V} = 2$)",
    "ST": r"Single-$t$",
    "TT": r"$t\bar{t}$",
}

colours = {
    "darkblue": "#1f78b4",
    "lightblue": "#a6cee3",
    "lightred": "#FF502E",
    "red": "#e31a1c",
    "darkred": "#A21315",
    "orange": "#ff7f00",
    "green": "#7CB518",
    "mantis": "#81C14B",
    "forestgreen": "#2E933C",
    "darkgreen": "#064635",
    "purple": "#9381FF",
    "darkpurple": "#7F2CCB",
    "slategray": "#63768D",
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

BG_COLOURS = {
    "QCD": "lightblue",
    "TT": "darkblue",
    "V+Jets": "green",
    "W+Jets": "green",
    "Z+Jets": "flax",
    "ST": "orange",
    "Diboson": "canary",
    "Hbb": "deeppurple",
    "HWW": "lightred",
    "HH": "ashgrey",
    "HHbbVV": "red",
    "qqHH_CV_1_C2V_0_kl_1_HHbbVV": "darkpurple",
}

sig_colour = "red"

SIG_COLOURS = [
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


def _combine_hbb_bgs(hists, bg_keys):
    # skip this if no hbb bg keys specified
    if len(set(bg_keys) & set(hbb_bg_keys)) == 0:
        return hists, bg_keys

    # combine all hbb backgrounds into a single "Hbb" background for plotting
    hbb_hists = []
    for key in hbb_bg_keys:
        if key in bg_keys:
            hbb_hists.append(hists[key, ...])
            bg_keys.remove(key)

    if "Hbb" not in bg_keys:
        bg_keys.append("Hbb")

    hbb_hist = sum(hbb_hists)

    # have to recreate hist with "Hbb" sample included
    h = Hist(
        hist.axis.StrCategory(list(hists.axes[0]) + ["Hbb"], name="Sample"),
        *hists.axes[1:],
        storage="double" if hists.storage_type == hist.storage.Double else "weight",
    )

    for i, sample in enumerate(hists.axes[0]):
        h.view()[i] = hists[sample, ...].view()

    h.view()[-1] = hbb_hist.view()

    return h, bg_keys


def _process_samples(sig_keys, bg_keys, bg_colours, sig_scale_dict, bg_order, syst, variation):
    # set up samples, colours and labels
    bg_keys = [key for key in bg_order if key in bg_keys]
    bg_colours = [colours[bg_colours[sample]] for sample in bg_keys]
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


def _divide_bin_widths(hists, data_err):
    """Divide histograms by bin widths"""
    edges = hists.axes[1].edges
    bin_widths = edges[1:] - edges[:-1]

    if data_err is None:
        data_err = (
            np.abs(poisson_interval(hists[data_key, ...]) - hists[data_key, ...]) / bin_widths
        )

    hists = hists / bin_widths[np.newaxis, :]
    return hists, data_err


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


def add_cms_label(ax, year, label="Preliminary"):
    if year == "all":
        hep.cms.label(
            label,
            data=True,
            lumi=f"{np.sum(list(LUMI.values())) / 1e3:.0f}",
            year=None,
            ax=ax,
        )
    else:
        hep.cms.label(label, data=True, lumi=f"{LUMI[year] / 1e3:.0f}", year=year, ax=ax)


def ratioHistPlot(
    hists: Hist,
    year: str,
    sig_keys: list[str],
    bg_keys: list[str],
    sig_colours: list[str] = None,
    bg_colours: dict[str, str] = None,
    sig_err: ArrayLike | str = None,
    bg_err: ArrayLike = None,
    data_err: ArrayLike | bool | None = None,
    title: str = None,
    name: str = "",
    sig_scale_dict: OrderedDict[str, float] = None,
    ylim: int = None,
    show: bool = True,
    syst: tuple = None,
    variation: str = None,
    bg_err_type: str = "shaded",
    plot_data: bool = True,
    bg_order: list[str] = bg_order,
    log: bool = False,
    ratio_ylims: list[float] = None,
    divide_bin_width: bool = False,
    plot_significance: bool = False,
    significance_dir: str = "right",
    plot_ratio: bool = True,
    axrax: tuple = None,
    ncol: int = None,
    cmslabel: str = None,
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
    if ncol is None:
        ncol = 2 if log else 1

    # copy hists and bg_keys so input objects are not changed
    hists, bg_keys = deepcopy(hists), deepcopy(bg_keys)
    hists, bg_keys = _combine_hbb_bgs(hists, bg_keys)

    bg_keys, bg_colours, bg_labels, sig_scale_dict, sig_labels = _process_samples(
        sig_keys, bg_keys, bg_colours, sig_scale_dict, bg_order, syst, variation
    )

    if syst is not None and variation is None:
        # plot up/down variations
        wshift, wsamples = syst
        sig_err = wshift  # will plot sig variations below
        bg_err = []
        for shift in ["down", "up"]:
            bg_sums = []
            for sample in bg_keys:
                if sample in wsamples and f"{sample}_{wshift}_{shift}" in hists.axes[0]:
                    bg_sums.append(hists[f"{sample}_{wshift}_{shift}", :].values())
                elif sample != "Hbb":
                    bg_sums.append(hists[sample, :].values())
            bg_err.append(np.sum(bg_sums, axis=0))

    pre_divide_hists = hists
    if divide_bin_width:
        hists, data_err = _divide_bin_widths(hists, data_err)

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
            2, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [3, 1], "hspace": 0}, sharex=True
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 11))

    plt.rcParams.update({"font.size": 24})

    # plot histograms
    y_label = r"Events / Bin Width (GeV$^{-1}$)" if divide_bin_width else "Events"
    ax.set_ylabel(y_label)

    # background samples
    hep.histplot(
        [hists[sample, :] for sample in bg_keys],
        ax=ax,
        histtype="fill",
        stack=True,
        label=bg_labels,
        color=bg_colours,
    )

    # signal samples
    if len(sig_scale_dict):
        hep.histplot(
            [hists[sig_key, :] * sig_scale for sig_key, sig_scale in sig_scale_dict.items()],
            ax=ax,
            histtype="step",
            label=list(sig_labels.values()),
            color=sig_colours[: len(sig_keys)],
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
        if divide_bin_width:
            raise NotImplementedError("Background error for divide bin width not checked yet")

        bg_tot = sum([pre_divide_hists[sample, :] for sample in bg_keys])
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
                label="Total Background Uncertainty",
            )
        else:
            ax.stairs(
                bg_tot.values(),
                hists.axes[1].edges,
                color="black",
                linewidth=3,
                label="BG Total",
                baseline=bg_tot.values(),
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
            histtype="errorbar",
            label=data_key,
            color="black",
        )

    if log:
        ax.set_yscale("log")
        # two column legend
        ax.legend(fontsize=20, ncol=2)
    else:
        ax.legend(fontsize=20, ncol=ncol)

    y_lowlim = 0 if not log else 1e-5
    if ylim is not None:
        ax.set_ylim([y_lowlim, ylim])
    else:
        ax.set_ylim(y_lowlim)

    # plot ratio below
    if plot_ratio:
        if plot_data:
            bg_tot = sum([pre_divide_hists[sample, :] for sample in bg_keys])
            # new: plotting data errors (black lines) and background errors (shaded) separately
            yerr = np.nan_to_num(
                np.abs(
                    poisson_interval(pre_divide_hists[data_key, ...])
                    - pre_divide_hists[data_key, ...]
                )
                / (bg_tot.values() + 1e-5)
            )

            # old version: using Garwood ratio intervals
            # yerr = ratio_uncertainty(
            #     pre_divide_hists[data_key, :].values(), bg_tot.values(), "poisson"
            # )

            hep.histplot(
                pre_divide_hists[data_key, :] / (bg_tot.values() + 1e-5),
                yerr=yerr,
                ax=rax,
                histtype="errorbar",
                color="black",
                capsize=4,
            )

            if bg_err is not None and bg_err_type == "shaded":
                # (bkg + err) / bkg
                rax.fill_between(
                    np.repeat(hists.axes[1].edges, 2)[1:-1],
                    np.repeat((bg_err[0].values()) / bg_tot, 2),
                    np.repeat((bg_err[1].values()) / bg_tot, 2),
                    color="black",
                    alpha=0.1,
                    hatch="//",
                    linewidth=0,
                )
        else:
            rax.set_xlabel(hists.axes[1].label)

        rax.set_ylabel("Data/MC")
        # rax.set_yscale("log")
        # formatter = mticker.ScalarFormatter(useOffset=False)
        # formatter.set_scientific(False)
        # rax.yaxis.set_major_formatter(formatter)
        rax.set_ylim(ratio_ylims)
        rax.grid()

    if plot_significance:
        bg_tot = sum([pre_divide_hists[sample, :] for sample in bg_keys]).values()
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

    add_cms_label(ax, year, label=cmslabel)

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
        2, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [3, 1], "hspace": 0}, sharex=True
    )

    bg_tot = np.sum(hists[bg_keys, :].values(), axis=0)
    plot_hists = [hists[sample, :] for sample in bg_keys]

    ax.set_ylabel("Events")
    hep.histplot(
        plot_hists + [sum(plot_hists)],
        ax=ax,
        histtype="step",
        label=bg_keys + ["Total"],
        color=[colours[bg_colours[sample]] for sample in bg_keys] + ["black"],
        yerr=0,
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

    # if sig_key in hists:
    #     hep.histplot(
    #         hists[sig_key, :] * sig_scale,
    #         ax=ax,
    #         histtype="step",
    #         label=f"{sig_key} $\\times$ {sig_scale:.1e}" if sig_scale != 1 else sig_key,
    #         color=colours[sig_colour],
    #     )

    hep.histplot(
        hists[data_key, :], ax=ax, yerr=data_err, histtype="errorbar", label=data_key, color="black"
    )

    if bg_err is not None:
        # Switch order so that uncertainty label comes at the end
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[1:] + handles[:1]
        labels = labels[1:] + labels[:1]
        ax.legend(handles, labels, ncol=2)
    else:
        ax.legend(ncol=2)

    ax.set_ylim(0, ax.get_ylim()[1] * 1.5)

    data_vals = hists[data_key, :].values()

    if not pulls:
        # datamc_ratio = data_vals / (bg_tot + 1e-5)

        # if bg_err == "ratio":
        #     yerr = ratio_uncertainty(data_vals, bg_tot, "poisson")
        # elif bg_err is None:
        #     yerr = 0
        # else:
        #     yerr = datamc_ratio * (bg_err / (bg_tot + 1e-8))

        yerr = ratio_uncertainty(data_vals, bg_tot, "poisson")

        hep.histplot(
            hists[data_key, :] / (sum([hists[sample, :] for sample in bg_keys]).values() + 1e-5),
            yerr=yerr,
            ax=rax,
            histtype="errorbar",
            color="black",
            capsize=4,
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

        rax.set_ylabel("Data/MC")
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
            color="black",
            capsize=4,
        )
        rax.set_ylabel("(Data - MC) / Data")
        rax.set_ylim(-0.5, 0.5)
        rax.grid()

    if title is not None:
        ax.set_title(title, y=1.08)

    add_cms_label(ax, year)

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
    region_labels: dict[str, str] = None,
    samples: list[str] = None,
    fail_zlim: float = None,
    pass_zlim: float = None,
    show: bool = True,
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
        hists[region]
        region_label = region_labels[region] if region_labels is not None else region
        pass_region = region.startswith("pass")
        lim = pass_zlim if pass_region else fail_zlim
        for sample in samples:
            if sample == "Data" and region == "pass":
                continue

            if "->H(bb)Y" in sample and not pass_region:
                continue

            if lim is not None:
                norm = mpl.colors.LogNorm(vmax=lim)
            else:
                norm = mpl.colors.LogNorm()

            plt.figure(figsize=(12, 12))
            hep.hist2dplot(hists[region][sample, ...], cmap="turbo", norm=norm)
            plt.title(f"{sample} in {region_label} Region")
            plt.savefig(f"{plot_dir}/{region}_{sample}_2d.pdf", bbox_inches="tight")

            if show:
                plt.show()
            else:
                plt.close()


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


def multiROCCurve(
    rocs: dict,
    thresholds=None,
    title=None,
    xlim=None,
    ylim=None,
    plot_dir="",
    name="",
    show=False,
):
    if ylim is None:
        ylim = [1e-06, 1]
    if xlim is None:
        xlim = [0, 1]
    if thresholds is None:
        thresholds = [[0.9, 0.98, 0.995, 0.9965, 0.998], [0.99, 0.997, 0.998, 0.999, 0.9997]]
    th_colours = [
        # "#36213E",
        "#9381FF",
        "#1f78b4",
        # "#a6cee3",
        # "#32965D",
        "#7CB518",
        "#EDB458",
        # "#ff7f00",
        "#a70000",
    ]

    roc_colours = ["#23CE6B", "#ff5252", "blue", "#ffbaba"]

    plt.rcParams.update({"font.size": 24})

    plt.figure(figsize=(12, 12))
    for i, roc_sigs in enumerate(rocs.values()):
        for j, roc in enumerate(roc_sigs.values()):
            if len(np.array(thresholds).shape) > 1:
                pthresholds = thresholds[j]
            else:
                pthresholds = thresholds

            plt.plot(
                roc["tpr"],
                roc["fpr"],
                label=roc["label"],
                linewidth=2,
                color=roc_colours[i * len(rocs) + j],
            )

            pths = {th: [[], []] for th in pthresholds}
            for th in pthresholds:
                idx = _find_nearest(roc["thresholds"], th)
                pths[th][0].append(roc["tpr"][idx])
                pths[th][1].append(roc["fpr"][idx])

            for k, th in enumerate(pthresholds):
                plt.scatter(
                    *pths[th],
                    marker="o",
                    s=80,
                    label=(
                        f"BDT Score > {th}"
                        if i == len(rocs) - 1  # and j == len(roc_sigs) - 1
                        else None
                    ),
                    color=th_colours[k],
                    zorder=100,
                )

                plt.vlines(
                    x=pths[th][0],
                    ymin=0,
                    ymax=pths[th][1],
                    color=th_colours[k],
                    linestyles="dashed",
                    alpha=0.5,
                )

                plt.hlines(
                    y=pths[th][1],
                    xmin=0,
                    xmax=pths[th][0],
                    color=th_colours[k],
                    linestyles="dashed",
                    alpha=0.5,
                )

    hep.cms.label(data=False, rlabel="")
    plt.yscale("log")
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background efficiency")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend(loc="lower right", fontsize=18)
    plt.title(title)

    if len(name):
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
            color=[colours[BG_COLOURS[sample]] for sample in training_keys],
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
        color=[colours[BG_COLOURS[sample]] for sample in training_keys],
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
