"""
Common plotting functions.

Author(s): Raghav Kansal
"""

from collections import OrderedDict
import numpy as np
from pandas import DataFrame

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib.ticker as mticker

plt.style.use(hep.style.CMS)
hep.style.use("CMS")
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))

# this is needed for some reason to update the font size for the first plot
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
plt.rcParams.update({"font.size": 24})
plt.close()

import hist
from hist import Hist
from hist.intervals import ratio_uncertainty, poisson_interval

from typing import Dict, List, Union, Tuple
from numpy.typing import ArrayLike

from hh_vars import LUMI, data_key, hbb_bg_keys
import utils
from utils import CUT_MAX_VAL

from copy import deepcopy


bg_order = ["Diboson", "HH", "HWW", "Hbb", "ST", "V+Jets", "TT", "QCD"]

sample_label_map = {
    "HHbbVV": "ggF HHbbVV",
    "VBFHHbbVV": "VBF HHbbVV",
    "qqHH_CV_1_C2V_0_kl_1_HHbbVV": r"VBF HHbbVV ($\kappa_{2V} = 0$)",
    "qqHH_CV_1_C2V_2_kl_1_HHbbVV": r"VBF HHbbVV ($\kappa_{2V} = 2$)",
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

bg_colours = {
    "QCD": "lightblue",
    "TT": "darkblue",
    "V+Jets": "green",
    "ST": "orange",
    "Diboson": "canary",
    "Hbb": "lightred",
    "HWW": "deeppurple",
    "HH": "ashgrey",
    "HHbbVV": "red",
}

sig_colour = "red"

sig_colours = [
    "#23CE6B",
    "#7F2CCB",
    "#ffbaba",
    "#ff7b7b",
    "#ff5252",
    "#a70000",
    "#885053",
    "#3C0919",
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
        storage="weight",
    )

    for i, sample in enumerate(hists.axes[0]):
        h.view()[i] = hists[sample, ...].view()

    h.view()[-1] = hbb_hist

    return h, bg_keys


def _process_samples(sig_keys, bg_keys, bg_colours, sig_scale_dict, variation):
    # set up samples, colours and labels
    bg_keys = [key for key in bg_order if key in bg_keys]
    bg_colours = [colours[bg_colours[sample]] for sample in bg_keys]
    bg_labels = deepcopy(bg_keys)

    if sig_scale_dict is None:
        sig_scale_dict = OrderedDict([(sig_key, 1.0) for sig_key in sig_keys])
    else:
        sig_scale_dict = deepcopy(sig_scale_dict)

    sig_labels = OrderedDict()
    for sig_key, sig_scale in sig_scale_dict.items():
        label = sig_key if sig_key not in sample_label_map else sample_label_map[sig_key]

        if sig_scale == 1:
            label = label
        elif sig_scale <= 100:
            label = f"{label} $\\times$ {sig_scale:.0f}"
        else:
            label = f"{label} $\\times$ {sig_scale:.1e}"

        sig_labels[sig_key] = label

    # set up systematic variations if needed
    if variation is not None:
        wshift, shift, wsamples = variation
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


def ratioHistPlot(
    hists: Hist,
    year: str,
    sig_keys: List[str],
    bg_keys: List[str],
    sig_colours: List[str] = sig_colours,
    bg_colours: Dict[str, str] = bg_colours,
    sig_err: Union[ArrayLike, str] = None,
    data_err: Union[ArrayLike, bool, None] = None,
    title: str = None,
    blind_region: list = None,
    name: str = "",
    sig_scale_dict: OrderedDict[str, float] = None,
    ylim: int = None,
    show: bool = True,
    variation: Tuple = None,
    plot_data: bool = True,
    bg_order: List[str] = bg_order,
    log: bool = False,
    ratio_ylims: List[float] = [0, 2],
    divide_bin_width: bool = False,
    plot_significance: bool = False,
    significance_dir: str = "right",
    axrax: Tuple = None,
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
        data_err (Union[ArrayLike, bool, None], optional): plot error on data.
          if True, will plot poisson error per bin
          if array, will plot given errors per bin
        title (str, optional): plot title. Defaults to None.
        blind_region (list): [min, max] range of values which should be blinded in the plot
          i.e. Data set to 0 in those bins
        name (str): name of file to save plot
        sig_scale_dict (Dict[str, float]): if scaling signals in the plot, dictionary of factors
          by which to scale each signal
        ylim (optional): y-limit on plot
        show (bool): show plots or not
        variation (Tuple): Tuple of
          (wshift: name of systematic e.g. pileup, shift: up or down, wsamples: list of samples which are affected by this)
        plot_data (bool): plot data
        bg_order (List[str]): order in which to plot backgrounds
        ratio_ylims (List[float]): y limits on the ratio plots
        divide_bin_width (bool): divide yields by the bin width (for resonant fit regions)
        plot_significance (bool): plot Asimov significance below ratio plot
        significance_dir (str): "Direction" for significance. i.e. a > cut ("right"), a < cut ("left"), or per-bin ("bin").
        axrax (Tuple): optionally input ax and rax instead of creating new ones
    """
    # copy hists and bg_keys so input objects are not changed
    hists, bg_keys = deepcopy(hists), deepcopy(bg_keys)
    hists, bg_keys = _combine_hbb_bgs(hists, bg_keys)

    bg_keys, bg_colours, bg_labels, sig_scale_dict, sig_labels = _process_samples(
        sig_keys, bg_keys, bg_colours, sig_scale_dict, variation
    )

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
            3, 1, figsize=(12, 18), gridspec_kw=dict(height_ratios=[3, 1, 1], hspace=0), sharex=True
        )
    else:
        fig, (ax, rax) = plt.subplots(
            2, 1, figsize=(12, 14), gridspec_kw=dict(height_ratios=[3, 1], hspace=0), sharex=True
        )

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
    if type(sig_err) == str:
        scolours = {"down": colours["lightred"], "up": colours["darkred"]}
        for skey, shift in [("Up", "up"), ("Down", "down")]:
            hep.histplot(
                [
                    hists[f"{sig_key}_{sig_err}_{shift}", :] * sig_scale
                    for sig_key, sig_scale in sig_scale_dict.items()
                ],
                yerr=0,
                ax=ax,
                histtype="step",
                label=[f"{sig_key} {skey}" for sig_key in sig_scale_dict],
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

    ax.legend()

    y_lowlim = 0 if not log else 1e-3
    if ylim is not None:
        ax.set_ylim([y_lowlim, ylim])
    else:
        ax.set_ylim(y_lowlim)

    # plot ratio below
    if plot_data:
        bg_tot = sum([pre_divide_hists[sample, :] for sample in bg_keys])
        yerr = ratio_uncertainty(pre_divide_hists[data_key, :].values(), bg_tot.values(), "poisson")

        hep.histplot(
            pre_divide_hists[data_key, :] / (bg_tot.values() + 1e-5),
            yerr=yerr,
            ax=rax,
            histtype="errorbar",
            color="black",
            capsize=4,
        )
    else:
        rax.set_xlabel(hists.axes[1].label)

    rax.set_ylabel("Data/MC")
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
            label=[
                sig_key if sig_key not in sample_label_map else sample_label_map[sig_key]
                for sig_key in sig_scale_dict
            ],
            color=sig_colours[: len(sig_keys)],
        )

        sax.legend(fontsize=12)
        sax.set_yscale("log")
        sax.set_ylim([1e-7, 10])
        sax.set_xlabel(hists.axes[1].label)

    if title is not None:
        ax.set_title(title, y=1.08)

    if year == "all":
        hep.cms.label(
            "Work in Progress",
            data=True,
            lumi=f"{np.sum(list(LUMI.values())) / 1e3:.0f}",
            year=None,
            ax=ax,
        )
    else:
        hep.cms.label(
            "Work in Progress", data=True, lumi=f"{LUMI[year] / 1e3:.0f}", year=year, ax=ax
        )

    if axrax is None:
        if len(name):
            plt.savefig(name, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()


def ratioLinePlot(
    hists: Hist,
    bg_keys: List[str],
    year: str,
    bg_colours: Dict[str, str] = bg_colours,
    sig_colour: str = sig_colour,
    bg_err: Union[np.ndarray, str] = None,
    data_err: Union[ArrayLike, bool, None] = None,
    title: str = None,
    blind_region: list = None,
    pulls: bool = False,
    name: str = "",
    sig_scale: float = 1.0,
    show: bool = True,
):
    """
    Makes and saves a histogram plot, with backgrounds stacked, signal separate (and optionally
    scaled) with a data/mc ratio plot below
    """
    plt.rcParams.update({"font.size": 24})

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(12, 14), gridspec_kw=dict(height_ratios=[3, 1], hspace=0), sharex=True
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
        )

    if sig_key in hists:
        hep.histplot(
            hists[sig_key, :] * sig_scale,
            ax=ax,
            histtype="step",
            label=f"{sig_key} $\\times$ {sig_scale:.1e}" if sig_scale != 1 else sig_key,
            color=colours[sig_colour],
        )
    hep.histplot(
        hists[data_key, :], ax=ax, yerr=data_err, histtype="errorbar", label=data_key, color="black"
    )
    ax.legend(ncol=2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.5)

    data_vals = hists[data_key, :].values()

    if not pulls:
        datamc_ratio = data_vals / (bg_tot + 1e-5)

        if bg_err == "ratio":
            yerr = ratio_uncertainty(data_vals, bg_tot, "poisson")
        elif bg_err is None:
            yerr = 0
        else:
            yerr = datamc_ratio * (bg_err / (bg_tot + 1e-8))

        hep.histplot(
            hists[data_key, :] / (sum([hists[sample, :] for sample in bg_keys]).values() + 1e-5),
            yerr=yerr,
            ax=rax,
            histtype="errorbar",
            color="black",
            capsize=4,
        )
        rax.set_ylabel("Data/MC")
        rax.set_ylim(0.5, 1.5)
        rax.grid()
    else:
        mcdata_ratio = bg_tot / (data_vals + 1e-5)
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

    hep.cms.label("Work in Progress", data=True, lumi=round(LUMI[year] * 1e-3), year=year, ax=ax)
    if len(name):
        plt.savefig(name, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def hist2ds(
    hists: Dict[str, Hist],
    plot_dir: str,
    regions: List[str] = None,
    region_labels: Dict[str, str] = None,
    samples: List[str] = None,
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
        samples = list(list(hists.values())[0].axes[0])

    if regions is None:
        regions = list(hists.keys())

    for region in regions:
        h = hists[region]
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


def rocCurve(
    fpr,
    tpr,
    auc=None,
    sig_eff_lines=[],
    # bg_eff_lines=[],
    title=None,
    xlim=[0, 0.8],
    ylim=[1e-6, 1],
    plotdir="",
    name="",
):
    """Plots a ROC curve"""
    line_style = {"colors": "lightgrey", "linestyles": "dashed"}

    plt.figure(figsize=(12, 12))

    plt.plot(tpr, fpr, label=f"AUC: {auc:.2f}" if auc is not None else None)

    for sig_eff in sig_eff_lines:
        y = fpr[np.searchsorted(tpr, sig_eff)]
        plt.hlines(y=y, xmin=0, xmax=sig_eff, **line_style)
        plt.vlines(x=sig_eff, ymin=0, ymax=y, **line_style)

    plt.yscale("log")
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background efficiency")
    plt.title(title)

    if auc is not None:
        plt.legend()

    plt.xlim(*xlim)
    plt.ylim(*ylim)
    hep.cms.label(data=False, rlabel="")
    plt.savefig(f"{plotdir}/{name}.pdf", bbox_inches="tight")


def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def multiROCCurve(
    rocs: Dict,
    thresholds=[0.6, 0.9, 0.96, 0.99, 0.997, 0.998, 0.999],
    title=None,
    xlim=[0, 1],
    ylim=[1e-6, 1],
    plotdir="",
    name="",
    show=False,
):
    th_colours = [
        "#36213E",
        "#9381FF",
        "#1f78b4",
        # "#a6cee3",
        # "#32965D",
        "#7CB518",
        "#EDB458",
        "#ff7f00",
        "#a70000",
    ]

    roc_colours = ["blue", "#23CE6B"][-len(rocs) :]

    plt.rcParams.update({"font.size": 24})

    plt.figure(figsize=(12, 12))
    for i, roc in enumerate(rocs.values()):
        plt.plot(
            roc["tpr"],
            roc["fpr"],
            label=roc["label"] if len(rocs) > 1 else None,
            linewidth=2,
            color=roc_colours[i],
        )

        pths = {th: [[], []] for th in thresholds}
        for th in thresholds:
            idx = _find_nearest(roc["thresholds"], th)
            pths[th][0].append(roc["tpr"][idx])
            pths[th][1].append(roc["fpr"][idx])

        for k, th in enumerate(thresholds):
            plt.scatter(
                *pths[th],
                marker="o",
                s=40,
                label=f"BDT Score > {th}" if i == len(rocs) - 1 else None,
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
    plt.legend(loc="lower right")

    if len(name):
        plt.savefig(f"{plotdir}/{name}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_HEM2d(hists2d: List[Hist], plot_keys: List[str], year: str, name: str, show: bool = False):
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
            hep.cms.label(
                "Work in Progress", data=True, lumi=round(LUMI[year] * 1e-3), year=year, ax=ax
            )
            ax.set_title(key, y=1.07)
            ax._children[0].colorbar.set_label("Events")

    plt.savefig(name, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def ratioTestTrain(
    h: Hist,
    training_keys: List[str],
    shape_var: utils.ShapeVar,
    year: str,
    plotdir="",
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
        2, 1, figsize=(12, 14), gridspec_kw=dict(height_ratios=[3, 1], hspace=0), sharex=True
    )

    for data in ["Train", "Test"]:
        plot_hists = [h[data, sample, :] for sample in training_keys]

        ax.set_ylabel("Events")
        hep.histplot(
            plot_hists,
            ax=ax,
            histtype="step",
            label=[data + " " + key for key in training_keys],
            color=[colours[bg_colours[sample]] for sample in training_keys],
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
        label=training_keys,
        color=[colours[bg_colours[sample]] for sample in training_keys],
        yerr=np.abs([err[i] * plot_hists[i].values() for i in range(len(plot_hists))]),
    )

    rax.set_ylim([0, 2])
    rax.set_xlabel(shape_var.label)
    rax.set_ylabel("Train / Test")
    rax.legend(fontsize=20, loc="upper left", ncol=3)
    rax.grid()

    hep.cms.label(data=False, year=year, ax=ax, lumi=f"{LUMI[year] / 1e3:.0f}")

    if len(name):
        plt.savefig(f"{plotdir}/{name}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def cutsLinePlot(
    events_dict: Dict[str, DataFrame],
    bb_masks: Dict[str, DataFrame],
    shape_var: utils.ShapeVar,
    plot_key: str,
    cut_var: str,
    cuts: List[float],
    year: str,
    weight_key: str,
    plotdir: str = "",
    name: str = "",
    show: bool = False,
):
    """Plot line plots of ``shape_var`` for different cuts on ``cut_var``."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    plt.rcParams.update({"font.size": 24})

    for i, cut in enumerate(cuts):
        sel, _ = utils.make_selection({cut_var: [cut, CUT_MAX_VAL]}, events_dict, bb_masks)
        h = utils.singleVarHist(
            events_dict, shape_var, bb_masks, weight_key=weight_key, selection=sel
        )

        hep.histplot(
            h[plot_key, ...] / np.sum(h[plot_key, ...].values()),
            yerr=True,
            label=f"BDTScore >= {cut}",
            # density=True,
            ax=ax,
            linewidth=2,
            alpha=0.8,
        )

    ax.set_xlabel(shape_var.label)
    ax.set_ylabel("Fraction of Events")
    ax.legend()

    hep.cms.label(ax=ax, data=False, year=year, lumi=round(LUMI[year] / 1e3))

    if len(name):
        plt.savefig(f"{plotdir}/{name}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
