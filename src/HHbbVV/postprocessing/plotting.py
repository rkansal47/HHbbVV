"""
Common plotting functions.

Author(s): Raghav Kansal
"""

from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib.ticker as mticker
import matplotlib

plt.style.use(hep.style.CMS)
hep.style.use("CMS")
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))
plt.rcParams.update({"font.size": 20})

from hist import Hist
from hist.intervals import ratio_uncertainty

from typing import Dict, List, Union, Tuple
from numpy.typing import ArrayLike

from hh_vars import LUMI, data_key

from copy import deepcopy

colours = {
    "darkblue": "#1f78b4",
    "lightblue": "#a6cee3",
    "lightred": "#FF502E",
    "red": "#e31a1c",
    "darkred": "#A21315",
    "orange": "#ff7f00",
    "green": "#7CB518",
    "purple": "#9381FF",
    "slategray": "#63768D",
    "deeppurple": "#36213E",
    "ashgrey": "#ACBFA4",
}
bg_colours = {
    "QCD": "lightblue",
    "TT": "darkblue",
    "V+Jets": "green",
    "ST": "orange",
    "Diboson": "purple",
    "Hbb": "lightred",
    "HWW": "deeppurple",
    "HH": "ashgrey",
}
sig_colour = "red"

sig_colours = [
    "#23CE6B",
    "#ffbaba",
    "#ff7b7b",
    "#ff5252",
    # "#EDB458",
    "#a70000",
    "#885053",
    "#3C0919",
]

bg_order = ["Diboson", "HH", "HWW", "Hbb", "ST", "V+Jets", "TT", "QCD"]


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


def ratioHistPlot(
    hists: Hist,
    year: str,
    sig_keys: List[str],
    bg_keys: List[str],
    sig_colours: Dict[str, str] = sig_colours,
    bg_colours: Dict[str, str] = bg_colours,
    sig_err: Union[ArrayLike, str] = None,
    data_err: Union[ArrayLike, bool, None] = None,
    title: str = None,
    blind_region: list = None,
    name: str = "",
    sig_scale_dict: Dict[str, float] = None,
    ylim: int = None,
    show: bool = True,
    variation: Tuple = None,
    plot_data: bool = True,
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
    """
    bg_keys = [key for key in bg_order if key in bg_keys]
    bg_colours = [colours[bg_colours[sample]] for sample in bg_keys]
    bg_labels = deepcopy(bg_keys)

    if sig_scale_dict is None:
        sig_scale_dict = OrderedDict([(sig_key, 1.0) for sig_key in sig_keys])
    else:
        sig_scale_dict = deepcopy(sig_scale_dict)

    sig_labels = OrderedDict(
        [
            (sig_key, f"{sig_key} $\\times$ {sig_scale:.1e}" if sig_scale != 1 else sig_key)
            for sig_key, sig_scale in sig_scale_dict.items()
        ]
    )

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

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(12, 14), gridspec_kw=dict(height_ratios=[3, 1], hspace=0), sharex=True
    )

    ax.set_ylabel("Events")
    hep.histplot(
        [hists[sample, :] for sample in bg_keys],
        ax=ax,
        histtype="fill",
        stack=True,
        label=bg_labels,
        color=bg_colours,
    )

    if len(sig_scale_dict):
        hep.histplot(
            [hists[sig_key, :] * sig_scale for sig_key, sig_scale in sig_scale_dict.items()],
            ax=ax,
            histtype="step",
            label=list(sig_labels.values()),
            color=sig_colours[: len(sig_keys)],
        )

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

    if plot_data:
        hep.histplot(
            hists[data_key, :],
            ax=ax,
            yerr=data_err,
            histtype="errorbar",
            label=data_key,
            color="black",
        )
    ax.legend()

    if ylim is not None:
        ax.set_ylim([0, ylim])
    else:
        ax.set_ylim(0)

    if plot_data:
        bg_tot = sum([hists[sample, :] for sample in bg_keys])
        yerr = ratio_uncertainty(hists[data_key, :].values(), bg_tot.values(), "poisson")

        hep.histplot(
            hists[data_key, :] / (bg_tot.values() + 1e-5),
            yerr=yerr,
            ax=rax,
            histtype="errorbar",
            color="black",
            capsize=4,
        )
    else:
        rax.set_xlabel(hists.axes[1].label)

    rax.set_ylabel("Data/MC")
    rax.set_ylim([0, 2])
    rax.grid()

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
                norm = matplotlib.colors.LogNorm(vmax=lim)
            else:
                norm = matplotlib.colors.LogNorm()

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
    auc,
    sig_eff_lines=[],
    # bg_eff_lines=[],
    title=None,
    xlim=[0, 0.4],
    ylim=[1e-6, 1e-2],
    plotdir="",
    name="",
):
    """Plots a ROC curve"""
    line_style = {"colors": "lightgrey", "linestyles": "dashed"}

    plt.figure(figsize=(12, 12))
    plt.plot(tpr, fpr, label=f"AUC: {auc:.2f}")

    for sig_eff in sig_eff_lines:
        y = fpr[np.searchsorted(tpr, sig_eff)]
        plt.hlines(y=y, xmin=0, xmax=sig_eff, **line_style)
        plt.vlines(x=sig_eff, ymin=0, ymax=y, **line_style)

    plt.yscale("log")
    plt.xlabel("Signal Eff.")
    plt.ylabel("BG Eff.")
    plt.title(title)
    plt.legend()
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.savefig(f"{plotdir}/{name}.pdf", bbox_inches="tight")
