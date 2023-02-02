"""
Common plotting functions.

Author(s): Raghav Kansal
"""

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib.ticker as mticker

plt.style.use(hep.style.CMS)
hep.style.use("CMS")
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))
plt.rcParams.update({"font.size": 16})

from hist import Hist
from hist.intervals import ratio_uncertainty

from typing import Dict, List, Union
from numpy.typing import ArrayLike

from sample_labels import sig_key, data_key

colours = {
    "darkblue": "#1f78b4",
    "lightblue": "#a6cee3",
    "red": "#e31a1c",
    "orange": "#ff7f00",
    "green": "#7CB518",
    "darkgreen": "#064635",
    "darkred": "#990000",
}
bg_colours = {"QCD": "lightblue", "TT": "darkblue", "ST": "orange"}

bg_colours = {
    "QCD": "lightblue",
    "Single Top": "darkblue",
    "TT Unmatched": "darkgreen",
    "TT W Matched": "green",
    "TT Top Matched": "orange",
    "W+Jets": "darkred",
    "Diboson": "red",
}

sig_colour = "red"

# from https://cds.cern.ch/record/2724492/files/DP2020_035.pdf
LUMI = {"2016APV": 20e3, "2016": 16e3, "2017": 41e3, "2018": 59e3}  # in pb^-1


def ratioHistPlot(
    hists: Hist,
    bg_keys: List[str],
    year: str,
    bg_colours: Dict[str, str] = bg_colours,
    sig_colour: str = sig_colour,
    bg_err: np.ndarray | str = None,
    data_err: Union[ArrayLike, bool, None] = None,
    title: str = None,
    blind_region: list = None,
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

    ax.set_ylabel("Events")
    hep.histplot(
        [hists[sample, :] for sample in bg_keys],
        ax=ax,
        histtype="fill",
        stack=True,
        label=bg_keys,
        color=[colours[bg_colours[sample]] for sample in bg_keys],
    )

    bg_tot = np.sum(hists[bg_keys, :].values(), axis=0)

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

    mcdata_ratio = bg_tot / (hists[data_key, :].values() + 1e-5)

    if bg_err == "ratio":
        yerr = ratio_uncertainty(bg_tot, hists[data_key, :].values(), "poisson")
    elif bg_err is None:
        yerr = 0
    else:
        yerr = bg_err / (hists[data_key, :].values() + 1e-5)

    hep.histplot(
        sum([hists[sample, :] for sample in bg_keys]) / (hists[data_key, :].values() + 1e-5),
        yerr=yerr,
        ax=rax,
        histtype="errorbar",
        color="black",
        capsize=4,
    )
    rax.set_ylabel("MC/Data")
    rax.set_ylim(0.5, 1.5)
    rax.grid()

    if title is not None:
        ax.set_title(title, y=1.08)

    hep.cms.label("Work in Progress", data=True, lumi=LUMI[year] * 1e-3, year=year, ax=ax)
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
    bg_err: np.ndarray | str = None,
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

    hep.cms.label("Work in Progress", data=True, lumi=LUMI[year] * 1e-3, year=year, ax=ax)
    if len(name):
        plt.savefig(name, bbox_inches="tight")

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
