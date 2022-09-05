"""
Common plotting functions.

Author(s): Raghav Kansal
"""

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib.ticker as mticker

plt.rcParams.update({"font.size": 16})
plt.style.use(hep.style.CMS)
hep.style.use("CMS")
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))

from hist import Hist
from hist.intervals import ratio_uncertainty

from typing import Dict, List
from numpy.typing import ArrayLike

from sample_labels import sig_key, data_key

colours = {"darkblue": "#1f78b4", "lightblue": "#a6cee3", "red": "#e31a1c", "orange": "#ff7f00"}
bg_colours = {"QCD": "lightblue", "TT": "darkblue", "ST": "orange"}
sig_colour = "red"


def ratioHistPlot(
    hists: Hist,
    bg_keys: List[str],
    bg_colours: Dict[str, str] = bg_colours,
    sig_colour: str = sig_colour,
    data_err: ArrayLike | bool | None = None,
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
    ax.legend()
    ax.set_ylim(0)

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
    rax.set_ylabel("Data/MC")
    rax.grid()

    if title is not None:
        ax.set_title(title, y=1.08)

    hep.cms.label("Work in Progress", data=True, lumi=40, year=2017, ax=ax)
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
