"""
Common plotting functions.

Author(s): Raghav Kansal
"""

import matplotlib.pyplot as plt
import mplhep as hep

plt.rcParams.update({"font.size": 16})
plt.style.use(hep.style.CMS)
hep.style.use("CMS")

from hist.intervals import ratio_uncertainty

from typing import Dict, List

sig_key = "HHbbVV"

colours = {"darkblue": "#1f78b4", "lightblue": "#a6cee3", "red": "#e31a1c", "orange": "#ff7f00"}
bg_colours = ["lightblue", "orange", "darkblue"]
sig_colour = "red"


def ratioHistPlot(
    hists: dict,
    bg_keys: List[str],
    bg_colours: list[str] = bg_colours,
    sig_colour: str = sig_colour,
    blind_region: list = None,
    name: str = "",
    sig_scale: float = 1.0,
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
        color=[colours[colour] for colour in bg_colours],
    )
    hep.histplot(
        hists[sig_key, :] * sig_scale,
        ax=ax,
        histtype="step",
        label=f"{sig_key} $\\times$ {sig_scale:.1e}" if sig_scale != 1 else sig_key,
        color=colours[sig_colour],
    )
    hep.histplot(hists["Data", :], ax=ax, histtype="errorbar", label="Data", color="black")
    ax.legend()
    ax.set_ylim(0)

    bg_tot = sum([hists[sample, :] for sample in bg_keys])
    yerr = ratio_uncertainty(hists["Data", :].values(), bg_tot.values(), "poisson")
    hep.histplot(
        hists["Data", :] / bg_tot, yerr=yerr, ax=rax, histtype="errorbar", color="black", capsize=4
    )
    rax.set_ylabel("Data/MC")
    rax.grid()

    hep.cms.label("Preliminary", data=True, lumi=40, year=2017, ax=ax)
    if len(name):
        plt.savefig(name, bbox_inches="tight")


def rocCurve(fpr, tpr, title=None, xlim=[0, 0.4], ylim=[1e-6, 1e-2], plotdir="", name=""):
    """Plots a ROC curve"""
    plt.figure(figsize=(12, 12))
    plt.plot(tpr, fpr)
    plt.yscale("log")
    plt.xlabel("Signal Eff.")
    plt.ylabel("BG Eff.")
    plt.title(title)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.savefig(f"{plotdir}/{name}.pdf", bbox_inches="tight")
