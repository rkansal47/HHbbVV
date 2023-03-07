"""
Common plotting functions.

Author(s): Raghav Kansal
"""

from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib.ticker as mticker

plt.style.use(hep.style.CMS)
hep.style.use("CMS")
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))
plt.rcParams.update({"font.size": 20})

from hist import Hist
from hist.intervals import ratio_uncertainty

from typing import Dict, List, Union, Tuple
from numpy.typing import ArrayLike

from hh_vars import LUMI, sig_keys, res_sig_keys, data_key

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
}
bg_colours = {
    "QCD": "lightblue",
    "TT": "darkblue",
    "V+Jets": "green",
    "ST": "orange",
    "Diboson": "purple",
}
sig_colour = "red"

sig_colours = [
    "#23CE6B",
    "#ffbaba",
    "#ff7b7b",
    "#ff5252",
    "#EDB458",
    "#a70000",
]

bg_order = ["Diboson", "ST", "V+Jets", "TT", "QCD"]


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
    bg_colours: Dict[str, str] = bg_colours,
    sig_colours: str = sig_colours,
    sig_err: Union[ArrayLike, str] = None,
    data_err: Union[ArrayLike, bool, None] = None,
    title: str = None,
    blind_region: list = None,
    name: str = "",
    sig_scale_dict: float = None,
    show: bool = True,
    variation: Tuple = None,
):
    """
    Makes and saves a histogram plot, with backgrounds stacked, signal separate (and optionally
    scaled) with a data/mc ratio plot below
    """

    bg_keys = [key for key in bg_order if key in bg_keys]
    bg_colours = [colours[bg_colours[sample]] for sample in bg_keys]
    bg_labels = deepcopy(bg_keys)

    if sig_scale_dict is None:
        sig_scale_dict = OrderedDict([(sig_key, 1.0) for sig_key in sig_keys])

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

        if sig_key in wsamples:
            sig_key = f"{sig_key}_{wshift}_{shift}"
            sig_label += skey

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
    hep.histplot(
        [hists[sig_key, :] * sig_scale for sig_key, sig_scale in sig_scale_dict.items()],
        ax=ax,
        histtype="step",
        label=list(sig_labels.values()),
        color=sig_colours,
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
                label=f"{sig_key} {skey}",
                color=scolours[shift],
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
