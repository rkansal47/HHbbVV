from __future__ import annotations

import matplotlib.pyplot as plt
import mplhep as hep

plt.rcParams.update({"font.size": 16})
plt.style.use(hep.style.CMS)
hep.style.use("CMS")


colours = {"darkblue": "#1f78b4", "lightblue": "#a6cee3", "red": "#e31a1c", "orange": "#ff7f00"}

bg_colours = ["lightblue", "orange", "darkblue"]
sig_colour = "red"


def singleHistPlot(
    data,
    weights=None,
    bins=None,
    xlabel="",
    ylabel="# Events",
    title="",
    plotdir="",
    name="",
    xlim=None,
    ylim=None,
    **histkwargs,
):
    """Makes and saves a single histogram plot"""
    plt.figure(figsize=(12, 12))
    plt.hist(data.reshape(-1), bins=bins, weights=weights, histtype="step", **histkwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(0, xlim)
    plt.ylim(0, ylim)
    if len(name):
        plt.savefig(f"{plotdir}{name}.pdf", bbox_inches="tight")


def multiHistPlot(
    data,
    labels,
    weights=None,
    bins=None,
    xlabel="",
    ylabel="# Events",
    title="",
    plotdir="",
    name="",
    xlim=None,
    ylim=None,
):
    """Makes and saves a plot with multiple histograms, `data` is a list of data to plot"""
    plt.figure(figsize=(12, 12))
    for i in range(len(data)):
        plt.hist(
            data[i].reshape(-1),
            bins=bins,
            weights=weights[i] if type(weights) == list else weights,
            histtype="step",
            label=labels[i],
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(prop={"size": 18})
    plt.xlim(0, xlim)
    plt.ylim(0, ylim)
    if len(name):
        plt.savefig(f"{plotdir}{name}.pdf", bbox_inches="tight")


def multiHistCutsPlot(
    data,
    cuts,
    labels,
    weights=None,
    bins=None,
    xlabel="",
    ylabel="# Events",
    title="",
    plotdir="",
    name="",
    xlim=None,
    ylim=None,
):
    """
    Makes and saves a plot with multiple histograms, all from the same data array but with
    different labels and cuts. None cut means all data is plotted
    """
    plt.figure(figsize=(12, 12))
    for i in range(len(cuts)):
        plt.hist(
            data[cuts[i]].reshape(-1) if cuts[i] is not None else data.reshape(-1),
            bins=bins,
            weights=(
                weights[cuts[i]].reshape(-1)
                if cuts[i] is not None and weights is not None
                else (weights.reshape(-1) if weights is not None else None)
            ),
            histtype="step",
            label=labels[i],
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(prop={"size": 18})
    plt.xlim(0, xlim)
    plt.ylim(0, ylim)
    if len(name):
        plt.savefig(f"{plotdir}{name}.pdf", bbox_inches="tight")


from hist.intervals import ratio_uncertainty


def ratioHistPlot(
    hists: dict,
    bg_keys: list[str],
    sig_key: str,
    bg_labels: list[str] = None,
    sig_label: str = None,
    bg_colours: list[str] = bg_colours,
    sig_colour: str = sig_colour,
    blind_region: list = None,
    plotdir: str = "",
    name: str = "",
    sig_scale: float = 1.0,
):
    """
    Makes and saves a histogram plot, with backgrounds stacked, signal separate (and optionally
    scaled) with a data/mc ratio plot below
    """
    if sig_label is None:
        sig_label = sig_key

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [3, 1], "hspace": 0}, sharex=True
    )

    ax.set_ylabel("Events")
    hep.histplot(
        [hists[key, :] for key in bg_keys],
        ax=ax,
        histtype="fill",
        stack=True,
        label=bg_labels,
        color=[colours[colour] for colour in bg_colours],
    )
    hep.histplot(
        hists[sig_key, :] * sig_scale,
        ax=ax,
        histtype="step",
        label=f"{sig_label} $\\times$ {sig_scale:.1e}" if sig_scale != 1 else sig_label,
        color=colours[sig_colour],
    )
    hep.histplot(hists["Data", :], ax=ax, histtype="errorbar", label="Data", color="black")
    ax.legend()
    ax.set_ylim(0)

    bg_tot = sum([hists[key, :] for key in bg_keys])
    yerr = ratio_uncertainty(hists["Data", :].values(), bg_tot.values(), "poisson")
    hep.histplot(
        hists["Data", :] / bg_tot, yerr=yerr, ax=rax, histtype="errorbar", color="black", capsize=4
    )
    rax.set_ylabel("Data/MC")
    rax.grid()

    hep.cms.label("Preliminary", data=True, lumi=40, year=2017, ax=ax)
    if len(name):
        plt.savefig(f"{plotdir}{name}.pdf", bbox_inches="tight")


def rocCurve(fpr, tpr, title=None, xlim=None, ylim=None, plotdir="", name=""):
    """Plots a ROC curve"""
    if ylim is None:
        ylim = [1e-06, 0.01]
    if xlim is None:
        xlim = [0, 0.4]
    plt.figure(figsize=(12, 12))
    plt.plot(tpr, fpr)
    plt.yscale("log")
    plt.xlabel("Signal Eff.")
    plt.ylabel("BG Eff.")
    plt.title(title)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.savefig(f"{plotdir}/{name}.pdf", bbox_inches="tight")
