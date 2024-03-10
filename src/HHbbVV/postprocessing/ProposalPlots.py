"""
Plots for proposal.

Author(s): Raghav Kansal
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
from hist import Hist
from hist.intervals import ratio_uncertainty

from HHbbVV.hh_vars import data_key, sig_key

plt.rcParams.update({"font.size": 16})
plt.style.use(hep.style.CMS)
hep.style.use("CMS")
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))


colours = {"darkblue": "#1f78b4", "lightblue": "#a6cee3", "red": "#e31a1c", "orange": "#ff7f00"}
bg_colours = {"QCD": "lightblue", "TT": "darkblue", "ST": "orange"}
sig_colour = "red"

MAIN_DIR = "../../../"
plot_dir = Path(f"{MAIN_DIR}/plots/ControlPlots/Jun27/")

sample_names = {"HHbbVV": r"$H\to VV\to 4q$", "QCD": "QCD", "TT": r"$t\bar{t}$"}


# load hists
with (plot_dir / "hists.pkl").open("rb") as f:
    hists = pickle.load(f)


# control plots
def ratioHistPlot(
    hists: Hist,
    bg_keys: list[str],
    bg_colours: dict[str, str] = bg_colours,
    sig_colour: str = sig_colour,
    name: str = "",
    sig_scale: float = 1.0,
):
    """
    Makes and saves a histogram plot, with backgrounds stacked, signal separate (and optionally
    scaled) with a data/mc ratio plot below
    """

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [3, 1], "hspace": 0}, sharex=True
    )

    ax.set_ylabel("Events")
    hep.histplot(
        [hists[sample, :] for sample in bg_keys],
        ax=ax,
        histtype="fill",
        stack=True,
        label=[sample_names[key] for key in bg_keys],
        color=[colours[bg_colours[sample]] for sample in bg_keys],
    )
    hep.histplot(
        hists[sig_key, :] * sig_scale,
        ax=ax,
        histtype="step",
        label=(
            f"{sample_names[sig_key]} $\\times$ ${formatter.format_data(float(f'{sig_scale:.3g}'))}$"
            if sig_scale != 1
            else sig_key
        ),
        color=colours[sig_colour],
    )
    hep.histplot(
        hists[data_key, :], ax=ax, yerr=True, histtype="errorbar", label=data_key, color="black"
    )
    ax.legend()
    ax.set_ylim(0)
    ax.yaxis.set_major_formatter(formatter)

    bg_tot = sum([hists[sample, :] for sample in bg_keys])
    yerr = ratio_uncertainty(hists[data_key, :].values(), bg_tot.values(), "poisson")
    hep.histplot(
        hists[data_key, :] / bg_tot.values(),
        yerr=yerr,
        ax=rax,
        histtype="errorbar",
        color="black",
        capsize=4,
    )
    rax.set_ylabel("Data/MC")
    rax.grid()

    hep.cms.label("Work in Progress", data=True, lumi=40, year=2017, ax=ax, pad=0.04)
    if len(name):
        plt.savefig(name, bbox_inches="tight")


# {var: (bins, label)}
hist_vars = {
    "VVFatJetParticleNet_Th4q": ([50, 0, 1], r"Probability($H \to 4q$)"),
    "VVFatJetParticleNetHWWMD_THWW4q": ([50, 0, 1], r"Probability($H \to VV \to 4q$)"),
}

sig_scale = 11920061.740869733

for var in hist_vars:
    name = f"{plot_dir}/{var}.pdf"
    ratioHistPlot(
        hists[var],
        list(hists[var].axes[0])[1:-1],
        name=name,
        sig_scale=sig_scale,
    )


# ROCs
plot_dir = Path(f"{MAIN_DIR}/plots/TaggerAnalysis/Jun27/")

cut_labels = {
    "pt_300_1500_msoftdrop_20_320": "$p_T$: [300, 1500] GeV\n$m_{SD}$: [20, 320] GeV",
    "pt_400_600_msoftdrop_60_150": "$p_T$: [400, 600] GeV\n$m_{SD}$: [60, 150] GeV",
}

# roc vars
roc_plot_vars = {
    "th4q": {
        "title": r"$H\to 4q$ Non-MD GNN tagger",
        "score_label": "fj_PN_H4qvsQCD",
        "colour": "orange",
    },
    "thvv4q": {
        "title": r"$H\to VV\to 4q$ MD GNN tagger",
        "score_label": "score_fj_THVV4q",
        "colour": "green",
    },
}

# load rocs
with (plot_dir / "rocs.pkl").open("rb") as f:
    rocs = pickle.load(f)

xlim = [0, 0.6]
ylim = [1e-6, 1]

for cutstr in cut_labels:
    plt.figure(figsize=(12, 12))
    for t, pvars in roc_plot_vars.items():
        plt.plot(
            rocs[cutstr][t]["tpr"],
            rocs[cutstr][t]["fpr"],
            label=f"{pvars['title']} AUC: {rocs[cutstr][t]['auc']:.2f}",
            linewidth=2,
            color=pvars["colour"],
        )
        plt.vlines(
            x=rocs[cutstr][t]["tpr"][np.searchsorted(rocs[cutstr][t]["fpr"], 0.01)],
            ymin=0,
            ymax=0.01,
            colors=pvars["colour"],
            linestyles="dashed",
        )
    plt.hlines(y=0.01, xmin=0, xmax=1, colors="lightgrey", linestyles="dashed")
    plt.yscale("log")
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Efficiency")
    # plt.suptitle("HVV FatJet ROC Curves", y=0.95)
    plt.text(0.02, 0.27, cut_labels[cutstr], fontsize=20)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend()
    plt.savefig(f"{plot_dir}/roccurve_{cutstr}.pdf", bbox_inches="tight")
