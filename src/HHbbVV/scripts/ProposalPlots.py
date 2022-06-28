"""
Plots for proposal.

Author(s): Raghav Kansal
"""

import pickle

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

from sample_labels import sig_key, data_key

colours = {"darkblue": "#1f78b4", "lightblue": "#a6cee3", "red": "#e31a1c", "orange": "#ff7f00"}
bg_colours = {"QCD": "lightblue", "TT": "darkblue", "ST": "orange"}
sig_colour = "red"

MAIN_DIR = "../../../"
plot_dir = f"{MAIN_DIR}/plots/ControlPlots/Jun27/"

sample_names = {"HHbbVV": r"$H\to VV\to 4q$", "QCD": "QCD", "TT": r"$t\bar{t}$"}


# load hists
with open(f"{plot_dir}/hists.pkl", "rb") as f:
    hists = pickle.load(f)


# control plots
def ratioHistPlot(
    hists: Hist,
    bg_keys: List[str],
    bg_colours: Dict[str, str] = bg_colours,
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
        label=[sample_names[key] for key in bg_keys],
        color=[colours[bg_colours[sample]] for sample in bg_keys],
    )
    hep.histplot(
        hists[sig_key, :] * sig_scale,
        ax=ax,
        histtype="step",
        label=f"{sample_names[sig_key]} $\\times$ ${formatter.format_data(float(f'{sig_scale:.3g}'))}$"
        if sig_scale != 1
        else sig_key,
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
