from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import plotting
import postprocessing
import scipy.integrate as integrate
import utils
from sklearn.metrics import roc_curve
from tqdm import tqdm

from HHbbVV.hh_vars import (
    nonres_samples,
    res_samples,
    samples,
    years,
)
from HHbbVV.run_utils import add_bool_arg

plt.style.use(hep.style.CMS)
hep.style.use("CMS")
plt.rcParams.update({"font.size": 24})


MAIN_DIR = Path("../../../")

samples = samples | nonres_samples


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def make_rocs(events_dict, bb_masks, sig_keys, cut_labels, cuts_dict, plot_vars):
    rocs = {}
    tot_bg_keys = ["TT", "QCD"]
    bg_skip = 1
    weight_key = "finalWeight"

    for cutstr in cut_labels:
        print(cutstr)
        rocs[cutstr] = {}
        # for sig_key in tqdm(nonres_sig_keys + res_sig_keys):
        for sig_key in tqdm(sig_keys):
            gensel = np.all(
                events_dict[sig_key]["ak8FatJetHVV"].to_numpy().astype(bool)
                == ~bb_masks[sig_key].to_numpy(),
                axis=1,
            )
            rocs[cutstr][sig_key] = {}
            sig_cut = cuts_dict[sig_key][cutstr][~bb_masks[sig_key]] * gensel
            for bg_label, bg_keys in (
                {"Combined": tot_bg_keys} | {bg_key: [bg_key] for bg_key in tot_bg_keys}
            ).items():
                rocs[cutstr][sig_key][bg_label] = {}
                bg_cuts = {
                    bg_key: cuts_dict[bg_key][cutstr][~bb_masks[bg_key]] for bg_key in bg_keys
                }

                y_true = np.concatenate(
                    [
                        np.ones(len(events_dict[sig_key][sig_cut])),
                        np.zeros(
                            int(
                                np.ceil(
                                    np.sum(
                                        [
                                            len(events_dict[bg_key][bg_cuts[bg_key]])
                                            for bg_key in bg_keys
                                        ]
                                    )
                                    / bg_skip
                                )
                            )
                        ),
                    ]
                )
                # print(y_true[np.sum(sig_cut):])

                weights = np.concatenate(
                    [events_dict[sig_key][weight_key][sig_cut]]
                    + [
                        events_dict[bg_key][weight_key][bg_cuts[bg_key]][::bg_skip]
                        for bg_key in bg_keys
                    ],
                )

                for t, pvars in plot_vars.items():
                    score_label = pvars["score_label"]
                    scores = np.concatenate(
                        [events_dict[sig_key][score_label][sig_cut]]
                        + [
                            events_dict[bg_key][score_label][bg_cuts[bg_key]][::bg_skip]
                            for bg_key in bg_keys
                        ],
                    )
                    # print(scores[np.sum(sig_cut):])
                    fpr, tpr, thresholds = roc_curve(y_true, scores, sample_weight=weights)
                    rocs[cutstr][sig_key][bg_label][t] = {
                        "fpr": fpr,
                        "tpr": tpr,
                        "thresholds": thresholds,
                        "auc": integrate.trapz(tpr, fpr),
                        "label": bg_label,
                    }

    return rocs


def main(args):
    plot_dir = MAIN_DIR / "plots/TaggerAnalysis" / args.plots_tag / args.year
    plot_dir.mkdir(parents=True, exist_ok=True)

    DATA_DIR = Path("/ceph/cms/store/user/rkansal/bbVV/skimmer/")

    sig_samples_dir = (DATA_DIR / "25Jan9UpdateLPFix") if not args.res else (DATA_DIR / "25Feb6XHY")
    bg_samples_dir = DATA_DIR / "24Mar6AllYearsBDTVars"

    nonres_sig_keys = ["HHbbVV"]
    nonres_sig_samples = {key: samples[key] for key in nonres_sig_keys}

    res_mass_points = [
        (4000, 80),
        (4000, 125),
        # (4000, 150),
        (4000, 190),
        (4000, 250),
        # (4000, 300),
        (4000, 400),
        (4000, 500),
        (4000, 600),
    ]

    res_sig_keys = [f"X[{mX}]->H(bb)Y[{mY}](VV)" for (mX, mY) in res_mass_points]
    res_sig_samples = {key: res_samples[key] for key in res_sig_keys}

    sig_keys = nonres_sig_keys if not args.res else res_sig_keys
    sig_samples = nonres_sig_samples if not args.res else res_sig_samples

    # (column name, number of subcolumns)
    load_columns = [
        ("weight", 1),
        ("weight_noTrigEffs", 1),
        ("ak8FatJetPt", 2),
        ("ak8FatJetMsd", 2),
        ("ak8FatJetHVV", 2),
        ("ak8FatJetParticleNetMD_Txbb", 2),
        ("VVFatJetParTMD_THWWvsT", 1),
    ]

    # # Both Jet's Regressed Mass above 50
    events_dict = postprocessing.load_samples(
        sig_samples_dir,
        sig_samples,
        args.year,
        # filters=postprocessing.load_filters,
        columns=utils.format_columns(load_columns),
        variations=False,
    )

    # (column name, number of subcolumns)
    load_columns = [
        ("weight", 1),
        ("weight_noTrigEffs", 1),
        ("ak8FatJetPt", 2),
        ("ak8FatJetMsd", 2),
        ("ak8FatJetParticleNetMD_Txbb", 2),
        ("VVFatJetParTMD_THWWvsT", 1),
    ]

    events_dict = {
        **events_dict,
        **postprocessing.load_samples(
            bg_samples_dir,
            {key: samples[key] for key in ["QCD", "TT"]},
            args.year,
            # filters=postprocessing.load_filters,
            columns=utils.format_columns(load_columns),
            variations=False,
        ),
    }

    cutflow = pd.DataFrame(index=list(events_dict.keys()))
    utils.add_to_cutflow(events_dict, "Preselection", "finalWeight", cutflow)
    cutflow.to_csv(plot_dir / "cutflow.csv")
    print(cutflow)

    bb_masks = postprocessing.bb_VV_assignment(events_dict)

    """
    ``cuts_dict`` will be of format:
    {
        sample1: {
            "cut1var1_min_max_cut1var2...": cut1,
            "cut2var2...": cut2,
            ...
        },
        sample2...
    }
    """

    pt_key = "Pt"
    msd_key = "Msd"
    var_prefix = "ak8FatJet"

    cutvars_dict = {"Pt": "pt", "Msd": "msoftdrop"}

    all_cuts = [
        {pt_key: [300, 3000]},
        # {pt_key: [400, 600], msd_key: [60, 150]},
        # {pt_key: [600, 1000], msd_key: [30, 250]},
        # {pt_key: [300, 1500], msd_key: [110, 140]},
    ]

    var_labels = {pt_key: r"$p_T$", msd_key: r"$m_{SD}$"}

    cuts_dict = {}
    cut_labels = {}  # labels for plot titles, formatted as "var1label: [min, max] var2label..."

    for sample, events in events_dict.items():
        # print(sample)
        cuts_dict[sample] = {}
        for cutvars in all_cuts:
            cutstrs = []
            cutlabel = []
            cuts = []
            for cutvar, (cutmin, cutmax) in cutvars.items():
                cutstrs.append(f"{cutvars_dict[cutvar]}_{cutmin}_{cutmax}")
                cutlabel.append(f"{var_labels[cutvar]}: [{cutmin}, {cutmax}]")
                cuts.append(events[f"{var_prefix}{cutvar}"] >= cutmin)
                cuts.append(events[f"{var_prefix}{cutvar}"] < cutmax)

            cutstr = "_".join(cutstrs)
            cut = np.prod(cuts, axis=0)
            cuts_dict[sample][cutstr] = cut.astype(bool)

            if cutstr not in cut_labels:
                cut_labels[cutstr] = " ".join(cutlabel)

    plot_vars = {
        "thvv4qt": {
            "title": r"ParT $T_{HWW}$",
            "score_label": "VVFatJetParTMD_THWWvsT",
            "colour": "green",
        },
    }

    print("\nMaking ROCs")
    rocs = make_rocs(events_dict, bb_masks, sig_keys, cut_labels, cuts_dict, plot_vars)

    print("\nPlotting ROCs")
    cutstr = "pt_300_3000"
    t = "thvv4qt"

    for plabel, prelim in zip(["prelim_", ""], [True, False]):
        for bkey, _blabel in zip(["QCD", "TT", "Combined"], ["QCD", r"t$\rightarrow$bW", r"QCD+t"]):
            procs = {"all": {}}
            for skey in sig_keys:
                roc = rocs[cutstr][skey][bkey][t].copy()
                # roc["fpr"] = roc["fpr"]  # mass reweighting
                roc["label"] = "SM HWW"
                procs["all"][skey] = roc

            plotting.multiROCCurve(
                procs,
                [],
                # title=rf"Y$\rightarrow$WW 4q vs {blabel}",
                xlim=[0, 1],
                ylim=[1e-4, 1],
                year="all",  # this is just to not plot any year at all
                # kin_label=r"600 < $p_T$ < 1000 GeV, |$\eta$| < 2.4" "\n" r"$m_{SD}>30$ GeV",
                plot_dir=plot_dir,
                name=f"{args.year}_{plabel}_ROC_{bkey}",
                prelim=prelim,
                show=False,
            )

        #     break
        # break


if __name__ == "__main__":
    mpl.use("Agg")

    parser = argparse.ArgumentParser()

    parser.add_argument("--plots-tag", help="plots directory", type=str, required=True)
    parser.add_argument("--year", help="year", type=str, required=True, choices=years)
    add_bool_arg(parser, "res", "Resonant or not", default=False)
    args = parser.parse_args()
    main(args)
