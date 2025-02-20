from __future__ import annotations

import argparse
import pickle
from collections import OrderedDict
from pathlib import Path

import hist
import matplotlib.pyplot as plt
import numpy as np
import plotting
import uproot
from datacardHelpers import sum_templates
from hist import Hist

from HHbbVV.hh_vars import bg_keys, data_key, years
from HHbbVV.postprocessing.postprocessing import res_shape_vars
from HHbbVV.run_utils import add_bool_arg

shapes = {
    "prefit": "Pre-Fit",
    "postfit": "Post-Fit",
}

shape_vars = res_shape_vars

selection_regions = OrderedDict(
    [
        ("fail", "Fail"),
        ("pass", "Pass"),
        ("failBlinded", "Validation Fail"),
        ("passBlinded", "Validation Pass"),
    ]
)

pass_ylims = [50, 10]
fail_ylims = [14000, 1700]
scale = 1  # optional scaling lims


def plot_fits_combined(
    hists: dict,
    bgerrs: dict,
    plot_dir: str,
    sig_key: str,
    p_bg_keys: list[str],
    preliminary: bool = True,
):
    plabel = "preliminary" if preliminary else "final"
    for shape in shapes:
        for i, shape_var in enumerate(shape_vars):
            # add "invisible" subplots between main plots to add spacing
            # https://stackoverflow.com/a/53643819/3759946
            fig, axs = plt.subplots(
                5,
                3,
                figsize=(25, 30),
                gridspec_kw=dict(
                    height_ratios=[3, 1, 0.6, 3, 1], width_ratios=[1, 0.12, 1], hspace=0, wspace=0
                ),
            )

            for ax in axs[2]:
                ax.set_visible(False)

            for ax in axs[:, 1]:
                ax.set_visible(False)

            for j, (region, region_label) in enumerate(selection_regions.items()):
                row = (j // 2) * 3
                col = (j % 2) * 2
                pass_region = region.startswith("pass")

                bgerr = np.linalg.norm(bgerrs[shape][region], axis=i)

                plot_params = {
                    "hists": hists[shape][region].project(0, i + 1),
                    "sig_keys": [sig_key],
                    "bg_keys": p_bg_keys,
                    "bg_err": bgerr,
                    "sig_scale_dict": {sig_key: 10},
                    "show": False,
                    "year": "all",
                    "ylim": pass_ylims[i] * scale if pass_region else fail_ylims[i] * scale,
                    # "name": f"{plot_dir}/{shape}_{region}_{shape_var.var}.pdf",
                    "divide_bin_width": True,
                    "axrax": (axs[row, col], axs[row + 1, col]),
                    "cmslabel": "Preliminary" if preliminary else None,
                    "cmsloc": 2,
                    "region_label": region_label,
                }

                plotting.ratioHistPlot(**plot_params)

            plt.savefig(
                f"{plot_dir}/{plabel}_combined_{shape}_{shape_var.var}.pdf", bbox_inches="tight"
            )


def plot_fits_separate(
    hists: dict,
    bgerrs: dict,
    plot_dir: str,
    sig_key: str,
    p_bg_keys: list[str],
    preliminary: bool = True,
):
    plabel = "preliminary" if preliminary else "final"
    for shape in shapes:
        for i, shape_var in enumerate(shape_vars):
            for _j, (region, region_label) in enumerate(selection_regions.items()):
                pass_region = region.startswith("pass")

                bgerr = np.linalg.norm(bgerrs[shape][region], axis=i)

                plot_params = {
                    "hists": hists[shape][region].project(0, i + 1),
                    "sig_keys": [sig_key],
                    "bg_keys": p_bg_keys,
                    "bg_err": bgerr,
                    "sig_scale_dict": {sig_key: 10},
                    "show": False,
                    "year": "all",
                    "ylim": pass_ylims[i] * scale if pass_region else fail_ylims[i] * scale,
                    "name": f"{plot_dir}/{plabel}_{shape}_{region}_{shape_var.var}.pdf",
                    "divide_bin_width": True,
                    "cmslabel": "Preliminary" if preliminary else None,
                    "cmsloc": 2,
                    "region_label": region_label,
                }

                plotting.ratioHistPlot(**plot_params)


def main(args):
    plot_dir = Path(args.plots_dir)
    for p in ["Preliminary", "Final"]:
        for c in ["Combined", "Separate"]:
            (plot_dir / p / c).mkdir(parents=True, exist_ok=True)

    cards_dir = Path(f"/uscms/home/rkansal/hhcombine/cards/{args.cards_tag}")
    file_name = "FitShapesB" if args.b_only else "FitShapesS"

    file = uproot.open(cards_dir / file_name)

    templates_dir = Path("templates/25Feb6ResBackgrounds")

    templates_dict = {}
    for year in years:
        with (templates_dir / f"{year}_templates.pkl").open("rb") as f:
            templates_dict[year] = pickle.load(f)

    pre_templates = sum_templates(templates_dict, years)

    mx, my = args.mxmy

    p_bg_keys = [k for k in bg_keys if k != "HWW"]
    sig_key = f"X[{mx}]->HY[{my}]"

    # (name in templates, name in cards)
    hist_label_map_inverse = OrderedDict(
        [
            ("QCD", "CMS_XHYbbWW_boosted_qcd_datadriven"),
            # ("Diboson", "diboson"),
            ("TT", "ttbar"),
            ("ST", "singletop"),
            ("Z+Jets", "zjets"),
            ("W+Jets", "wjets"),
            # ("X[3000]->H(bb)Y[190](VV)", "xhy_mx3000_my190"),
            (f"X[{mx}]->HY[{my}]", f"xhy_mx{mx}_my{my}"),
            (data_key, "data_obs"),
        ]
    )

    samples = p_bg_keys + [sig_key, data_key]

    hists = {}
    bgerrs = {}
    bgtots = {}

    for shape in shapes:
        hists[shape] = {
            region: Hist(
                hist.axis.StrCategory(samples, name="Sample"),
                *[shape_var.axis for shape_var in shape_vars],
                storage="double",
            )
            for region in selection_regions
        }
        bgerrs[shape] = {}
        bgtots[shape] = {}

        for region in selection_regions:
            h = hists[shape][region]
            bgerrs[shape][region] = []
            bgtots[shape][region] = []

            for i in range(len(shape_vars[1].axis)):  # mX bins
                # templates = file[shape][f"mXbin{i}{region}"]
                templates = file[f"mXbin{i}{region}_{shape}"]
                for key, file_key in hist_label_map_inverse.items():
                    if key != data_key:
                        if file_key not in templates:
                            # print(f"No {key} in mXbin{i}{region}")
                            continue

                        data_key_index = np.where(np.array(list(h.axes[0])) == key)[0][0]
                        h.view(flow=False)[data_key_index, :, i] = templates[file_key].values()

                # if key not in fit output, take from templates
                for key in p_bg_keys:
                    if key not in hist_label_map_inverse:
                        data_key_index = np.where(np.array(list(h.axes[0])) == key)[0][0]
                        h.view(flow=False)[data_key_index, :] = pre_templates[region][
                            key, ...
                        ].values()

                data_key_index = np.where(np.array(list(h.axes[0])) == data_key)[0][0]
                h.view(flow=False)[data_key_index, :, i] = np.nan_to_num(
                    templates[hist_label_map_inverse[data_key]].values()
                )

                bgerrs[shape][region].append(templates["TotalBkg"].errors())
                bgtots[shape][region].append(templates["TotalBkg"].values())

            bgerrs[shape][region] = np.array(bgerrs[shape][region])
            bgtots[shape][region] = np.array(bgtots[shape][region])
            bgerrs[shape][region] = np.minimum(bgerrs[shape][region], bgtots[shape][region])

    for preliminary, plabel in zip([True, False], ["Preliminary", "Final"]):
        plot_fits_combined(
            hists, bgerrs, args.plots_dir / plabel / "Combined", sig_key, p_bg_keys, preliminary
        )
        plot_fits_separate(
            hists, bgerrs, args.plots_dir / plabel / "Final", sig_key, p_bg_keys, preliminary
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--cards-tag", help="Cards directory", required=True, type=str)
    parser.add_argument("--plots-dir", help="plots directory", type=str, required=True)
    parser.add_argument("--mxmy", help="mX mY", type=int, required=True, nargs=2)
    add_bool_arg(parser, "b-only", "B-only fit or not", default=True)
    args = parser.parse_args()
    main(args)
