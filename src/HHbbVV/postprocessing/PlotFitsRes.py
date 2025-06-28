from __future__ import annotations

import argparse
import json
import pickle
from collections import OrderedDict
from pathlib import Path

import hist
import matplotlib as mpl
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
    "postfitb": "Post-Fit B-only",
    "postfits": "Post-Fit S+B",
}

shape_vars = res_shape_vars

selection_regions = OrderedDict(
    [
        ("fail", "FM SF"),
        ("pass", "FM SP"),
        ("failBlinded", "FM VF"),
        ("passBlinded", "FM VP"),
    ]
)

pass_ylims = [55, 9]
fail_ylims = [14000, 1700]
scale = 1  # optional scaling lims


def get_1d_plot_params(
    i, region, bgerrs, shape, hists, sig_key, p_bg_keys, sig_scale, region_label, preliminary
):
    pass_region = region.startswith("pass")
    vregion = "Blinded" in region

    bgerr = np.linalg.norm(bgerrs[shape][region], axis=i)

    plot_params = {
        "hists": hists[shape][region].project(0, i + 1),
        "sig_keys": [sig_key],
        "bg_keys": p_bg_keys,
        "bg_err": bgerr,
        "sig_scale_dict": {sig_key: sig_scale},
        "year": "all",
        "ylim": pass_ylims[i] * scale if pass_region else fail_ylims[i] * scale,
        # "name": f"{plot_dir}/{shape}_{region}_{shape_var.var}.pdf",
        "region_label": region_label,
        "combine_other_bgs": True,
        "plot_pulls": True,
        "divide_bin_width": True,
        "cmslabel": "Preliminary" if preliminary else None,
        "cmsloc": 0,
        "resonant": True,
        "plot_signal": pass_region and not vregion,
    }

    # pprint(plot_params)

    return plot_params


def plot_fits_combined(
    hists: dict,
    bgerrs: dict,
    plot_dir: str,
    sig_key: str,
    p_bg_keys: list[str],
    preliminary: bool = True,
    sig_scale: float = 10,
):
    plabel = "preliminary" if preliminary else "final"
    for shape in shapes:
        print("\t\t", shape)
        # if shape == "prefit":
        #     continue
        for i, shape_var in enumerate(shape_vars):
            print("\t\t\t", shape_var.var)
            # add "invisible" subplots between main plots to add spacing
            # https://stackoverflow.com/a/53643819/3759946
            fig, axs = plt.subplots(
                5,
                3,
                figsize=(25, 30),
                gridspec_kw=dict(
                    height_ratios=[3, 1, 0.3, 3, 1], width_ratios=[1, 0.12, 1], hspace=0.1, wspace=0
                ),
            )

            for ax in axs[2]:
                ax.set_visible(False)

            for ax in axs[:, 1]:
                ax.set_visible(False)

            for j, (region, region_label) in enumerate(selection_regions.items()):
                print("\t\t\t\t", region_label)
                # print(region)
                row = (j // 2) * 3
                col = (j % 2) * 2

                plot_params = get_1d_plot_params(
                    i,
                    region,
                    bgerrs,
                    shape,
                    hists,
                    sig_key,
                    p_bg_keys,
                    sig_scale,
                    region_label,
                    preliminary,
                )
                plot_params["axrax"] = (axs[row, col], axs[row + 1, col])

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
    sig_scale: float = 10,
):
    plabel = "preliminary" if preliminary else "final"
    for shape in shapes:
        print("\t\t", shape)
        for i, shape_var in enumerate(shape_vars):
            print("\t\t\t", shape_var.var)
            for _j, (region, region_label) in enumerate(selection_regions.items()):

                plot_params = get_1d_plot_params(
                    i,
                    region,
                    bgerrs,
                    shape,
                    hists,
                    sig_key,
                    p_bg_keys,
                    sig_scale,
                    region_label,
                    preliminary,
                )
                plot_params["name"] = f"{plot_dir}/{plabel}_{shape}_{region}_{shape_var.var}.pdf"

                plotting.ratioHistPlot(**plot_params)


def plot_fits_slices(
    hists: dict,
    bgerrs: dict,
    plot_dir: str,
    sig_key: str,
    p_bg_keys: list[str],
    preliminary: bool = True,
    sig_scale: float = 10,
):
    plabel = "preliminary" if preliminary else "final"
    for shape in shapes:
        print("\t\t", shape)
        for _j, (region, region_label) in enumerate(selection_regions.items()):
            print("\t\t\t", region_label)
            pdir = plot_dir / shape / region
            pdir.mkdir(parents=True, exist_ok=True)
            for i in range(10):
                mxbin = hists[shape][region].axes[2][i]
                plot_params = get_1d_plot_params(
                    0,
                    region,
                    bgerrs,
                    shape,
                    hists,
                    sig_key,
                    p_bg_keys,
                    sig_scale,
                    region_label,
                    preliminary,
                )
                plot_params["hists"] = hists[shape][region][:, :, i]
                plot_params["bg_err"] = bgerrs[shape][region][i]
                plot_params["name"] = f"{pdir}/mXbin{i}_{plabel}.pdf"
                plot_params["ylim"] = plot_params["ylim"] / 4.5
                plot_params["region_label"] = (
                    "\n"
                    + region_label
                    + "\n"
                    + rf"$M^{{rec}}_X \in [{mxbin[0]:.0f}, {mxbin[1]:.0f}]$"
                )
                plotting.ratioHistPlot(**plot_params)

                # plot_params = {
                #     "hists": hists[shape][region][:, :, i],
                #     "sig_keys": [sig_key],
                #     "bg_keys": p_bg_keys,
                #     "bg_err": bgerrs[shape][region][i],
                #     "sig_scale_dict": {sig_key: sig_scale},
                #     "show": False,
                #     "year": "all",
                #     "ylim": (
                #         (pass_ylims[0] * scale / 5.0)
                #         if pass_region
                #         else (fail_ylims[0] * scale / 5.0)
                #     ),
                #     "name": f"{pdir}/mXbin{i}_{plabel}.pdf",
                #     "divide_bin_width": True,
                #     "cmslabel": "Preliminary" if preliminary else None,
                #     "cmsloc": 2,
                #     "region_label": "\n"
                #     + region_label
                #     + "\n"
                #     + rf"$M^{{rec}}_X \in [{mxbin[0]:.0f}, {mxbin[1]:.0f}]$",
                # }


def main(args):
    cards_dir = Path("/uscms/home/rkansal/hhcombine/cards")
    plot_dir = Path("/uscms/home/rkansal/nobackup/HHbbVV/plots/PostFit")
    if not cards_dir.exists():
        print(f"{cards_dir} not found. Trying UCSD.")
        cards_dir = Path("/home/users/rkansal/combineenv/CMSSW_11_3_4/src/cards")
        plot_dir = Path("/home/users/rkansal/HHbbVV/plots/PostFit")
        if not cards_dir.exists():
            print(f"{cards_dir} also not found. Exiting!")
            return

    cards_dir = cards_dir / args.cards_tag
    plot_dir = plot_dir / args.plots_tag

    # TODO: add option to do both
    if args.b_only:
        del shapes["postfits"]
        file_name = "FitShapesB.root"
    else:
        del shapes["postfitb"]
        file_name = "FitShapesS.root"

    mx, my = args.mxmy

    if not (cards_dir / file_name).exists():
        cards_dir = cards_dir / f"NMSSM_XToYHTo2W2BTo4Q2B_MX-{mx}_MY-{my}"
        plot_dir = plot_dir / f"NMSSM_XToYHTo2W2BTo4Q2B_MX-{mx}_MY-{my}"
        if not (cards_dir / file_name).exists():
            print(f"{cards_dir / file_name} does not exist! Exiting.")
            return

    for p in ["Preliminary", "Final"]:
        if args.hists1d:
            for c in ["Combined", "Separate", "Slices"]:
                (plot_dir / p / c).mkdir(parents=True, exist_ok=True)

        if args.hists2d:
            for shape in shapes:
                (plot_dir / p / shape).mkdir(parents=True, exist_ok=True)

    # save args as json in plot_dir
    with (plot_dir / "args.json").open("w") as f:
        json.dump(args.__dict__, f, indent=4)

    file = uproot.open(cards_dir / file_name)

    templates_dir = Path("templates/25Feb6ResBackgrounds")

    templates_dict = {}
    for year in years:
        with (templates_dir / f"{year}_templates.pkl").open("rb") as f:
            templates_dict[year] = pickle.load(f)

    pre_templates = sum_templates(templates_dict, years)

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
            (sig_key, f"xhy_mx{mx}_my{my}"),
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
                skey = "prefit" if shape == "prefit" else "postfit"
                templates = file[f"mXbin{i}{region}_{skey}"]
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
        print(plabel)
        if args.hists1d:
            print("\t", "Combined")
            plot_fits_combined(
                hists,
                bgerrs,
                plot_dir / plabel / "Combined",
                sig_key,
                p_bg_keys,
                preliminary,
                sig_scale=args.sig_scale,
            )
            print("\t", "Separate")
            plot_fits_separate(
                hists,
                bgerrs,
                plot_dir / plabel / "Separate",
                sig_key,
                p_bg_keys,
                preliminary,
                sig_scale=args.sig_scale,
            )

        if preliminary and args.slices:
            print("\t", "Slices")
            plot_fits_slices(
                hists,
                bgerrs,
                plot_dir / plabel / "Slices",
                sig_key,
                p_bg_keys,
                preliminary,
                sig_scale=args.sig_scale,
            )

        if args.hists2d:
            print("\t 2d")
            pplabel = "preliminary_" if preliminary else ""
            plotting.hist2dPullPlot(
                hists["postfits"]["pass"],
                bgerrs["postfits"]["pass"],
                sig_key,
                p_bg_keys,
                "FM SP",
                preliminary=preliminary,
                name=f"{plot_dir}/{plabel}/{pplabel}pull2d.pdf",
            )

            # for shape in shapes:
            #     samples = ["Data", "TT", "Z+Jets", "W+Jets", "QCD", "Hbb", "Diboson", sig_key]
            #     if shape == "shapes_prefit":
            #         samples = samples[1:]  # no need to plot data again in post-fit

            #     plotting.hist2ds(
            #         hists[shape],
            #         plot_dir / plabel / shape,
            #         regions=["pass", "fail"],
            #         region_labels=selection_regions,
            #         samples=samples,
            #         fail_zlim=[1, 1e5],
            #         pass_zlim=[1e-4, 100],
            #     )


if __name__ == "__main__":
    mpl.use("Agg")

    parser = argparse.ArgumentParser()

    parser.add_argument("--cards-tag", help="Cards directory", required=True, type=str)
    parser.add_argument("--plots-tag", help="plots directory", type=str, required=True)
    parser.add_argument("--mxmy", help="mX mY", type=int, required=True, nargs=2)
    parser.add_argument("--sig-scale", help="optional signal scaling", default=2, type=float)
    add_bool_arg(parser, "b-only", "B-only fit or not", default=False)
    add_bool_arg(parser, "hists1d", "make 1D hists", default=True)
    add_bool_arg(parser, "slices", "1d slices", default=True)
    add_bool_arg(parser, "hists2d", "make 2D hists", default=True)
    args = parser.parse_args()
    main(args)
