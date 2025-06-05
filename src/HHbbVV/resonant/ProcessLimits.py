from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate
from tqdm import tqdm

from HHbbVV.hh_vars import res_sigs
from HHbbVV.postprocessing import plotting
from HHbbVV.postprocessing.utils import mxmy
from HHbbVV.run_utils import add_bool_arg


def label_map(key):
    if key == "50.0":
        label = "Median expected 95% CL exclusion limits [fb]"
    elif key == "Observed":
        label = "95% CL observed exclusion limits [fb]"
    elif key == "Significance":
        label = "Local Significance"
    else:
        label = f"{key}% expected 95% CL exclusion limits [fb]"

    return label


def _parse_limits(mx, my, lines, limits, sign_lines):
    nums = 0
    for i in np.arange(len(lines) - 1, -1, -1):
        line = lines[i][:-1]
        start_str = "Observed Limit: r < "
        if line.startswith(start_str):
            nums += 1
            continue

        for key in limits:
            start_str = f"Expected {key}%: r < "
            if line.startswith(start_str):
                nums += 1
                break

        if nums == 6:
            break

    if nums != 6:
        print(f"Missing some limits for {mx}, {my}! Skipping")
        return

    nums = 0
    for i in np.arange(len(lines) - 1, -1, -1):
        line = lines[i][:-1]
        start_str = "Observed Limit: r < "
        if line.startswith(start_str):
            limits["Observed"].append([mx, my, float(line.split(start_str)[1])])
            nums += 1
            continue

        for key in limits:
            start_str = f"Expected {key}%: r < "
            if line.startswith(start_str):
                limits[key].append([mx, my, float(line.split(start_str)[1])])
                nums += 1
                break

        if nums == 6:
            break

    for i in np.arange(len(sign_lines) - 1, -1, -1):
        sign_line = sign_lines[i][:-1]
        start_str = "Significance: "
        if sign_line.startswith(start_str):
            limits["Significance"].append([mx, my, float(sign_line.split(start_str)[1])])
            break


def read_limits(cards_dir: Path, limits: dict):
    for sample in tqdm(res_sigs):
        limits_path = Path(f"{cards_dir}/{sample}/AsymptoticLimits.txt")
        sign_path = Path(f"{cards_dir}/{sample}/Significance.txt")
        mx, my = mxmy(sample)
        if limits_path.exists():
            with limits_path.open() as f:
                lines = f.readlines()
            with sign_path.open() as f:
                sign_lines = f.readlines()
        else:
            print(f"Missing {sample}")
            continue

        _parse_limits(mx, my, lines, limits, sign_lines)

    for key, val in limits.items():
        limits[key] = np.array(val)


def get_limits(cards_dir: Path, overwrite_limits: bool = False) -> np.ndarray:
    """Parses and saves limits from ``cards_dir`` if not already parsed, otherwise loads them"""

    limit_dir = cards_dir / "limits"
    limits = {
        " 2.5": [],
        "16.0": [],
        "50.0": [],
        "84.0": [],
        "97.5": [],
        "Observed": [],
        "Significance": [],
    }

    read_limits_check = False

    for key in limits:
        if not (limit_dir / f"limits_{key}.csv").exists():
            read_limits_check = True
            break

    if read_limits_check or overwrite_limits:
        read_limits(cards_dir, limits)

        limit_dir.mkdir(exist_ok=True)
        for key, limit in limits.items():
            df = pd.DataFrame(limit, columns=["MX", "MY", "Limit (fb)"])
            df.to_csv(f"{limit_dir}/limits_{key}.csv")
    else:
        for key in limits:
            limits[key] = pd.read_csv(limit_dir / f"limits_{key}.csv").to_numpy()[:, 1:]

    return limits


def get_limits_amitav():
    alimits_path = Path(
        "/uscms/home/ammitra/nobackup/2DAlphabet/fitting/CMSSW_14_1_0_pre4/src/XHYbbWW/limits/"
    )
    alimits = {
        " 2.5": [],
        "16.0": [],
        "50.0": [],
        "84.0": [],
        "97.5": [],
        "Observed": [],
        "Significance": [],
    }
    key_map = {
        # mine: amitav's
        " 2.5": "limits_Minus2",
        "16.0": "limits_Minus1",
        "50.0": "limits_Expected",
        "84.0": "limits_Plus1",
        "97.5": "limits_Plus2",
        "Observed": "limits_OBSERVED",
        "Significance": "significance",
    }

    for mkey, akey in key_map.items():
        try:
            alimits[mkey] = pd.read_csv(alimits_path / f"{akey}.csv").to_numpy()[:, 1:]
        except:
            print(f"{alimits_path}/{akey}.csv not found!")

    return alimits


def get_lim(limits: dict, mxy: tuple):
    mx, my = mxy
    match = (limits[:, 0] == mx) * (limits[:, 1] == my)
    return match, limits[match]


def boosted_plots(limits: dict, plot_dir: Path):
    (plot_dir / "boosted").mkdir(parents=True, exist_ok=True)

    mymax = 600
    mxs = np.logspace(np.log10(900), np.log10(3999), 100, base=10)
    mys = np.logspace(np.log10(60), np.log10(mymax), 100, base=10)

    xx, yy = np.meshgrid(mxs, mys)

    interpolated = {}
    grids = {}

    for key, val in limits.items():
        interpolated[key] = interpolate.LinearNDInterpolator(val[:, :2], np.log(val[:, 2]))
        grids[key] = np.exp(interpolated[key](xx, yy))

    for key, grid in grids.items():
        if key == "Significance":
            vmin, vmax, log = 0, 5, False
        else:
            vmin, vmax, log = 0.05, 1e4, True

        plotting.colormesh(
            xx,
            yy,
            grid,
            label_map(key),
            f"{plot_dir}/boosted/upper{mymax}_mesh_{key}_turbo.pdf",
            vmin=vmin,
            vmax=vmax,
            log=log,
            show=False,
        )

    for key, val in limits.items():
        if key == "Significance":
            vmin, vmax, log = 0, 5, False
        else:
            vmin, vmax, log = 0.05, 1e4, True

        plotting.XHYscatter2d(
            val, label_map(key), name=f"{plot_dir}/boosted/scatter_{key}.pdf", show=False
        )


def semimerged_boosted_plots(limits: dict, alimits: dict, plot_dir: Path):
    (plot_dir / "both").mkdir(parents=True, exist_ok=True)

    for key, val in alimits.items():
        plotting.XHYscatter2d(
            val, label_map(key), name=f"{plot_dir}/both/amitav_scatter_{key}.pdf", show=False
        )

    # check whose limits are better
    sb_better = []
    alim_med = alimits["50.0"]

    for mx, my, lim in limits["50.0"]:
        match = (alim_med[:, 0] == mx) * (alim_med[:, 1] == my)
        alim = float(alim_med[:, 2][match]) if np.any(match) else np.inf

        if alim < lim:
            pbetter = (lim - alim) / lim
            print(f"Semiboosted better for ({mx}, {my}) by {pbetter * 100:.2f}%")
            sb_better.append([mx, my, pbetter])

    sb_better = np.array(sb_better)

    plotting.scatter2d_overlay(
        limits["50.0"],
        sb_better,
        "Median expected exclusion limits (fb)",
        f"{plot_dir}/both/scatter_overlay.pdf",
        show=False,
    )


def get_combined_limits(limits: dict, alimits: dict, cards_dir: Path):
    combined_limits = {
        " 2.5": [],
        "16.0": [],
        "50.0": [],
        "84.0": [],
        "97.5": [],
        "Observed": [],
        # "Significance": [],
    }
    alim_med = np.array(alimits["50.0"])
    blim_med = np.array(limits["50.0"])

    checked_mxmy = []

    for mxyt in np.vstack((alim_med, blim_med))[:, :2]:
        mx, my = mxyt
        mxy = (int(mx), int(my))
        if mx < 900:
            continue

        if mxy in checked_mxmy:
            continue

        amatch, alim = get_lim(alim_med, mxy)
        bmatch, blim = get_lim(blim_med, mxy)

        alim = alim[0, 2] if np.any(amatch) else np.inf
        blim = blim[0, 2] if np.any(bmatch) else np.inf

        if alim < blim and (my < 200):
            # skipping samples for which 2018 PFNano failed !! :(
            print(f"Skipping {mxy} because of missing PFNano!")
            continue

        if blim < alim and (my > (134.5 + mx * 0.1285)):
            print(f"Skipping {mxy} because of missing from Amitav's limits!")
            continue

        use_lims = alimits if alim < blim else limits

        for key, lims in combined_limits.items():
            umatch, lim = get_lim(use_lims[key], mxy)
            if np.any(umatch):
                lims.append([*mxy, use_lims[key][umatch][0, 2]])
            else:
                print(f"Missing {mxy} for {key}!")

        checked_mxmy.append(mxy)

    for key, val in combined_limits.items():
        combined_limits[key] = np.array(val)

    combined_df = {}

    combined_df["MX"] = combined_limits["Observed"][:, 0]
    combined_df["MY"] = combined_limits["Observed"][:, 1]

    for key, val in combined_limits.items():
        if key != "Observed":
            combined_df[f"Expected {key}"] = val[:, 2]
        else:
            combined_df[key] = val[:, 2]

    pd.DataFrame(combined_df).to_csv(cards_dir / "combined_limits.csv")

    return combined_limits


def combined_plots(combined_limits: dict, plot_dir: Path):
    (plot_dir / "combined").mkdir(parents=True, exist_ok=True)

    mxs = np.logspace(np.log10(900), np.log10(3999), 300, base=10)
    mys = np.logspace(np.log10(60), np.log10(2800), 300, base=10)
    cxx, cyy = np.meshgrid(mxs, mys)

    for key, val in combined_limits.items():
        if key == "Significance":
            continue

        if key == "Significance":
            vmin, vmax, log = 0, 5, False
        else:
            vmin, vmax, log = 0.05, 1e4, True

        interpolated = interpolate.LinearNDInterpolator(val[:, :2], np.log(val[:, 2]))
        grid = np.exp(interpolated(cxx, cyy))

        for prelim, plabel in zip([True, False], ["prelim_", ""]):
            plotting.colormesh(
                cxx,
                cyy,
                grid,
                label_map(key),
                f"{plot_dir}/combined/{plabel}combined_mesh_{key}.pdf",
                vmin=vmin,
                vmax=vmax,
                log=log,
                region_labels=True,
                figsize=(14, 10),
                preliminary=prelim,
                show=False,
            )

    for key, val in combined_limits.items():
        if key != "Significance":
            continue

        plotting.XHYscatter2d(
            val, label_map(key), name=f"{plot_dir}/combined/combined_scatter_{key}.pdf", show=False
        )


def main(args):
    limits = get_limits(args.cards_dir)
    alimits = get_limits_amitav()
    combined_limits = get_combined_limits(limits, alimits, args.cards_dir)

    if args.boosted:
        boosted_plots(limits, args.plot_dir)

    if args.semimerged_boosted:
        semimerged_boosted_plots(limits, alimits, args.plot_dir)

    if args.combined:
        combined_plots(combined_limits, args.plot_dir)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cards-dir",
        type=str,
        default="/eos/uscms/store/user/rkansal/bbVV/cards/25Mar29QCDTF11nTF21",
        help="Directory containing the limit cards",
    )
    parser.add_argument("--tag", type=str, required=True, help="tag for the plots")

    add_bool_arg(parser, "boosted", "Generate boosted plots")
    add_bool_arg(parser, "semimerged-boosted", "Generate semimerged boosted plots")
    add_bool_arg(parser, "combined", "Generate combined plots")

    args = parser.parse_args()
    MAIN_DIR = "../../../"
    args.plot_dir = Path(f"{MAIN_DIR}/plots/XHY/Limits/{args.tag}")

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
