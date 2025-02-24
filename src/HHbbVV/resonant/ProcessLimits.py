from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from HHbbVV.hh_vars import res_sigs
from HHbbVV.postprocessing.utils import mxmy


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


def get_lim(limits: dict, mxy: tuple):
    mx, my = mxy
    match = (limits[:, 0] == mx) * (limits[:, 1] == my)
    return match, limits[match]
