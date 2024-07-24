"""
Accumulating Lund plane density histograms from signal condor jobs.

Author: Raghav Kansal
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from tqdm import tqdm

from HHbbVV.hh_vars import nonres_samples, years
from HHbbVV.postprocessing import utils

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", required=True, help="path to skimmer outputs", type=str)
args = parser.parse_args()


package_path = Path(__file__).parent.parent.resolve()

for year in years:
    print(year)
    for key in tqdm(nonres_samples.values()):
        sig_lp_hist = utils.get_pickles(f"{args.data_path}/{year}/{key}/pickles", year, key)[
            "lp_hist"
        ]

        # remove negatives
        sig_lp_hist.values()[sig_lp_hist.values() < 0] = 0

        with (package_path / f"corrections/lp_ratios/signals/{year}_{key}.hist").open("wb") as f:
            pickle.dump(sig_lp_hist, f)

        break
