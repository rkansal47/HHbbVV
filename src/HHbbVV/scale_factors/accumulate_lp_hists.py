"""
Accumulating Lund plane density histograms from signal condor jobs.

Author: Raghav Kansal
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from tqdm import tqdm

from HHbbVV import run_utils
from HHbbVV.hh_vars import nonres_samples, res_samples, years
from HHbbVV.postprocessing import utils

package_path = Path(__file__).parent.parent.resolve()


parser = argparse.ArgumentParser()
parser.add_argument("--data-path", required=True, help="path to skimmer outputs", type=str)
parser.add_argument(
    "--year",
    nargs="+",
    choices=["2016", "2016APV", "2017", "2018"],
    type=str,
    default=None,
)
run_utils.add_bool_arg(parser, "resonant", default=False, help="resonant or nonresonant signals")
run_utils.add_bool_arg(parser, "ttsfs", default=False, help="TT samples or signal")
args = parser.parse_args()

if args.resonant and args.ttsfs:
    raise ValueError("Both resonant and tt samples flags cannot be true")

if args.year is None:
    args.year = years

if args.resonant:
    samples = list(res_samples.values())
elif args.ttsfs:
    samples = [
        "TTToSemiLeptonic",
        "ST_tW_antitop_5f_NoFullyHadronicDecays",
        "ST_tW_top_5f_NoFullyHadronicDecays",
        "ST_s-channel_4f_leptonDecays",
        "ST_t-channel_antitop_4f_InclusiveDecays",
        "ST_t-channel_top_4f_InclusiveDecays",
    ]
else:
    samples = list(nonres_samples.values())

for year in args.year:
    print(year)
    for key in tqdm(samples):
        try:
            sig_lp_hist = utils.get_pickles(f"{args.data_path}/{year}/{key}/pickles", year, key)[
                "lp_hist"
            ]
        except FileNotFoundError as e:
            print(e)
            continue

        # remove negatives
        sig_lp_hist.values()[sig_lp_hist.values() < 0] = 0

        with (package_path / f"corrections/lp_ratios/signals/{year}_{key}.hist").open("wb") as f:
            pickle.dump(sig_lp_hist, f)

        # break
