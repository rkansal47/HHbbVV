import os
import rhalphalib as rl
import numpy as np
import pickle
import logging
import sys
from collections import OrderedDict

rl.util.install_roofit_helpers()
rl.ParametericSample.PreferRooParametricHist = False
logging.basicConfig(level=logging.DEBUG)
adjust_posdef_yields = False

from utils import add_bool_arg


LUMI = {"2017": 41.48}


NUISANCE_PARAMS = {"lumi_13TeV_2017": "lnN"}


def main(args):

    nuisance_params_dict = {
        param: rl.NuisanceParameter(param, unc) for param, unc in NUISANCE_PARAMS.items()
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-hists", default="", type=str, help="input dictionary of hist.Hist templates"
    )
    parser.add_argument("--card-dir", default="cards", type=str, help="output card directory")
    parser.add_argument(
        "--nMCTF", default=0, type=int, dest="nMCTF", help="order of polynomial for TF from MC"
    )
    parser.add_argument(
        "--nDataTF",
        default=2,
        type=int,
        dest="nDataTF",
        help="order of polynomial for TF from Data",
    )
    args = parser.parse_args()

    os.system(f"mkdir -p {args.card_dir}")

    main(args)
