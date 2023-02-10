"""
Creates datacards for Higgs Combine using hist.Hist templates output from PostProcess.py
(1) Adds systematics for all samples,
(2) Sets up data-driven QCD background estimate ('rhalphabet' method)

Based on https://github.com/LPC-HH/combine-hh/blob/master/create_datacard.py

Author: Raghav Kansal
"""


import os
import numpy as np
import pickle
import logging
from collections import OrderedDict
import utils

# import hist
# from hist import Hist
import rhalphalib as rl

rl.util.install_roofit_helpers()
rl.ParametericSample.PreferRooParametricHist = False
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
adjust_posdef_yields = False

# from utils import add_bool_arg
from sample_labels import qcd_key, data_key

LUMI = {"2017": 41.48}
LP_SF = 1.3

mc_samples = OrderedDict(
    [
        ("ggHH_kl_1_kt_1_hbbhww4q", "HHbbVV"),
        ("ttbar", "TT"),
    ]
)


# dictionary of nuisance params -> modifier
nuisance_params = {"lumi_13TeV_2017": "lnN", "lp_sf": "lnN"}
nuisance_params_dict = {
    param: rl.NuisanceParameter(param, unc) for param, unc in nuisance_params.items()
}

# dictionary of shape systematics -> name in cards
systs = OrderedDict([])

CMS_PARAMS_LABEL = "CMS_bbWW_boosted_ggf"
PARAMS_LABEL = "bbWW_boosted_ggf"


# for local interactive testing
args = type("test", (object,), {})()
args.data_dir = "../../../../data/skimmer/Apr28/"
args.plot_dir = "../../../plots/05_26_testing"
args.year = "2017"
args.bdt_preds = f"{args.data_dir}/absolute_weights_preds.npy"
args.templates_file = "templates/test2.pkl"
args.cat = "1"
args.nDataTF = 2


# import importlib
# importlib.reload(utils)


def main(args):
    # pass, fail x unblinded, blinded
    regions = [
        f"{pf}{blind_str}"
        for pf in [f"passCat{args.cat}", "fail"]
        for blind_str in ["", "Blinded"]
        # f"{pf}{blind_str}"
        # for pf in [f"passCat{args.cat}", "fail"]
        # for blind_str in ["Blinded"]
    ]

    with open(args.templates_file, "rb") as f:
        templates = pickle.load(f)

    # random template from which to extract common data
    sample_templates = templates[regions[0]]
    shape_var = sample_templates.axes[1].name

    # get bins, centers, and scale centers for polynomial evaluation
    msd_bins = sample_templates.axes[1].edges
    msd_pts = msd_bins[:-1] + 0.5 * np.diff(msd_bins)
    msd_scaled = (msd_pts - msd_bins[0]) / (msd_bins[-1] - msd_bins[0])

    msd_obs = rl.Observable(shape_var, msd_bins)

    # QCD overall pass / fail efficiency
    qcd_eff = (
        templates[f"passCat{args.cat}"][qcd_key, :].sum().value
        / templates[f"fail"][qcd_key, :].sum().value
    )

    # transfer factor
    tf_dataResidual = rl.BasisPoly(
        f"{CMS_PARAMS_LABEL}_tf_dataResidual_cat{args.cat}",
        (args.nDataTF,),
        [shape_var],
        basis="Bernstein",
        limits=(-20, 20),
    )
    tf_dataResidual_params = tf_dataResidual(msd_scaled)
    tf_params_pass = qcd_eff * tf_dataResidual_params  # scale params initially by qcd eff

    # qcd params
    qcd_params = np.array(
        [
            rl.IndependentParameter(f"{CMS_PARAMS_LABEL}_qcdparam_msdbin{i}", 0)
            for i in range(msd_obs.nbins)
        ]
    )

    # build actual fit model now
    model = rl.Model("HHModel")

    for region in regions:
        region_templates = templates[region]

        logging.info("starting region: %s" % region)
        ch = rl.Channel(region.replace("_", ""))  # can't have '_'s in name
        model.addChannel(ch)

        # TODO: shape systs

        for card_name, sample_name in mc_samples.items():
            logging.info("get templates for: %s" % sample_name)

            sample_template = region_templates[sample_name, :]

            stype = rl.Sample.SIGNAL if "HH" in card_name else rl.Sample.BACKGROUND
            sample = rl.TemplateSample(ch.name + "_" + card_name, stype, sample_template)

            # systematics
            sample.setParamEffect(nuisance_params_dict["lumi_13TeV_2017"], 1.02)

            if sample_name == "HHbbVV":
                sample.setParamEffect(nuisance_params_dict["lp_sf"], LP_SF)

            # nominal values, errors
            values_nominal = np.maximum(sample_template.values(), 0.0)

            mask = values_nominal > 0
            errors_nominal = np.ones_like(values_nominal)
            errors_nominal[mask] = (
                1.0 + np.sqrt(sample_template.variances()[mask]) / values_nominal[mask]
            )

            logging.debug("nominal   : {nominal}".format(nominal=values_nominal))
            logging.debug("error     : {errors}".format(errors=errors_nominal))

            # set mc stat uncs
            logging.info("setting autoMCStats for %s in %s" % (sample_name, region))

            sample_name = region.split("Blinded")[0] + f"_{sample_name}"
            if not args.bblite:
                sample.autoMCStats(sample_name=sample_name)

            # TODO: shape systematics
            ch.addSample(sample)

        if args.bblite:
            channel_name = region.split("Blinded")[0]
            ch.autoMCStats(channel_name=channel_name)

        # data observed
        ch.setObservation(region_templates[data_key, :])

    for blind_str in ["", "Blinded"]:
        # for blind_str in ["Blinded"]:
        passChName = f"passCat{args.cat}{blind_str}".replace("_", "")
        failChName = f"fail{blind_str}".replace("_", "")
        logging.info(
            "setting transfer factor for pass region %s, fail region %s" % (passChName, failChName)
        )
        failCh = model[failChName]
        passCh = model[passChName]

        # sideband fail
        # was integer, and numpy complained about subtracting float from it
        initial_qcd = failCh.getObservation().astype(float)
        for sample in failCh:
            logging.debug("subtracting %s from qcd" % sample._name)
            initial_qcd -= sample.getExpectation(nominal=True)

        if np.any(initial_qcd < 0.0):
            raise ValueError("initial_qcd negative for some bins..", initial_qcd)

        sigmascale = 10  # to scale the deviation from initial
        scaled_params = (
            initial_qcd * (1 + sigmascale / np.maximum(1.0, np.sqrt(initial_qcd))) ** qcd_params
        )

        # add samples
        fail_qcd = rl.ParametericSample(
            f"{failChName}_{PARAMS_LABEL}_qcd_datadriven",
            rl.Sample.BACKGROUND,
            msd_obs,
            scaled_params,
        )
        failCh.addSample(fail_qcd)

        pass_qcd = rl.TransferFactorSample(
            f"{passChName}_{PARAMS_LABEL}_qcd_datadriven",
            rl.Sample.BACKGROUND,
            tf_params_pass,
            fail_qcd,
        )
        passCh.addSample(pass_qcd)

    os.system(f"mkdir -p {args.cards_dir}")

    logging.info("rendering combine model")

    if args.model_name is not None:
        out_dir = os.path.join(str(args.cards_dir), args.model_name)
    else:
        out_dir = args.cards_dir

    model.renderCombine(out_dir)

    with open(f"{out_dir}/model.pkl", "wb") as fout:
        pickle.dump(model, fout, 2)  # use python 2 compatible protocol


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--templates-file",
        default="",
        type=str,
        help="input pickle file of dict of hist.Hist templates",
    )
    parser.add_argument("--cards-dir", default="cards", type=str, help="output card directory")
    parser.add_argument("--cat", default="1", type=str, choices=["1"], help="category")
    parser.add_argument(
        "--nDataTF",
        default=2,
        type=int,
        dest="nDataTF",
        help="order of polynomial for TF from Data",
    )
    parser.add_argument("--model-name", default=None, type=str, help="output model name")
    utils.add_bool_arg(parser, "bblite", "use barlow-beeston-lite method")
    args = parser.parse_args()

    main(args)
