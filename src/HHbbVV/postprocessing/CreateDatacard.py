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

# import hist
# from hist import Hist
import rhalphalib as rl

rl.util.install_roofit_helpers()
rl.ParametericSample.PreferRooParametricHist = False
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
adjust_posdef_yields = False

# from utils import add_bool_arg
from hh_vars import sig_key, qcd_key, data_key, years

years = "2017"

mc_samples = OrderedDict(
    [
        ("ggHH_kl_1_kt_1_hbbhww4q", "HHbbVV"),
        ("ttbar", "TT"),
        ("singletop", "ST"),
        ("vjets", "V+Jets"),
    ]
)

all_mc = list(mc_samples.keys())

# dictionary of nuisance params -> modifier
nuisance_params = {"lumi_13TeV_2017": "lnN", "lp_sf": "lnN", "trigger": "lnN"}
nuisance_params_dict = {
    param: rl.NuisanceParameter(param, unc) for param, unc in nuisance_params.items()
}

# https://gitlab.cern.ch/hh/naming-conventions#experimental-uncertainties
syst_vals = {
    "lumi_13TeV_2016": 1.01,
    "lumi_13TeV_2017": 1.02,
    "lumi_13TeV_2018": 1.015,
}


# dictionary of shape systematics -> (name in cards, samples affected by it)
shape_systs = {
    "FSRPartonShower": ("ps_fsr", [sig_key, "V+Jets"]),
    "ISRPartonShower": ("ps_isr", [sig_key, "V+Jets"]),
    "PDFalphaS": ("CMS_bbbb_boosted_ggf_ggHHPDFacc", [sig_key]),
    "JES": ("CMS_scale_j_2017", all_mc),  # TODO: separate into individual
    "txbb": ("CMS_bbbb_boosted_ggf_PNetHbbScaleFactors_correlated", [sig_key]),
    # 'triggerEffSF': 'CMS_bbbb_boosted_ggf_triggerEffSF_uncorrelated'  # TODO: update once trigger uncs. are sorted
}

uncorr_year_shape_systs = {
    "pileup": ("CMS_pileup", all_mc),
    "JER": ("CMS_res_j", all_mc),
    "JMS": ("CMS_bbbb_boosted_ggf_jms", all_mc),
    "JMR": ("CMS_bbbb_boosted_ggf_jmr", all_mc),
}

for skey, (sname, ssamples) in uncorr_year_shape_systs.items():
    for year in ["2017"]:
        shape_systs[skey] = (f"{sname}_{year}", ssamples)
        # systs[skey + year] = (f"{sname}_{year}", ssamples)

shape_systs_dict = {
    skey: rl.NuisanceParameter(sname, "shape") for skey, (sname,) in shape_systs.items()
}


CMS_PARAMS_LABEL = "CMS_bbWW_boosted_ggf"
PARAMS_LABEL = "bbWW_boosted_ggf"


def main(args):
    # (pass, fail) x (unblinded, blinded)
    regions = [f"{pf}{blind_str}" for pf in [f"pass", "fail"] for blind_str in ["", "Blinded"]]

    with open(args.templates_file, "rb") as f:
        templates = pickle.load(f)

    syst_vals = {**syst_vals, **templates["systematics"]}  # LP SF and trig effs.

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
        templates[f"pass"][qcd_key, :].sum().value / templates[f"fail"][qcd_key, :].sum().value
    )

    # transfer factor
    tf_dataResidual = rl.BasisPoly(
        f"{CMS_PARAMS_LABEL}_tf_dataResidual",
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

            # TODO: might need to change if HH4b is a background
            stype = rl.Sample.SIGNAL if "HH" in card_name else rl.Sample.BACKGROUND
            sample = rl.TemplateSample(ch.name + "_" + card_name, stype, sample_template)

            # systematics
            sample.setParamEffect(
                nuisance_params_dict["lumi_13TeV_2017"], syst_vals["lumi_13TeV_2017"]
            )

            if sample_name == "HHbbVV":
                sample.setParamEffect(nuisance_params_dict["lp_sf"], 1 + syst_vals["lp_sf_unc"])

            # nominal values, errors
            values_nominal = np.maximum(sample_template.values(), 0.0)

            mask = values_nominal > 0
            errors_nominal = np.ones_like(values_nominal)
            errors_nominal[mask] = (
                1.0 + np.sqrt(sample_template.variances()[mask]) / values_nominal[mask]
            )

            logging.debug("nominal   : {nominal}".format(nominal=values_nominal))
            logging.debug("error     : {errors}".format(errors=errors_nominal))

            if not args.bblite:
                # set mc stat uncs
                logging.info("setting autoMCStats for %s in %s" % (sample_name, region))

                # same stats unc. for blinded and unblinded cards
                stats_sample_name = region.split("Blinded")[0] + f"_{sample_name}"
                sample.autoMCStats(sample_name=stats_sample_name)

            # TODO: shape systematics

            for skey, (sname, ssamples) in shape_systs.items():
                if sample_name not in ssamples:
                    continue

                print(f"{skey} for {sample_name}")

                values_up = region_templates[f"{sample_name}_{skey}_up", :].values()
                values_down = region_templates[f"{sample_name}_{skey}_down", :].values()

                effect_up = np.ones_like(values_nominal)
                effect_down = np.ones_like(values_nominal)

                mask_up = values_up >= 0
                mask_down = values_down >= 0

                effect_up[mask & mask_up] = (
                    values_up[mask & mask_up] / values_nominal[mask & mask_up]
                )
                effect_down[mask & mask_down] = (
                    values_down[mask & mask_down] / values_nominal[mask & mask_down]
                )

                logger = logging.getLogger(
                    "validate_shapes_{}_{}_{}".format(region, sample_name, skey)
                )
                shape_checks(values_up, values_down, values_nominal, effect_up, effect_down, logger)

                logging.debug("nominal   : {nominal}".format(nominal=values_nominal))
                logging.debug("effect_up  : {effect_up}".format(effect_up=effect_up))
                logging.debug("effect_down: {effect_down}".format(effect_down=effect_down))

                sample.setParamEffect(shape_systs_dict[skey], effect_up, effect_down)

            ch.addSample(sample)

        if args.bblite:
            channel_name = region.split("Blinded")[0]
            ch.autoMCStats(channel_name=channel_name)

        # data observed
        ch.setObservation(region_templates[data_key, :])

    for blind_str in ["", "Blinded"]:
        # for blind_str in ["Blinded"]:
        passChName = f"pass{blind_str}".replace("_", "")
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


def shape_checks(values_up, values_down, values_nominal, effect_up, effect_down, logger):
    norm_up = np.sum(values_up)
    norm_down = np.sum(values_down)
    norm_nominal = np.sum(values_nominal)
    prob_up = values_up / norm_up
    prob_down = values_down / norm_down
    prob_nominal = values_nominal / norm_nominal
    shapeEffect_up = np.sum(
        np.abs(prob_up - prob_nominal) / (np.abs(prob_up) + np.abs(prob_nominal))
    )
    shapeEffect_down = np.sum(
        np.abs(prob_down - prob_nominal) / (np.abs(prob_down) + np.abs(prob_nominal))
    )

    valid = True
    if np.allclose(effect_up, 1.0) and np.allclose(effect_down, 1.0):
        valid = False
        logger.warning("No shape effect")
    elif np.allclose(effect_up, effect_down):
        valid = False
        logger.warning("Up is the same as Down, but different from nominal")
    elif np.allclose(effect_up, 1.0) or np.allclose(effect_down, 1.0):
        valid = False
        logger.warning("Up or Down is the same as nominal (one-sided)")
    elif shapeEffect_up < 0.001 and shapeEffect_down < 0.001:
        valid = False
        logger.warning("No genuine shape effect (just norm)")
    elif (norm_up > norm_nominal and norm_down > norm_nominal) or (
        norm_up < norm_nominal and norm_down < norm_nominal
    ):
        valid = False
        logger.warning("Up and Down vary norm in the same direction")

    if valid:
        logger.info("Shapes are valid")


def add_bool_arg(parser, name, help, default=False, no_name=None):
    """Add a boolean command line argument for argparse"""
    varname = "_".join(name.split("-"))  # change hyphens to underscores
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=varname, action="store_true", help=help)
    if no_name is None:
        no_name = "no-" + name
        no_help = "don't " + help
    else:
        no_help = help
    group.add_argument("--" + no_name, dest=varname, action="store_false", help=no_help)
    parser.set_defaults(**{varname: default})


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
    add_bool_arg(parser, "bblite", "use barlow-beeston-lite method", default=True)
    args = parser.parse_args()

    main(args)
