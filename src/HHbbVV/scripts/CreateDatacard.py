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
logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
adjust_posdef_yields = False

print("rl stuff")

# from utils import add_bool_arg
from sample_labels import qcd_key, data_key

LUMI = {"2017": 41.48}

# dictionary of nuisance params -> modifier
nuisance_params = {"lumi_13TeV_2017": "lnN"}

# dictionary of shape systematics -> name in cards
systs = OrderedDict([])

CMS_PARAMS_LABEL = "CMS_bbWW_boosted_ggf"
PARAMS_LABEL = "bbWW_boosted_ggf"


# for local interactive testing
args = type("test", (object,), {})()
args.data_dir = "../../../../data/skimmer/Apr28/"
args.plot_dir = "../../plots/05_26_testing"
args.year = "2017"
args.bdt_preds = f"{args.data_dir}/absolute_weights_preds.npy"
args.templates_file = "templates/test.pkl"
args.cat = "1"
args.nDataTF = 2


# import importlib
# importlib.reload(utils)
# th = Hist.new.StrCat([sig_key], name="Sample").Reg(10, -5, 5, name="msd").Double()
# str(type(th[sig_key, :]))
#
# th[sig_key, :].axes[0].name


def main(args):
    nuisance_params_dict = {
        param: rl.NuisanceParameter(param, unc) for param, unc in nuisance_params.items()
    }

    # pass, fail x unblinded, blinded
    regions = [
        f"{pf}_cat{args.cat}{blind_str}"
        for pf in ["pass", "fail"]
        for blind_str in ["", "_blinded"]
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
        templates[f"pass_cat{args.cat}"][qcd_key, :].sum().value
        / templates[f"fail_cat{args.cat}"][qcd_key, :].sum().value
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

    region = regions[0]

    for region in regions:
        region_templates = templates[region]

        logging.info("starting region: %s" % region)
        ch = rl.Channel(region.replace("_", ""))  # can't have '_'s in name
        model.addChannel(ch)

        # TODO: shape systs

        mc_samples = list(region_templates.axes[0])
        mc_samples.remove(data_key)

        for sample_name in mc_samples:
            logging.info("get templates for: %s" % sample_name)

            sample_template = region_templates[sample_name, :]

            stype = rl.Sample.SIGNAL if "HH" in sample_name else rl.Sample.BACKGROUND
            sample = rl.TemplateSample(ch.name + "_" + sample_name, stype, sample_template)

            # systematics
            sample.setParamEffect(nuisance_params_dict["lumi_13TeV_2017"], 1.02)

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
            sample.autoMCStats()

            # TODO: shape systematics
            ch.addSample(sample)

        # data observed
        ch.setObservation(region_templates[data_key, :])

    for blind_str in ["", "_blinded"]:
        passChName = f"pass_cat{args.cat}{blind_str}"
        failChName = f"fail_cat{args.cat}{blind_str}"
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

    os.system(f"mkdir -p {args.card_dir}")

    with open(os.path.join(str(args.card_dir), 'HHModel.pkl'), "wb") as fout:
        pickle.dump(model, fout, 2)  # use python 2 compatible protocol

    logging.info('rendering combine model')
    model.renderCombine(os.path.join(str(args.card_dir), 'HHModel'))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--templates-file",
        default="",
        type=str,
        help="input pickle file of dict of hist.Hist templates",
    )
    parser.add_argument("--card-dir", default="cards", type=str, help="output card directory")
    parser.add_argument("--cat", default="1", type=str, choices=["1"], help="category")
    parser.add_argument(
        "--nDataTF",
        default=2,
        type=int,
        dest="nDataTF",
        help="order of polynomial for TF from Data",
    )
    args = parser.parse_args()

    main(args)
