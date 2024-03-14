"""
Creates datacards for Higgs Combine using hist.Hist templates output from PostProcess.py
(1) Adds systematics for all samples,
(2) Sets up data-driven QCD background estimate ('rhalphabet' method)

Based on https://github.com/LPC-HH/combine-hh/blob/master/create_datacard.py

Authors: Raghav Kansal, Andres Nava
"""

from __future__ import annotations

# from utils import add_bool_arg
import argparse
import json
import logging
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import datacardHelpers as helpers
import numpy as np
import rhalphalib as rl
from datacardHelpers import (
    ShapeVar,
    Syst,
    add_bool_arg,
    combine_templates,
    get_channels,
    get_effect_updown,
    join_with_padding,
    mxmy,
    rem_neg,
    sum_templates,
)
from hist import Hist

from HHbbVV.hh_vars import LUMI, data_key, jecs, jmsr, qcd_key, res_sig_keys, years

try:
    rl.util.install_roofit_helpers()
    rl.ParametericSample.PreferRooParametricHist = False
except:
    print("rootfit install failed (not an issue for VBF)")

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
adjust_posdef_yields = False

parser = argparse.ArgumentParser()
parser.add_argument(
    "--templates-dir",
    default="",
    type=str,
    help="input pickle file of dict of hist.Hist templates",
)

add_bool_arg(parser, "vbf", "VBF category datacards", default=False)

add_bool_arg(parser, "sig-separate", "separate templates for signals and bgs", default=False)
add_bool_arg(parser, "do-jshifts", "Do JEC/JMC corrections.", default=True)

add_bool_arg(parser, "only-sm", "Only add SM HH samples for (for debugging nonres)", default=False)
add_bool_arg(
    parser,
    "combine-lasttwo",
    "Check combining last two bins in nonres for HWW review",
    default=False,
)

parser.add_argument("--cards-dir", default="cards", type=str, help="output card directory")

parser.add_argument("--mcstats-threshold", default=100, type=float, help="mcstats threshold n_eff")
parser.add_argument(
    "--epsilon",
    default=1e-2,
    type=float,
    help="epsilon to avoid numerical errs - also used to decide whether to add mc stats error",
)
parser.add_argument(
    "--scale-templates", default=None, type=float, help="scale all templates for bias tests"
)
parser.add_argument(
    "--min-qcd-val", default=1e-3, type=float, help="clip the pass QCD to above a minimum value"
)

parser.add_argument(
    "--sig-sample", default=None, type=str, help="can specify a specific signal key"
)

parser.add_argument(
    "--nTF",
    default=None,
    nargs="*",
    type=int,
    help="order of polynomial for TF in [dim 1, dim 2] = [mH(bb), -] for nonresonant or [mY, mX] for resonant."
    "Default is 0 for nonresonant and (1, 2) for resonant.",
)

parser.add_argument("--model-name", default=None, type=str, help="output model name")
parser.add_argument(
    "--year",
    help="year",
    type=str,
    default="all",
    choices=["2016APV", "2016", "2017", "2018", "all"],
)
add_bool_arg(parser, "mcstats", "add mc stats nuisances", default=True)
add_bool_arg(parser, "bblite", "use barlow-beeston-lite method", default=True)
add_bool_arg(parser, "resonant", "for resonant or nonresonant", default=False)
args = parser.parse_args()


CMS_PARAMS_LABEL = "CMS_bbWW_hadronic" if not args.resonant else "CMS_XHYbbWW_boosted"
qcd_data_key = "qcd_datadriven"

if args.nTF is None:
    args.nTF = [1, 2] if args.resonant else [0]

# (name in templates, name in cards)
mc_samples = OrderedDict(
    [
        ("TT", "ttbar"),
        ("W+Jets", "wjets"),
        ("Z+Jets", "zjets"),
        # ("Diboson", "diboson"),
        ("ST", "singletop"),
    ]
)

# TODO: float VJets normalization?

bg_keys = list(mc_samples.keys())
nonres_sig_keys_ggf = [
    "HHbbVV",
    "ggHH_kl_2p45_kt_1_HHbbVV",
    "ggHH_kl_5_kt_1_HHbbVV",
    "ggHH_kl_0_kt_1_HHbbVV",
]
nonres_sig_keys_vbf = [
    "VBFHHbbVV",
    "qqHH_CV_1_C2V_0_kl_1_HHbbVV",
    "qqHH_CV_1p5_C2V_1_kl_1_HHbbVV",
    "qqHH_CV_1_C2V_1_kl_2_HHbbVV",
    "qqHH_CV_1_C2V_2_kl_1_HHbbVV",
    "qqHH_CV_1_C2V_1_kl_0_HHbbVV",
    # "qqHH_CV_0p5_C2V_1_kl_1_HHbbVV",
]

if args.only_sm:
    nonres_sig_keys_ggf, nonres_sig_keys_vbf = ["HHbbVV"], []

nonres_sig_keys = nonres_sig_keys_ggf + nonres_sig_keys_vbf
sig_keys = []
hist_names = {}  # names of hist files for the samples

if args.resonant:
    if args.sig_sample != "":
        mX, mY = mxmy(args.sig_sample)
        mc_samples[f"X[{mX}]->H(bb)Y[{mY}](VV)"] = f"xhy_mx{mX}_my{mY}"
        hist_names[f"X[{mX}]->H(bb)Y[{mY}](VV)"] = f"NMSSM_XToYHTo2W2BTo4Q2B_MX-{mX}_MY-{mY}"
        sig_keys.append(f"X[{mX}]->H(bb)Y[{mY}](VV)")
    else:
        print("--sig-sample needs to be specified for resonant cards")
        sys.exit()
else:
    # change names to match HH combination convention
    for key in nonres_sig_keys:
        # check in case single sig sample is specified
        if args.sig_sample is None or key == args.sig_sample:
            if key == "HHbbVV":
                mc_samples["HHbbVV"] = "ggHH_kl_1_kt_1_hbbhww"
            elif key == "VBFHHbbVV":
                mc_samples["VBFHHbbVV"] = "qqHH_CV_1_C2V_1_kl_1_hbbhww"
            else:
                mc_samples[key] = key.replace("HHbbVV", "hbbhww")

            sig_keys.append(key)

all_mc = list(mc_samples.keys())

if args.year != "all":
    years = [args.year]
    full_lumi = LUMI[args.year]
else:
    full_lumi = np.sum(list(LUMI.values()))

rate_params = {
    sig_key: rl.IndependentParameter(f"{mc_samples[sig_key]}Rate", 1.0, 0, 1)
    for sig_key in sig_keys
}

# dictionary of nuisance params -> (modifier, samples affected by it, value)
nuisance_params = {
    # https://gitlab.cern.ch/hh/naming-conventions#experimental-uncertainties
    "lumi_13TeV_2016": Syst(
        prior="lnN", samples=all_mc, value=1.01 ** ((LUMI["2016"] + LUMI["2016APV"]) / full_lumi)
    ),
    "lumi_13TeV_2017": Syst(prior="lnN", samples=all_mc, value=1.02 ** (LUMI["2017"] / full_lumi)),
    "lumi_13TeV_2018": Syst(prior="lnN", samples=all_mc, value=1.015 ** (LUMI["2018"] / full_lumi)),
    "lumi_13TeV_correlated": Syst(
        prior="lnN",
        samples=all_mc,
        value=(
            (1.006 ** ((LUMI["2016"] + LUMI["2016APV"]) / full_lumi))
            * (1.009 ** (LUMI["2017"] / full_lumi))
            * (1.02 ** (LUMI["2018"] / full_lumi))
        ),
    ),
    "lumi_13TeV_1718": Syst(
        prior="lnN",
        samples=all_mc,
        value=((1.006 ** (LUMI["2017"] / full_lumi)) * (1.002 ** (LUMI["2018"] / full_lumi))),
    ),
    # https://gitlab.cern.ch/hh/naming-conventions#theory-uncertainties
    "BR_hbb": Syst(
        prior="lnN", samples=nonres_sig_keys + res_sig_keys, value=1.0124, value_down=0.9874
    ),
    "BR_hww": Syst(prior="lnN", samples=nonres_sig_keys, value=1.0153, value_down=0.9848),
    "pdf_gg": Syst(prior="lnN", samples=["TT"], value=1.042),
    "pdf_qqbar": Syst(prior="lnN", samples=["ST"], value=1.027),
    "pdf_Higgs_ggHH": Syst(prior="lnN", samples=nonres_sig_keys_ggf, value=1.030),
    "pdf_Higgs_qqHH": Syst(prior="lnN", samples=nonres_sig_keys_vbf, value=1.021),
    # TODO: add these if fitting single Higgs backgrounds
    # "pdf_Higgs_gg": Syst(prior="lnN", samples=ggfh_keys, value=1.019),
    "QCDscale_ttbar": Syst(
        prior="lnN",
        samples=["ST", "TT"],
        value={"ST": 1.03, "TT": 1.024},
        value_down={"ST": 0.978, "TT": 0.965},
        diff_samples=True,
    ),
    "QCDscale_qqHH": Syst(
        prior="lnN", samples=nonres_sig_keys_vbf, value=1.0003, value_down=0.9996
    ),
    # "QCDscale_ggH": Syst(
    #     prior="lnN",
    #     samples=ggfh_keys,
    #     value=1.039,
    # ),
    # "alpha_s": for single Higgs backgrounds
    # value will be added in from the systematics JSON
    f"{CMS_PARAMS_LABEL}_triggerEffSF_uncorrelated": Syst(
        prior="lnN", samples=all_mc, diff_regions=True
    ),
}

for sig_key in sig_keys:
    # values will be added in from the systematics JSON
    nuisance_params[f"{CMS_PARAMS_LABEL}_lp_sf_{mc_samples[sig_key]}"] = Syst(
        prior="lnN", samples=[sig_key]
    )

if args.year != "all":
    # remove other years' keys
    for key in [
        "lumi_13TeV_2016",
        "lumi_13TeV_2017",
        "lumi_13TeV_2018",
        "lumi_13TeV_correlated",
        "lumi_13TeV_1718",
    ]:
        if key != f"lumi_13TeV_{args.year}":
            del nuisance_params[key]

nuisance_params_dict = {
    param: rl.NuisanceParameter(param, syst.prior) for param, syst in nuisance_params.items()
}

# TODO: pileupID, lepton IDs (probably not necessary)

# dictionary of correlated shape systematics: name in templates -> name in cards, etc.
corr_year_shape_systs = {
    "FSRPartonShower": Syst(name="ps_fsr", prior="shape", samples=all_mc),
    "ISRPartonShower": Syst(name="ps_isr", prior="shape", samples=all_mc),
    "scale": Syst(
        name=f"{CMS_PARAMS_LABEL}_QCDScaleacc",
        prior="shape",
        samples=nonres_sig_keys,
        samples_corr=False,
    ),
    "pdf": Syst(
        name=f"{CMS_PARAMS_LABEL}_PDFacc",
        prior="shape",
        samples=nonres_sig_keys,
        samples_corr=False,
    ),
    # TODO: separate into individual
    "JES": Syst(name="CMS_scale_j", prior="shape", samples=all_mc),
    "txbb": Syst(
        name=f"{CMS_PARAMS_LABEL}_PNetHbbScaleFactors_correlated",
        prior="shape",
        samples=sig_keys,
        pass_only=True,
    ),
    # "top_pt": Syst(name="CMS_top_pT_reweighting", prior="shape", samples=["TT"])  # TODO
}

uncorr_year_shape_systs = {
    "pileup": Syst(name="CMS_pileup", prior="shape", samples=all_mc),
    # TODO: add 2016APV template into this
    "L1EcalPrefiring": Syst(
        name="CMS_l1_ecal_prefiring",
        prior="shape",
        samples=all_mc,
        uncorr_years=["2016APV", "2016", "2017"],
    ),
    "JER": Syst(name="CMS_res_j", prior="shape", samples=all_mc),
    "JMS": Syst(name=f"{CMS_PARAMS_LABEL}_jms", prior="shape", samples=all_mc),
    "JMR": Syst(name=f"{CMS_PARAMS_LABEL}_jmr", prior="shape", samples=all_mc),
}

if not args.do_jshifts:
    del corr_year_shape_systs["JES"]
    del uncorr_year_shape_systs["JER"]
    del uncorr_year_shape_systs["JMS"]
    del uncorr_year_shape_systs["JMR"]


shape_systs_dict = {}
for skey, syst in corr_year_shape_systs.items():
    if not syst.samples_corr:
        # separate nuisance param for each affected sample (TODO: propagate below)
        for sample in syst.samples:
            shape_systs_dict[f"{skey}_{sample}"] = rl.NuisanceParameter(
                f"{syst.name}_{mc_samples[sample]}", "shape"
            )
    else:
        shape_systs_dict[skey] = rl.NuisanceParameter(syst.name, "shape")
for skey, syst in uncorr_year_shape_systs.items():
    for year in years:
        if year in syst.uncorr_years:
            shape_systs_dict[f"{skey}_{year}"] = rl.NuisanceParameter(
                f"{syst.name}_{year}", "shape"
            )


def get_templates(
    templates_dir: Path,
    years: list[str],
    sig_separate: bool,
    scale: float = None,
    combine_lasttwo: bool = False,
):
    """Loads templates, combines bg and sig templates if separate, sums across all years"""
    templates_dict: dict[str, dict[str, Hist]] = {}

    if not sig_separate:
        # signal and background templates in same hist, just need to load and sum across years
        for year in years:
            with (templates_dir / f"{year}_templates.pkl").open("rb") as f:
                templates_dict[year] = rem_neg(pickle.load(f))
    else:
        # signal and background in different hists - need to combine them into one hist
        for year in years:
            with (templates_dir / f"backgrounds/{year}_templates.pkl").open("rb") as f:
                bg_templates = rem_neg(pickle.load(f))

            sig_templates = []

            for sig_key in sig_keys:
                with (templates_dir / f"{hist_names[sig_key]}/{year}_templates.pkl").open(
                    "rb"
                ) as f:
                    sig_templates.append(rem_neg(pickle.load(f)))

            templates_dict[year] = combine_templates(bg_templates, sig_templates)

    if scale is not None and scale != 1:
        for year in templates_dict:
            for key in templates_dict[year]:
                for j, sample in enumerate(templates_dict[year][key].axes[0]):
                    # only scale backgrounds / data
                    is_sig_key = False
                    for sig_key in sig_keys:
                        if sample.startswith(sig_key):
                            is_sig_key = True
                            break

                    if not is_sig_key:
                        vals = templates_dict[year][key][sample, ...].values()
                        variances = templates_dict[year][key][sample, ...].variances()
                        templates_dict[year][key].values()[j, ...] = vals * scale
                        templates_dict[year][key].variances()[j, ...] = variances * (scale**2)

    if combine_lasttwo:
        helpers.combine_last_two_bins(templates_dict, years)

    templates_summed: dict[str, Hist] = sum_templates(templates_dict, years)  # sum across years
    return templates_dict, templates_summed


def process_systematics_combined(systematics: dict):
    """Get total uncertainties from per-year systs in ``systematics``"""
    for sig_key in sig_keys:
        # already for all years
        nuisance_params[f"{CMS_PARAMS_LABEL}_lp_sf_{mc_samples[sig_key]}"].value = (
            1 + systematics[sig_key]["lp_sf_unc"]
        )

    tdict = {}
    for region in systematics[years[0]]:
        if len(years) > 1:
            trig_totals, trig_total_errs = [], []
            for year in years:
                trig_totals.append(systematics[year][region]["trig_total"])
                trig_total_errs.append(systematics[year][region]["trig_total_err"])

            trig_total = np.sum(trig_totals)
            trig_total_errs = np.linalg.norm(trig_total_errs)

            tdict[region] = 1 + (trig_total_errs / trig_total)
        else:
            year = years[0]
            tdict[region] = 1 + (
                systematics[year][region]["trig_total_err"]
                / systematics[year][region]["trig_total"]
            )

    nuisance_params[f"{CMS_PARAMS_LABEL}_triggerEffSF_uncorrelated"].value = tdict


def process_systematics_separate(bg_systematics: dict, sig_systs: dict[str, dict]):
    """Get total uncertainties from per-year systs separated into bg and sig systs"""
    for sig_key in sig_keys:
        # already for all years
        nuisance_params[f"{CMS_PARAMS_LABEL}_lp_sf_{mc_samples[sig_key]}"].value = (
            1 + sig_systs[sig_key][sig_key]["lp_sf_unc"]
        )

    # use only bg trig uncs.
    tdict = {}
    for region in bg_systematics[years[0]]:
        if len(years) > 1:
            trig_totals, trig_total_errs = [], []
            for year in years:
                trig_totals.append(bg_systematics[year][region]["trig_total"])
                trig_total_errs.append(bg_systematics[year][region]["trig_total_err"])

            trig_total = np.sum(trig_totals)
            trig_total_errs = np.linalg.norm(trig_total_errs)

            tdict[region] = 1 + (trig_total_errs / trig_total)
        else:
            year = years[0]
            tdict[region] = 1 + (
                bg_systematics[year][region]["trig_total_err"]
                / bg_systematics[year][region]["trig_total"]
            )

    nuisance_params[f"{CMS_PARAMS_LABEL}_triggerEffSF_uncorrelated"].value = tdict

    print("Nuisance Parameters\n", nuisance_params)


def process_systematics(templates_dir: str, sig_separate: bool):
    """Processes systematics based on whether signal and background JSONs are combined or not"""
    if not sig_separate:
        with (templates_dir / "systematics.json").open("r") as f:
            systematics = json.load(f)

        process_systematics_combined(systematics)  # LP SF and trig effs.
    else:
        with (templates_dir / "backgrounds/systematics.json").open("r") as f:
            bg_systematics = json.load(f)

        sig_systs = {}
        for sig_key in sig_keys:
            with (templates_dir / f"{hist_names[sig_key]}/systematics.json").open("r") as f:
                sig_systs[sig_key] = json.load(f)

        process_systematics_separate(bg_systematics, sig_systs)  # LP SF and trig effs.


# TODO: separate function for VBF?
def get_year_updown(
    templates_dict, sample, region, region_noblinded, blind_str, year, skey, mX_bin=None, vbf=False
):
    """
    Return templates with only the given year's shapes shifted up and down by the ``skey`` systematic.
    Returns as [up templates, down templates]
    """
    updown = []

    for shift in ["up", "down"]:
        sshift = f"{skey}_{shift}"
        # get nominal templates for each year
        templates = {y: templates_dict[y][region][sample, ...] for y in years}

        # replace template for this year with the shifted template
        if skey in jecs or skey in jmsr:
            # JEC/JMCs saved as different "region" in dict
            reg_name = (
                f"{region_noblinded}_{sshift}{blind_str}"
                if mX_bin is None
                else f"{region}_{sshift}"
            )

            templates[year] = templates_dict[year][reg_name][sample, ...]
        else:
            # weight uncertainties saved as different "sample" in dict
            templates[year] = templates_dict[year][region][f"{sample}_{sshift}", ...]

        if mX_bin is not None:
            for year, template in templates.items():
                templates[year] = template[:, mX_bin]

        # sum templates with year's template replaced with shifted
        if vbf:
            updown.append(sum([t.value for t in templates.values()]))
        else:
            updown.append(sum(list(templates.values())).values())

    return updown


def fill_regions(
    model: rl.Model,
    regions: list[str],
    templates_dict: dict,
    templates_summed: dict,
    mc_samples: dict[str, str],
    nuisance_params: dict[str, Syst],
    nuisance_params_dict: dict[str, rl.NuisanceParameter],
    corr_year_shape_systs: dict[str, Syst],
    uncorr_year_shape_systs: dict[str, Syst],
    shape_systs_dict: dict[str, rl.NuisanceParameter],
    bblite: bool = True,
    mX_bin: int = None,
):
    """Fill samples per region including given rate, shape and mcstats systematics.
    Ties "blinded" and "nonblinded" mc stats parameters together.

    Args:
        model (rl.Model): rhalphalib model
        regions (List[str]): list of regions to fill
        templates_dict (Dict): dictionary of all templates
        templates_summed (Dict): dictionary of templates summed across years
        mc_samples (Dict[str, str]): dict of mc samples and their names in the given templates -> card names
        nuisance_params (Dict[str, Tuple]): dict of nuisance parameter names and tuple of their
          (modifier, samples affected by it, value)
        nuisance_params_dict (Dict[str, rl.NuisanceParameter]): dict of nuisance parameter names
          and NuisanceParameter object
        corr_year_shape_systs (Dict[str, Tuple]): dict of shape systs (correlated across years)
          and tuple of their (name in cards, samples affected by it)
        uncorr_year_shape_systs (Dict[str, Tuple]): dict of shape systs (unccorrelated across years)
          and tuple of their (name in cards, samples affected by it)
        shape_systs_dict (Dict[str, rl.NuisanceParameter]): dict of shape syst names and
          NuisanceParameter object
        pass_only (List[str]): list of systematics which are only applied in the pass region(s)
        bblite (bool): use Barlow-Beeston-lite method or not (single mcstats param across MC samples)
        mX_bin (int): if doing 2D fit (for resonant), which mX bin to be filled
    """

    for region in regions:
        if mX_bin is None:
            region_templates = templates_summed[region]
        else:
            region_templates = templates_summed[region][:, :, mX_bin]

        pass_region = region.startswith("pass")
        region_noblinded = region.split("Blinded")[0]
        blind_str = "Blinded" if region.endswith("Blinded") else ""

        logging.info("starting region: %s" % region)
        binstr = "" if mX_bin is None else f"mXbin{mX_bin}"
        ch = rl.Channel(binstr + region.replace("_", ""))  # can't have '_'s in name
        model.addChannel(ch)

        for sample_name, card_name in mc_samples.items():
            # don't add signals in fail regions
            # also skip resonant signals in pass blinded - they are ignored in the validation fits anyway
            if sample_name in sig_keys and (
                not pass_region or (mX_bin is not None and region == "passBlinded")
            ):
                logging.info(f"\nSkipping {sample_name} in {region} region\n")
                continue

            # single top only in fail regions
            if sample_name == "ST" and pass_region:
                logging.info(f"\nSkipping ST in {region} region\n")
                continue

            logging.info("get templates for: %s" % sample_name)

            sample_template = region_templates[sample_name, :]

            stype = rl.Sample.SIGNAL if sample_name in sig_keys else rl.Sample.BACKGROUND
            sample = rl.TemplateSample(ch.name + "_" + card_name, stype, sample_template)

            # # rate params per signal to freeze them for individual limits
            # if stype == rl.Sample.SIGNAL and len(sig_keys) > 1:
            #     srate = rate_params[sample_name]
            #     sample.setParamEffect(srate, 1 * srate)

            # nominal values, errors
            values_nominal = np.maximum(sample_template.values(), 0.0)

            mask = values_nominal > 0
            errors_nominal = np.ones_like(values_nominal)
            errors_nominal[mask] = (
                1.0 + np.sqrt(sample_template.variances()[mask]) / values_nominal[mask]
            )

            logging.debug(f"nominal   : {values_nominal}")
            logging.debug(f"error     : {errors_nominal}")

            if not bblite and args.mcstats:
                # set mc stat uncs
                logging.info(f"setting autoMCStats for {sample_name} in {region}")

                # tie MC stats parameters together in blinded and "unblinded" region in nonresonant
                region_name = region if args.resonant else region_noblinded
                stats_sample_name = f"{CMS_PARAMS_LABEL}_{region_name}_{card_name}"
                sample.autoMCStats(
                    sample_name=stats_sample_name,
                    # this function uses a different threshold convention from combine
                    threshold=np.sqrt(1 / args.mcstats_threshold),
                    epsilon=args.epsilon,
                )

            # rate systematics
            for skey, syst in nuisance_params.items():
                if sample_name not in syst.samples or (not pass_region and syst.pass_only):
                    continue

                logging.info(f"Getting {skey} rate")

                param = nuisance_params_dict[skey]

                val, val_down = syst.value, syst.value_down
                if syst.diff_regions:
                    region_name = region if args.resonant else region_noblinded
                    val = val[region_name]
                    val_down = val_down[region_name] if val_down is not None else val_down
                if syst.diff_samples:
                    val = val[sample_name]
                    val_down = val_down[sample_name] if val_down is not None else val_down

                sample.setParamEffect(param, val, effect_down=val_down)

            # correlated shape systematics
            for skey, syst in corr_year_shape_systs.items():
                if sample_name not in syst.samples or (not pass_region and syst.pass_only):
                    continue

                logging.info(f"Getting {skey} shapes")

                if skey in jecs or skey in jmsr:
                    # JEC/JMCs saved as different "region" in dict
                    if mX_bin is None:
                        up_hist = templates_summed[f"{region_noblinded}_{skey}_up{blind_str}"][
                            sample_name, :
                        ]
                        down_hist = templates_summed[f"{region_noblinded}_{skey}_down{blind_str}"][
                            sample_name, :
                        ]
                    else:
                        # regions names are different from different blinding strats
                        up_hist = templates_summed[f"{region}_{skey}_up"][sample_name, :, mX_bin]
                        down_hist = templates_summed[f"{region}_{skey}_down"][
                            sample_name, :, mX_bin
                        ]

                    values_up = up_hist.values()
                    values_down = down_hist.values()
                else:
                    # weight uncertainties saved as different "sample" in dict
                    values_up = region_templates[f"{sample_name}_{skey}_up", :].values()
                    values_down = region_templates[f"{sample_name}_{skey}_down", :].values()

                logger = logging.getLogger(f"validate_shapes_{region}_{sample_name}_{skey}")

                effect_up, effect_down = get_effect_updown(
                    values_nominal, values_up, values_down, mask, logger, args.epsilon
                )

                # separate syst if not correlated across samples
                sdkey = skey if syst.samples_corr else f"{skey}_{sample_name}"
                sample.setParamEffect(shape_systs_dict[sdkey], effect_up, effect_down)

            # uncorrelated shape systematics
            for skey, syst in uncorr_year_shape_systs.items():
                if sample_name not in syst.samples or (not pass_region and syst.pass_only):
                    continue

                logging.info(f"Getting {skey} shapes")

                for year in years:
                    if year not in syst.uncorr_years:
                        continue

                    values_up, values_down = get_year_updown(
                        templates_dict,
                        sample_name,
                        region,
                        region_noblinded,
                        blind_str,
                        year,
                        skey,
                        mX_bin=mX_bin,
                    )
                    logger = logging.getLogger(f"validate_shapes_{region}_{sample_name}_{skey}")

                    effect_up, effect_down = get_effect_updown(
                        values_nominal, values_up, values_down, mask, logger, args.epsilon
                    )
                    sample.setParamEffect(
                        shape_systs_dict[f"{skey}_{year}"], effect_up, effect_down
                    )

            ch.addSample(sample)

        if bblite and args.mcstats:
            # tie MC stats parameters together in blinded and "unblinded" region in nonresonant
            channel_name = region if args.resonant else region_noblinded
            ch.autoMCStats(
                channel_name=f"{CMS_PARAMS_LABEL}_{channel_name}",
                threshold=args.mcstats_threshold,
                epsilon=args.epsilon,
            )

        # data observed
        ch.setObservation(region_templates[data_key, :])


def nonres_alphabet_fit(
    model: rl.Model,
    shape_vars: list[ShapeVar],
    templates_summed: dict,
    scale: float = None,
    min_qcd_val: float = None,
):
    shape_var = shape_vars[0]
    m_obs = rl.Observable(shape_var.name, shape_var.bins)

    # QCD overall pass / fail efficiency
    qcd_eff = (
        templates_summed["pass"][qcd_key, :].sum().value
        / templates_summed["fail"][qcd_key, :].sum().value
    )

    # transfer factor
    tf_dataResidual = rl.BasisPoly(
        f"{CMS_PARAMS_LABEL}_tf_dataResidual",
        (shape_var.order,),
        [shape_var.name],
        basis="Bernstein",
        limits=(-20, 20),
        square_params=True,
    )
    tf_dataResidual_params = tf_dataResidual(shape_var.scaled)
    tf_params_pass = qcd_eff * tf_dataResidual_params  # scale params initially by qcd eff

    # qcd params
    qcd_params = np.array(
        [
            rl.IndependentParameter(f"{CMS_PARAMS_LABEL}_tf_dataResidual_Bin{i}", 0)
            for i in range(m_obs.nbins)
        ]
    )

    for blind_str in ["", "Blinded"]:
        # for blind_str in ["Blinded"]:
        passChName = f"pass{blind_str}".replace("_", "")
        failChName = f"fail{blind_str}".replace("_", "")
        logging.info(
            f"setting transfer factor for pass region {passChName}, fail region {failChName}"
        )
        failCh = model[failChName]
        passCh = model[passChName]

        # sideband fail
        # was integer, and numpy complained about subtracting float from it
        initial_qcd = failCh.getObservation().astype(float)
        for sample in failCh:
            if sample.sampletype == rl.Sample.SIGNAL:
                continue
            logging.debug("subtracting %s from qcd" % sample._name)
            initial_qcd -= sample.getExpectation(nominal=True)

        if np.any(initial_qcd < 0.0):
            raise ValueError("initial_qcd negative for some bins..", initial_qcd)

        # idea here is that the error should be 1/sqrt(N), so parametrizing it as (1 + 1/sqrt(N))^qcdparams
        # will result in qcdparams errors ~±1
        # but because qcd is poorly modelled we're scaling sigma scale

        sigmascale = 10  # to scale the deviation from initial
        if scale is not None:
            sigmascale *= scale

        scaled_params = (
            initial_qcd * (1 + sigmascale / np.maximum(1.0, np.sqrt(initial_qcd))) ** qcd_params
        )

        # add samples
        fail_qcd = rl.ParametericSample(
            f"{failChName}_{CMS_PARAMS_LABEL}_qcd_datadriven",
            rl.Sample.BACKGROUND,
            m_obs,
            scaled_params,
        )
        failCh.addSample(fail_qcd)

        pass_qcd = rl.TransferFactorSample(
            f"{passChName}_{CMS_PARAMS_LABEL}_qcd_datadriven",
            rl.Sample.BACKGROUND,
            tf_params_pass,
            fail_qcd,
            min_val=min_qcd_val,
        )
        passCh.addSample(pass_qcd)


def res_alphabet_fit(
    model: rl.Model,
    shape_vars: list[ShapeVar],
    templates_summed: dict,
    scale: float = None,
    min_qcd_val: float = None,
):
    shape_var_mY, shape_var_mX = shape_vars
    m_obs = rl.Observable(shape_var_mY.name, shape_var_mY.bins)

    # QCD overall pass / fail efficiency
    qcd_eff = (
        templates_summed["pass"][qcd_key, ...].sum().value
        / templates_summed["fail"][qcd_key, ...].sum().value
    )

    # transfer factor
    tf_dataResidual = rl.BasisPoly(
        f"{CMS_PARAMS_LABEL}_tf_dataResidual",
        (shape_var_mX.order, shape_var_mY.order),
        [shape_var_mX.name, shape_var_mY.name],
        basis="Bernstein",
        limits=(-20, 20),
        square_params=True,
    )

    # based on https://github.com/nsmith-/rhalphalib/blob/9472913ef0bab3eb47bc942c1da4e00d59fb5202/tests/test_rhalphalib.py#L38
    mX_scaled_grid, mY_scaled_grid = np.meshgrid(
        shape_var_mX.scaled, shape_var_mY.scaled, indexing="ij"
    )
    # numpy array of
    tf_dataResidual_params = tf_dataResidual(mX_scaled_grid, mY_scaled_grid)
    tf_params_pass = qcd_eff * tf_dataResidual_params  # scale params initially by qcd eff

    for mX_bin in range(len(shape_var_mX.pts)):
        # qcd params
        qcd_params = np.array(
            [
                rl.IndependentParameter(f"{CMS_PARAMS_LABEL}_qcdparam_mXbin{mX_bin}_mYbin{i}", 0)
                for i in range(m_obs.nbins)
            ]
        )

        for blind_str in ["", "Blinded"]:
            # for blind_str in ["Blinded"]:
            passChName = f"mXbin{mX_bin}pass{blind_str}".replace("_", "")
            failChName = f"mXbin{mX_bin}fail{blind_str}".replace("_", "")
            logging.info(
                f"setting transfer factor for pass region {passChName}, fail region {failChName}"
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
                # raise ValueError("initial_qcd negative for some bins..", initial_qcd)
                logging.warning(f"initial_qcd negative for some bins... {initial_qcd}")
                initial_qcd[initial_qcd < 0] = 0

            # idea here is that the error should be 1/sqrt(N), so parametrizing it as (1 + 1/sqrt(N))^qcdparams
            # will result in qcdparams errors ~±1
            # but because qcd is poorly modelled we're scaling sigma scale

            sigmascale = 10  # to scale the deviation from initial
            if scale is not None:
                sigmascale *= np.sqrt(scale)

            scaled_params = (
                initial_qcd * (1 + sigmascale / np.maximum(1.0, np.sqrt(initial_qcd))) ** qcd_params
            )
            # scaled_params = initial_qcd * (1.1 ** qcd_params)

            # add samples
            fail_qcd = rl.ParametericSample(
                f"{failChName}_{CMS_PARAMS_LABEL}_qcd_datadriven",
                rl.Sample.BACKGROUND,
                m_obs,
                scaled_params,
            )
            failCh.addSample(fail_qcd)

            pass_qcd = rl.TransferFactorSample(
                f"{passChName}_{CMS_PARAMS_LABEL}_qcd_datadriven",
                rl.Sample.BACKGROUND,
                tf_params_pass[mX_bin, :],
                fail_qcd,
                min_val=min_qcd_val,
            )
            passCh.addSample(pass_qcd)


def createDatacardAlphabet(args, templates_dict, templates_summed, shape_vars):
    # (pass, fail) x (unblinded, blinded)
    regions: list[str] = [
        f"{pf}{blind_str}" for pf in ["pass", "fail"] for blind_str in ["", "Blinded"]
    ]

    # build actual fit model now
    model = rl.Model("HHModel" if not args.resonant else "XHYModel")

    # Fill templates per sample, incl. systematics
    fill_args = [
        model,
        regions,
        templates_dict,
        templates_summed,
        mc_samples,
        nuisance_params,
        nuisance_params_dict,
        corr_year_shape_systs,
        uncorr_year_shape_systs,
        shape_systs_dict,
        args.bblite,
    ]

    fit_args = [model, shape_vars, templates_summed, args.scale_templates, args.min_qcd_val]

    if args.resonant:
        # fill 1 channel per mX bin
        for i in range(len(shape_vars[1].pts)):
            logging.info(f"\n\nFilling templates for mXbin {i}")
            fill_regions(*fill_args, mX_bin=i)

        res_alphabet_fit(*fit_args)
    else:
        fill_regions(*fill_args)
        nonres_alphabet_fit(*fit_args)

    ##############################################
    # Save model
    ##############################################

    logging.info("Rendering combine model")

    out_dir = args.cards_dir / args.model_name if args.model_name is not None else args.cards_dir
    model.renderCombine(out_dir)

    with (out_dir / "model.pkl").open("wb") as fout:
        pickle.dump(model, fout, 2)  # use python 2 compatible protocol

    logging.info(f"Wrote model to {out_dir}")


def fill_yields(channels, channels_summed):
    """Fill top-level datacard info for VBF"""

    # dict storing all the substitutions for the datacard template
    datacard_dict = {
        "num_bins": len(channels),
        "num_bgs": len(bg_keys) + 1,  # + 1 for qcd
        "bins": join_with_padding(channels, padding=16),
        "observations": join_with_padding(
            [channels_summed[channel][data_key].value for channel in channels], padding=16
        ),
        "qcdlabel": qcd_data_key,
    }

    for i, label in enumerate(["A", "B", "C", "D"]):
        # channel labels
        datacard_dict[f"bin{label}"] = f"{channels[i]:<14}"

        # fill in data - MC yields for ABCD method
        if i > 0:
            data_obs = channels_summed[channels[i]][data_key].value
            mc_bg_yields = sum([channels_summed[channels[i]][key].value for key in bg_keys])
            datacard_dict[f"dataqcd{label}"] = data_obs - mc_bg_yields

    # collecting MC samples and yields per channel
    channel_bins_dict = {
        "bins_x_processes": [],
        "processes_per_bin": [],
        "processes_index": [],
        "processes_rates": [],
    }

    rates_dict = {}

    for channel in channels:
        rates_dict[channel] = {}
        for i, key in enumerate(sig_keys + bg_keys + [qcd_data_key]):
            channel_bins_dict["bins_x_processes"].append(channel)
            channel_bins_dict["processes_index"].append(i + 1 - len(sig_keys))
            if key == qcd_data_key:
                channel_bins_dict["processes_per_bin"].append(qcd_data_key)
                channel_bins_dict["processes_rates"].append(1)
            else:
                channel_bins_dict["processes_per_bin"].append(mc_samples[key])
                channel_bins_dict["processes_rates"].append(channels_summed[channel][key].value)
                rates_dict[channel][key] = channels_summed[channel][key]

    for key, arr in channel_bins_dict.items():
        datacard_dict[key] = join_with_padding(arr)

    return datacard_dict, rates_dict


def get_systematics_abcd(channels, channels_dict, channels_summed, rates_dict):
    channel_systs_dict = {}

    for region in channels:
        logging.info("starting region: %s" % region)

        channel_systs_dict[region] = {}

        pass_region = region.startswith("pass")
        region_nosidebands = region.split("_sidebands")[0]
        sideband_str = "_sidebands" if region.endswith("_sidebands") else ""

        # TODO: bblite
        channel_systs_dict[region]["mcstats"] = {}

        for sample_name, _card_name in mc_samples.items():
            systs_dict = {}
            channel_systs_dict[region][sample_name] = systs_dict

            # skip signal nuisances in non-signal region
            if sample_name in sig_keys and region != "pass":
                continue

            # MC stats
            mcstats_err = (
                np.sqrt(rates_dict[region][sample_name].variance)
                / rates_dict[region][sample_name].value
            )
            if mcstats_err > args.epsilon:
                channel_systs_dict[region]["mcstats"][sample_name] = 1.0 + mcstats_err

            # rate systematics
            for skey, syst in nuisance_params.items():
                if sample_name not in syst.samples or (not pass_region and syst.pass_only):
                    continue

                val, val_down = syst.value, syst.value_down
                if syst.diff_regions:
                    val = val[region_nosidebands]
                    val_down = val_down[region_nosidebands] if val_down is not None else val_down
                if syst.diff_samples:
                    val = val[sample_name]
                    val_down = val_down[sample_name] if val_down is not None else val_down

                systs_dict[skey] = (val, val_down)

            # correlated shape systematics
            for skey, syst in corr_year_shape_systs.items():
                if sample_name not in syst.samples or (not pass_region and syst.pass_only):
                    continue

                if skey in jecs or skey in jmsr:
                    # JEC/JMCs saved as different "region" in dict
                    val_up = channels_summed[f"{region_nosidebands}_{skey}_up{sideband_str}"][
                        sample_name
                    ].value
                    val_down = channels_summed[f"{region_nosidebands}_{skey}_down{sideband_str}"][
                        sample_name
                    ].value
                else:
                    # weight uncertainties saved as different "sample" in dict
                    val_up = channels_summed[region][f"{sample_name}_{skey}_up"].value
                    val_down = channels_summed[region][f"{sample_name}_{skey}_down"].value

                srate = rates_dict[region][sample_name].value
                systs_dict[skey] = (val_up / srate, val_down / srate)

            # uncorrelated shape systematics
            for skey, syst in uncorr_year_shape_systs.items():
                if sample_name not in syst.samples or (not pass_region and syst.pass_only):
                    continue

                # TODO: figure out why pileup is going crazy
                if skey == "pileup":
                    continue

                for year in years:
                    if year not in syst.uncorr_years:
                        continue

                    val_up, val_down = get_year_updown(
                        channels_dict,
                        sample_name,
                        region,
                        region_nosidebands,
                        sideband_str,
                        year,
                        skey,
                        vbf=True,
                    )

                    srate = rates_dict[region][sample_name].value
                    systs_dict[skey] = (val_up / srate, val_down / srate)

    syst_strs = []
    all_processes = sig_keys + bg_keys + [qcd_data_key]
    num_ps = len(all_processes)

    # mc stats
    for j, channel in enumerate(channels):
        cmcstats = channel_systs_dict[channel]["mcstats"]

        # mc stats error too low to add
        if cmcstats is None:
            continue

        # bblite criteria satisfied for this channel - single nuisance for all mc samples
        if isinstance(cmcstats, float):
            vals = []
            skey = f"mcstats_{channel}"
            sstr = f"{skey:<54}lnN   "
            vals += ["-"] * (j * num_ps)
            # add same nuisance for all processes except qcd
            vals += [cmcstats] * (num_ps - 1) + ["-"]
            vals += ["-"] * ((len(channels) - j - 1) * num_ps)
            sstr += join_with_padding(vals)
            syst_strs.append(sstr)

        # bblite not satisfied - separate nuisance for all mc samples
        else:
            for i, key in enumerate(sig_keys + bg_keys):
                if key in cmcstats:
                    vals = []
                    skey = f"mcstats_{channel}_{key}"
                    sstr = f"{skey:<54}lnN   "
                    # add single nuisance for this sample
                    vals += ["-"] * ((j * num_ps) + i)
                    vals += [cmcstats[key]]
                    vals += ["-"] * ((len(channels) - j - 1) * num_ps + (num_ps - i - 1))
                    sstr += join_with_padding(vals)
                    syst_strs.append(sstr)

    # all other nuisances
    for skey in (
        list(nuisance_params.keys())
        + list(corr_year_shape_systs.keys())
        + list(uncorr_year_shape_systs.keys())
    ):
        sstr = f"{skey:<54}lnN   "
        vals = []
        for channel in channels:
            for _, key in enumerate(all_processes):
                if key == qcd_data_key or skey not in channel_systs_dict[channel][key]:
                    vals.append("-")
                else:
                    val, val_down = channel_systs_dict[channel][key][skey]
                    val_str = val if val_down is None else f"{val_down}/{val}"
                    vals.append(val_str)

        sstr += join_with_padding(vals)
        syst_strs.append(sstr)

    syst_str = "\n".join(syst_strs)

    return syst_str


def createDatacardABCD(args, templates_dict, templates_summed, shape_vars):
    # A, B, C, D (in order)
    channels = ["pass", "pass_sidebands", "fail", "fail_sidebands"]  # analogous to regions
    channels_dict, channels_summed = get_channels(templates_dict, templates_summed, shape_vars[0])

    datacard_dict, rates_dict = fill_yields(channels, channels_summed)
    datacard_dict["systematics"] = get_systematics_abcd(
        channels, channels_dict, channels_summed, rates_dict
    )

    out_dir = args.cards_dir / args.model_name if args.model_name is not None else args.cards_dir

    with (out_dir / "datacard.txt").open("w") as f:
        f.write(helpers.abcd_datacard_template.substitute(datacard_dict))

    return


def main(args):
    args.templates_dir = Path(args.templates_dir)
    args.cards_dir = Path(args.cards_dir)

    # templates per region per year, templates per region summed across years
    templates_dict, templates_summed = get_templates(
        args.templates_dir, years, args.sig_separate, args.scale_templates, args.combine_lasttwo
    )

    # TODO: check if / how to include signal trig eff uncs. (rn only using bg uncs.)
    process_systematics(args.templates_dir, args.sig_separate)

    # random template from which to extract shape vars
    sample_templates: Hist = templates_summed[next(iter(templates_summed.keys()))]

    # [mH(bb)] for nonresonant, [mY, mX] for resonant
    shape_vars = [
        ShapeVar(name=axis.name, bins=axis.edges, order=args.nTF[i])
        for i, axis in enumerate(sample_templates.axes[1:])
    ]

    args.cards_dir.mkdir(parents=True, exist_ok=True)
    with (args.cards_dir / "templates.txt").open("w") as f:
        f.write(str(args.templates_dir.absolute()))

    dc_args = [args, templates_dict, templates_summed, shape_vars]
    if args.vbf:
        createDatacardABCD(*dc_args)
    else:
        createDatacardAlphabet(*dc_args)


main(args)
