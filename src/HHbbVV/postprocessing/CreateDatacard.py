"""
Creates datacards for Higgs Combine using hist.Hist templates output from PostProcess.py
(1) Adds systematics for all samples,
(2) Sets up data-driven QCD background estimate ('rhalphabet' method)

Based on https://github.com/LPC-HH/combine-hh/blob/master/create_datacard.py

Author: Raghav Kansal
"""


import os
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import pickle, json
import logging
from collections import OrderedDict

import hist
from hist import Hist
import rhalphalib as rl

rl.util.install_roofit_helpers()
rl.ParametericSample.PreferRooParametricHist = False
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
adjust_posdef_yields = False

# from utils import add_bool_arg
from hh_vars import LUMI, res_sig_keys, qcd_key, data_key, years, jecs, jmsr, res_mps


import argparse


@dataclass
class Syst:
    """For storing info about systematics"""

    name: str = None
    prior: str = None  # e.g. "lnN", "shape", etc.
    # float if same value in all regions, dictionary of values per region if not
    value: Union[float, Dict[str, float]] = None
    samples: List[str] = None  # samples affected by it
    # in case of uncorrelated unc., which years to split into
    uncorr_years: List[str] = field(default_factory=lambda: years)
    pass_only: bool = False  # is it applied only in the pass regions


@dataclass
class ShapeVar:
    """For storing and calculating info about variables used in fit"""

    name: str = None
    bins: np.ndarray = None  # bin edges
    order: int = None  # TF order

    def __post_init__(self):
        # use bin centers for polynomial fit
        self.pts = self.bins[:-1] + 0.5 * np.diff(self.bins)
        # scale to be between [0, 1]
        self.scaled = (self.pts - self.bins[0]) / (self.bins[-1] - self.bins[0])


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


def mxmy(sample):
    mY = int(sample.split("-")[-1])
    mX = int(sample.split("NMSSM_XToYHTo2W2BTo4Q2B_MX-")[1].split("_")[0])

    return (mX, mY)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--templates-dir",
    default="",
    type=str,
    help="input pickle file of dict of hist.Hist templates",
)
add_bool_arg(parser, "sig-separate", "separate templates for signals and bgs", default=False)
add_bool_arg(parser, "do-jshifts", "Do JEC/JMC corrections.", default=True)

parser.add_argument("--cards-dir", default="cards", type=str, help="output card directory")

parser.add_argument("--mcstats-threshold", default=100, type=float, help="mcstats threshold n_eff")
parser.add_argument("--epsilon", default=1e-3, type=float, help="epsilon to avoid numerical errs")

parser.add_argument(
    "--sig-sample", default=None, type=str, help="can specify a specific signal key"
)

parser.add_argument(
    "--nTF",
    default=[1, 0],
    nargs="*",
    type=int,
    help="order of polynomial for TF in [dim 1, dim 2] = [mH(bb), -] for nonresonant or [mY, mX] for resonant",
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

# (name in templates, name in cards)
mc_samples = OrderedDict(
    [
        ("TT", "ttbar"),
        ("V+Jets", "vjets"),
        # ("Diboson", "diboson"),
        ("ST", "singletop"),
    ]
)

bg_keys = list(mc_samples.keys())
nonres_sig_keys_ggf = [
    "ggHH_kl_1_kt_1_HHbbVV",
    "ggHH_kl_2p45_kt_1_HHbbVV",
    "ggHH_kl_5_kt_1_HHbbVV",
    "ggHH_kl_0_kt_1_HHbbVV"
]
nonres_sig_keys_vbf = [
    "qqHH_CV_1_C2V_1_kl_1_HHbbVV",
    "qqHH_CV_1_C2V_0_kl_1_HHbbVV",
    "qqHH_CV_1p5_C2V_1_kl_1_HHbbVV",
    "qqHH_CV_1_C2V_1_kl_2_HHbbVV",
    "qqHH_CV_1_C2V_2_kl_1_HHbbVV",
    "qqHH_CV_1_C2V_2_kl_1_HHbbVV",
    "qqHH_CV_1_C2V_1_kl_0_HHbbVV",
    "qqHH_CV_0p5_C2V_1_kl_1_HHbbVV",
]
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
        res_mps = [(3000, 190), (1000, 100), (2600, 250)]
        for mX, mY in res_mps:
            mc_samples[f"X[{mX}]->H(bb)Y[{mY}](VV)"] = f"xhy_mx{mX}_my{mY}"
            hist_names[f"X[{mX}]->H(bb)Y[{mY}](VV)"] = f"NMSSM_XToYHTo2W2BTo4Q2B_MX-{mX}_MY-{mY}"
            sig_keys.append(f"X[{mX}]->H(bb)Y[{mY}](VV)")
else:
    for key in nonres_sig_keys:
        mc_samples[key] = key.replace("HHbbVV","hbbhww4q")
    sig_keys = nonres_sig_keys

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

# https://gitlab.cern.ch/hh/naming-conventions#experimental-uncertainties
# dictionary of nuisance params -> (modifier, samples affected by it, value)
nuisance_params = {
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
        value=((1.002 ** (LUMI["2017"] / full_lumi)) * (1.006 ** (LUMI["2018"] / full_lumi))),
    ),
    # value will be added in from the systematics JSON
    "triggerEffSF_uncorrelated": Syst(prior="lnN", samples=all_mc),
}

for sig_key in sig_keys:
    # values will be added in from the systematics JSON
    nuisance_params[f"lp_sf_{mc_samples[sig_key]}"] = Syst(prior="lnN", samples=[sig_key])

# remove keys in
if args.year != "all":
    for key in [
        "lumi_13TeV_2016",
        "lumi_13TeV_2017",
        "lumi_13TeV_2018",
        "lumi_13TeV_correlated",
        "lumi_13TeV_1718",
    ]:
        if key != f"lumi_13TeV{args.year}":
            del nuisance_params[key]

nuisance_params_dict = {
    param: rl.NuisanceParameter(param, syst.prior) for param, syst in nuisance_params.items()
}


# dictionary of correlated shape systematics: name in templates -> name in cards, etc.
corr_year_shape_systs = {
    "FSRPartonShower": Syst(name="ps_fsr", prior="shape", samples=nonres_sig_keys + ["V+Jets"]),
    "ISRPartonShower": Syst(name="ps_isr", prior="shape", samples=nonres_sig_keys + ["V+Jets"]),
    # TODO: should we be applying QCDscale for "others" process?
    # https://github.com/LPC-HH/HHLooper/blob/master/python/prepare_card_SR_final.py#L290
    "QCDscale": Syst(
        name="CMS_bbWW_boosted_ggf_ggHHQCDacc", prior="shape", samples=nonres_sig_keys_ggf
    ),
    "PDFalphaS": Syst(
        name="CMS_bbWW_boosted_ggf_ggHHPDFacc", prior="shape", samples=nonres_sig_keys_ggf
    ),
    # TODO: separate into individual
    "JES": Syst(name="CMS_scale_j", prior="shape", samples=all_mc),
    "txbb": Syst(
        name="CMS_bbWW_boosted_ggf_PNetHbbScaleFactors_correlated",
        prior="shape",
        samples=sig_keys,
        pass_only=True,
    ),
    # "top_pt": Syst(name="CMS_top_pT_reweighting", prior="shape", samples=["TT"])  # TODO
}

uncorr_year_shape_systs = {
    "pileup": Syst(name="CMS_pileup", prior="shape", samples=all_mc),
    # TODO: add 2016APV template into this
    # "L1EcalPrefiring": Syst(
    #     name="CMS_l1_ecal_prefiring", prior="shape", samples=all_mc, uncorr_years=["2016", "2017"]
    # ),
    "JER": Syst(name="CMS_res_j", prior="shape", samples=all_mc),
    "JMS": Syst(name="CMS_bbWW_boosted_ggf_jms", prior="shape", samples=all_mc),
    "JMR": Syst(name="CMS_bbWW_boosted_ggf_jmr", prior="shape", samples=all_mc),
}

if not args.do_jshifts:
    del corr_year_shape_systs["JES"]
    del uncorr_year_shape_systs["JER"]
    del uncorr_year_shape_systs["JMS"]
    del uncorr_year_shape_systs["JMR"]


shape_systs_dict = {}
for skey, syst in corr_year_shape_systs.items():
    shape_systs_dict[skey] = rl.NuisanceParameter(syst.name, "shape")
for skey, syst in uncorr_year_shape_systs.items():
    for year in years:
        if year in syst.uncorr_years:
            shape_systs_dict[f"{skey}_{year}"] = rl.NuisanceParameter(
                f"{syst.name}_{year}", "shape"
            )

CMS_PARAMS_LABEL = "CMS_bbWW_boosted_ggf" if not args.resonant else "CMS_XHYbbWW_boosted"
PARAMS_LABEL = "bbWW_boosted_ggf" if not args.resonant else "XHYbbWW_boosted"


def main(args):
    # (pass, fail) x (unblinded, blinded)
    regions: List[str] = [
        f"{pf}{blind_str}" for pf in [f"pass", "fail"] for blind_str in ["", "Blinded"]
    ]

    # templates per region per year, templates per region summed across years
    templates_dict, templates_summed = get_templates(args.templates_dir, years, args.sig_separate)

    # TODO: check if / how to include signal trig eff uncs. (rn only using bg uncs.)
    process_systematics(args.templates_dir, args.sig_separate)

    # random template from which to extract shape vars
    sample_templates: Hist = templates_summed[regions[0]]
    # [mH(bb)] for nonresonant, [mY, mX] for resonant
    shape_vars = [
        ShapeVar(name=axis.name, bins=axis.edges, order=args.nTF[i])
        for i, axis in enumerate(sample_templates.axes[1:])
    ]

    # build actual fit model now
    model = rl.Model("HHModel" if not args.resonant else "XHYModel")

    # Fill templates per sample, incl. systematics
    # TODO: blinding for resonant
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

    fit_args = [model, shape_vars, templates_summed]

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

    logging.info("rendering combine model")

    os.system(f"mkdir -p {args.cards_dir}")

    out_dir = (
        os.path.join(str(args.cards_dir), args.model_name)
        if args.model_name is not None
        else args.cards_dir
    )
    model.renderCombine(out_dir)

    with open(f"{out_dir}/model.pkl", "wb") as fout:
        pickle.dump(model, fout, 2)  # use python 2 compatible protocol


def _rem_neg(template_dict: Dict):
    for sample, template in template_dict.items():
        template.values()[template.values() < 0] = 0

    return template_dict


def _match_samples(template: Hist, match_template: Hist) -> Hist:
    """Temporary solution for case where 2018 templates don't have L1 prefiring in their axis

    Args:
        template (Hist): template to which samples may need to be added
        match_template (Hist): template from which to extract extra samples
    """
    samples = list(match_template.axes[0])

    # if template already has all the samples, don't do anything
    if list(template.axes[0]) == samples:
        return template

    # otherwise remake template with samples from ``match_template``
    h = hist.Hist(*match_template.axes, storage="weight")

    for sample in template.axes[0]:
        sample_index = np.where(np.array(list(h.axes[0])) == sample)[0][0]
        h.view()[sample_index] = template[sample, ...].view()

    return h


def sum_templates(template_dict: Dict):
    """Sum templates across years"""

    ttemplate = list(template_dict.values())[0]  # sample templates to extract values from
    combined = {}

    for region in ttemplate:
        thists = []

        for year in years:
            # temporary solution for case where 2018 templates don't have L1 prefiring in their axis
            thists.append(
                _match_samples(template_dict[year][region], template_dict["2016"][region])
            )

        combined[region] = sum(thists)

    return combined


def combine_templates(
    bg_templates: Dict[str, Hist], sig_templates: List[Dict[str, Hist]]
) -> Dict[str, Hist]:
    """
    Combines BG and signal templates into a single Hist (per region).

    Args:
        bg_templates (Dict[str, Hist]): dictionary of region -> Hist
        sig_templates (List[Dict[str, Hist]]): list of dictionaries of region -> Hist for each
          signal samples
    """
    ctemplates = {}

    for region, bg_template in bg_templates.items():
        # combined sig + bg samples
        csamples = list(bg_template.axes[0]) + [
            s for sig_template in sig_templates for s in list(sig_template[region].axes[0])
        ]

        # new hist with all samples
        ctemplate = Hist(
            hist.axis.StrCategory(csamples, name="Sample"),
            *bg_template.axes[1:],
            storage="weight",
        )

        # add background hists
        for sample in bg_template.axes[0]:
            sample_key_index = np.where(np.array(list(ctemplate.axes[0])) == sample)[0][0]
            ctemplate.view(flow=True)[sample_key_index, ...] = bg_template[sample, ...].view(
                flow=True
            )

        # add signal hists
        for st in sig_templates:
            sig_template = st[region]
            for sample in sig_template.axes[0]:
                sample_key_index = np.where(np.array(list(ctemplate.axes[0])) == sample)[0][0]
                ctemplate.view(flow=True)[sample_key_index, ...] = sig_template[sample, ...].view(
                    flow=True
                )

        ctemplates[region] = ctemplate

    return ctemplates


def get_templates(templates_dir: str, years: List[str], sig_separate: bool):
    """Loads templates, combines bg and sig templates if separate, sums across all years"""
    templates_dict: Dict[str, Dict[str, Hist]] = {}

    if not sig_separate:
        # signal and background templates in same hist, just need to load and sum across years
        for year in years:
            with open(f"{templates_dir}/{year}_templates.pkl", "rb") as f:
                templates_dict[year] = _rem_neg(pickle.load(f))
    else:
        # signal and background in different hists - need to combine them into one hist
        for year in years:
            with open(f"{templates_dir}/backgrounds/{year}_templates.pkl", "rb") as f:
                bg_templates = _rem_neg(pickle.load(f))

            sig_templates = []

            for sig_key in sig_keys:
                with open(f"{templates_dir}/{hist_names[sig_key]}/{year}_templates.pkl", "rb") as f:
                    sig_templates.append(_rem_neg(pickle.load(f)))

            templates_dict[year] = combine_templates(bg_templates, sig_templates)

    templates_summed: Dict[str, Hist] = sum_templates(templates_dict)  # sum across years
    return templates_dict, templates_summed


def process_systematics_combined(systematics: Dict):
    """Get total uncertainties from per-year systs in ``systematics``"""
    global nuisance_params
    for sig_key in sig_keys:
        # already for all years
        nuisance_params[f"lp_sf_{mc_samples[sig_key]}"].value = (
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

    nuisance_params["triggerEffSF_uncorrelated"].value = tdict

    print("Nuisance Parameters\n", nuisance_params)


def process_systematics_separate(bg_systematics: Dict, sig_systs: Dict[str, Dict]):
    """Get total uncertainties from per-year systs separated into bg and sig systs"""
    global nuisance_params

    for sig_key in sig_keys:
        # already for all years
        nuisance_params[f"lp_sf_{mc_samples[sig_key]}"].value = (
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

    nuisance_params["triggerEffSF_uncorrelated"].value = tdict

    print("Nuisance Parameters\n", nuisance_params)


def process_systematics(templates_dir: str, sig_separate: bool):
    """Processses systematics based on whether signal and background JSONs are combined or not"""
    if not sig_separate:
        with open(f"{templates_dir}/systematics.json", "r") as f:
            systematics = json.load(f)

        process_systematics_combined(systematics)  # LP SF and trig effs.
    else:
        with open(f"{templates_dir}/backgrounds/systematics.json", "r") as f:
            bg_systematics = json.load(f)

        sig_systs = {}
        for sig_key in sig_keys:
            with open(f"{templates_dir}/{hist_names[sig_key]}/systematics.json", "r") as f:
                sig_systs[sig_key] = json.load(f)

        process_systematics_separate(bg_systematics, sig_systs)  # LP SF and trig effs.


def fill_regions(
    model: rl.Model,
    regions: List[str],
    templates_dict: Dict,
    templates_summed: Dict,
    mc_samples: Dict[str, str],
    nuisance_params: Dict[str, Syst],
    nuisance_params_dict: Dict[str, rl.NuisanceParameter],
    corr_year_shape_systs: Dict[str, Syst],
    uncorr_year_shape_systs: Dict[str, Syst],
    shape_systs_dict: Dict[str, rl.NuisanceParameter],
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
            if sample_name in sig_keys:
                if not pass_region or (mX_bin is not None and region == "passBlinded"):
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

            # rate params per signal to freeze them for individual limits
            if stype == rl.Sample.SIGNAL and len(sig_keys) > 1:
                srate = rate_params[sample_name]
                sample.setParamEffect(srate, 1 * srate)

            # nominal values, errors
            values_nominal = np.maximum(sample_template.values(), 0.0)

            mask = values_nominal > 0
            errors_nominal = np.ones_like(values_nominal)
            errors_nominal[mask] = (
                1.0 + np.sqrt(sample_template.variances()[mask]) / values_nominal[mask]
            )

            logging.debug("nominal   : {nominal}".format(nominal=values_nominal))
            logging.debug("error     : {errors}".format(errors=errors_nominal))

            if not bblite and args.mcstats:
                # set mc stat uncs
                logging.info("setting autoMCStats for %s in %s" % (sample_name, region))

                # tie MC stats parameters together in blinded and "unblinded" region in nonresonant
                stats_sample_name = region if args.resonant else region_noblinded
                stats_sample_name += f"_{card_name}"
                sample.autoMCStats(
                    sample_name=stats_sample_name,
                    # this fn uses a different threshold convention from combine
                    threshold=np.sqrt(1 / args.mcstats_threshold),
                    epsilon=args.epsilon,
                )

            # rate systematics
            for skey, syst in nuisance_params.items():
                if sample_name not in syst.samples or (not pass_region and syst.pass_only):
                    continue

                logging.info(f"Getting {skey} rate")

                param = nuisance_params_dict[skey]
                val = syst.value[region_noblinded] if isinstance(syst.value, dict) else syst.value
                sample.setParamEffect(param, val)

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

                logger = logging.getLogger(
                    "validate_shapes_{}_{}_{}".format(region, sample_name, skey)
                )

                effect_up, effect_down = get_effect_updown(
                    values_nominal, values_up, values_down, mask, logger
                )
                sample.setParamEffect(shape_systs_dict[skey], effect_up, effect_down)

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
                    logger = logging.getLogger(
                        "validate_shapes_{}_{}_{}".format(region, sample_name, skey)
                    )

                    effect_up, effect_down = get_effect_updown(
                        values_nominal, values_up, values_down, mask, logger
                    )
                    sample.setParamEffect(
                        shape_systs_dict[f"{skey}_{year}"], effect_up, effect_down
                    )

            ch.addSample(sample)

        if bblite and args.mcstats:
            # tie MC stats parameters together in blinded and "unblinded" region in nonresonant
            channel_name = region if args.resonant else region_noblinded
            ch.autoMCStats(
                channel_name=channel_name, threshold=args.mcstats_threshold, epsilon=args.epsilon
            )

        # data observed
        ch.setObservation(region_templates[data_key, :])


def nonres_alphabet_fit(model: rl.Model, shape_vars: List[ShapeVar], templates_summed: Dict):
    shape_var = shape_vars[0]
    m_obs = rl.Observable(shape_var.name, shape_var.bins)

    # QCD overall pass / fail efficiency
    qcd_eff = (
        templates_summed[f"pass"][qcd_key, :].sum().value
        / templates_summed[f"fail"][qcd_key, :].sum().value
    )

    # transfer factor
    tf_dataResidual = rl.BasisPoly(
        f"{CMS_PARAMS_LABEL}_tf_dataResidual",
        (shape_var.order,),
        [shape_var.name],
        basis="Bernstein",
        limits=(-20, 20),
    )
    tf_dataResidual_params = tf_dataResidual(shape_var.scaled)
    tf_params_pass = qcd_eff * tf_dataResidual_params  # scale params initially by qcd eff

    # qcd params
    qcd_params = np.array(
        [
            rl.IndependentParameter(f"{CMS_PARAMS_LABEL}_qcdparam_msdbin{i}", 0)
            for i in range(m_obs.nbins)
        ]
    )

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
            if args.resonant and sample.sampletype == rl.Sample.SIGNAL:
                continue
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
            m_obs,
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


def res_alphabet_fit(model: rl.Model, shape_vars: List[ShapeVar], templates_summed: Dict):
    shape_var_mY, shape_var_mX = shape_vars
    m_obs = rl.Observable(shape_var_mY.name, shape_var_mY.bins)

    # QCD overall pass / fail efficiency
    qcd_eff = (
        templates_summed[f"pass"][qcd_key, ...].sum().value
        / templates_summed[f"fail"][qcd_key, ...].sum().value
    )

    # transfer factor
    tf_dataResidual = rl.BasisPoly(
        f"{CMS_PARAMS_LABEL}_tf_dataResidual",
        (shape_var_mX.order, shape_var_mY.order),
        [shape_var_mX.name, shape_var_mY.name],
        basis="Bernstein",
        limits=(-20, 20),
    )

    # based on https://github.com/nsmith-/rhalphalib/blob/9472913ef0bab3eb47bc942c1da4e00d59fb5202/tests/test_rhalphalib.py#L38
    mX_scaled_grid, mY_scaled_grid = np.meshgrid(
        shape_var_mX.scaled, shape_var_mY.scaled, indexing="ij"
    )
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
                "setting transfer factor for pass region %s, fail region %s"
                % (passChName, failChName)
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

            sigmascale = 10  # to scale the deviation from initial
            scaled_params = (
                initial_qcd * (1 + sigmascale / np.maximum(1.0, np.sqrt(initial_qcd))) ** qcd_params
            )

            # add samples
            fail_qcd = rl.ParametericSample(
                f"{failChName}_{PARAMS_LABEL}_qcd_datadriven",
                rl.Sample.BACKGROUND,
                m_obs,
                scaled_params,
            )
            failCh.addSample(fail_qcd)

            pass_qcd = rl.TransferFactorSample(
                f"{passChName}_{PARAMS_LABEL}_qcd_datadriven",
                rl.Sample.BACKGROUND,
                tf_params_pass[mX_bin, :],
                fail_qcd,
            )
            passCh.addSample(pass_qcd)


def _shape_checks(values_up, values_down, values_nominal, effect_up, effect_down, logger):
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


def get_effect_updown(values_nominal, values_up, values_down, mask, logger):
    effect_up = np.ones_like(values_nominal)
    effect_down = np.ones_like(values_nominal)

    mask_up = mask & (values_up >= 0)
    mask_down = mask & (values_down >= 0)

    effect_up[mask_up] = values_up[mask_up] / values_nominal[mask_up]
    effect_down[mask_down] = values_down[mask_down] / values_nominal[mask_down]

    zero_up = values_up == 0
    zero_down = values_down == 0

    effect_up[mask_up & zero_up] = values_nominal[mask_up & zero_up] * args.epsilon
    effect_down[mask_down & zero_down] = values_nominal[mask_down & zero_down] * args.epsilon

    _shape_checks(values_up, values_down, values_nominal, effect_up, effect_down, logger)

    logging.debug("nominal   : {nominal}".format(nominal=values_nominal))
    logging.debug("effect_up  : {effect_up}".format(effect_up=effect_up))
    logging.debug("effect_down: {effect_down}".format(effect_down=effect_down))

    return effect_up, effect_down


def get_year_updown(
    templates_dict, sample, region, region_noblinded, blind_str, year, skey, mX_bin=None
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
        # replace template for this year with the shifted tempalte
        if skey in jecs or skey in jmsr:
            # JEC/JMCs saved as different "region" in dict
            reg_name = (
                f"{region_noblinded}_{sshift}{blind_str}"
                if mX_bin is None
                else f"{region}_{sshift}"
            )
            templates[year] = templates_dict[year][reg_name][sample, :, ...]
        else:
            # weight uncertainties saved as different "sample" in dict
            templates[year] = templates_dict[year][region][f"{sample}_{sshift}", ...]

        if mX_bin is not None:
            for year, template in templates.items():
                templates[year] = template[:, mX_bin]

        # sum templates with year's template replaced with shifted
        updown.append(sum(list(templates.values())).values())

    return updown


main(args)
