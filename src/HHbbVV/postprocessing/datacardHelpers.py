from typing import Dict, List, Tuple, Union
from dataclasses import dataclass, field
import logging

import numpy as np
import hist
from hist import Hist
import utils


#################################################
# Common
#################################################


@dataclass
class Syst:
    """For storing info about systematics"""

    name: str = None
    prior: str = None  # e.g. "lnN", "shape", etc.

    # float if same value in all regions/samples, dictionary of values per region/sample if not
    # if both, region should be the higher level of the dictionary
    value: Union[float, Dict[str, float]] = None
    value_down: Union[float, Dict[str, float]] = None  # if None assumes symmetric effect
    # if the value is different for different regions or samples
    diff_regions: bool = False
    diff_samples: bool = False

    samples: List[str] = None  # samples affected by it
    # in case of uncorrelated unc., which years to split into
    uncorr_years: List[str] = field(default_factory=lambda: years)
    pass_only: bool = False  # is it applied only in the pass regions

    def __post_init__(self):
        if isinstance(self.value, dict) and not (self.diff_regions or self.diff_samples):
            raise RuntimeError(
                f"Value for systematic is a dictionary but neither ``diff_regions`` nor ``diff_samples`` is set."
            )


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


#################################################
# Template manipulation
#################################################


def combine_last_two_bins(templates_dict: Dict, years: List[str]):
    for year in years:
        for region in templates_dict[year]:
            templates_dict[year][region] = utils.rebin_hist(
                templates_dict[year][region],
                "bbFatJetParticleNetMass",
                list(range(50, 240, 10)) + [250],
            )


def rem_neg(template_dict: Dict):
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


def sum_templates(template_dict: Dict, years: List[str]):
    """Sum templates across years"""

    ttemplate = list(template_dict.values())[0]  # sample templates to extract values from
    combined = {}

    for region in ttemplate:
        thists = []

        for year in years:
            # temporary solution for case where 2018 templates don't have L1 prefiring in their axis
            thists.append(
                _match_samples(template_dict[year][region], template_dict["2017"][region])
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


#################################################
# Shape Fills
#################################################


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


#################################################
# VBF
#################################################


def join_with_padding(items, padding: int = 36):
    """Joins items into a string, padded by ``padding`` spaces"""
    ret = f"{{:<{padding}}}" * len(items)
    return ret.format(*items)


def rebin_channels(templates: Dict, axis_name: str, mass_window: List[float]):
    """Convert templates into single-bin hists per ABCD channel"""
    rebinning_edges = [50] + mass_window + [250]  # [0] [1e4]

    channels = {}
    for region, template in templates.items():
        # rebin into [left sideband, mass window, right window]
        rebinned = utils.rebin_hist(template, axis_name, rebinning_edges)
        channels[region] = rebinned[..., 1]
        # sum sidebands
        channels[f"{region}_sidebands"] = rebinned[..., 0] + rebinned[..., 2]

    return channels


def get_channels(templates_dict: Dict, templates_summed: Dict, shape_var: ShapeVar):
    """Convert templates into single-bin hists per ABCD channel"""

    mass_window = [100, 150]
    axis_name = shape_var.name

    channels_summed = rebin_channels(templates_summed, axis_name, mass_window)
    channels_dict = {}
    for key, templates in templates_dict.items():
        channels_dict[key] = rebin_channels(templates, axis_name, mass_window)

    return channels_dict, channels_summed


#################################################
# Resonant
#################################################


def mxmy(sample):
    mY = int(sample.split("-")[-1])
    mX = int(sample.split("NMSSM_XToYHTo2W2BTo4Q2B_MX-")[1].split("_")[0])

    return (mX, mY)
