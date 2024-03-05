"""
General utilities for postprocessing.

Author: Raghav Kansal
"""

from __future__ import annotations

import contextlib
import pickle
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from os import listdir
from pathlib import Path

import hist
import numpy as np
import pandas as pd
from hist import Hist

from HHbbVV.hh_vars import (
    data_key,
    jec_shifts,
    jec_vars,
    jmsr_shifts,
    jmsr_vars,
    nonres_sig_keys,
    res_sig_keys,
)

MAIN_DIR = "./"
CUT_MAX_VAL = 9999.0


@dataclass
class ShapeVar:
    """Class to store attributes of a variable to make a histogram of.

    Args:
        var (str): variable name
        label (str): variable label
        bins (List[int]): [# num bins, min, max] if ``reg`` is False, else list of bin edges
        reg (bool, optional): Use a regular axis or variable binning. Defaults to True.
        blind_window (List[int], optional): if blinding, set min and max values to set 0. Defaults to None.
        significance_dir (str, optional): if plotting significance, which direction to plot it in.
          See more in plotting.py:ratioHistPlot(). Options are ["left", "right", "bin"]. Defaults to "right".
    """

    var: str = None
    label: str = None
    bins: list[int] = None
    reg: bool = True
    blind_window: list[int] = None
    significance_dir: str = "right"

    def __post_init__(self):
        # create axis used for histogramming
        if self.reg:
            self.axis = hist.axis.Regular(*self.bins, name=self.var, label=self.label)
        else:
            self.axis = hist.axis.Variable(self.bins, name=self.var, label=self.label)


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


@contextlib.contextmanager
def timer():
    old_time = time.monotonic()
    try:
        yield
    finally:
        new_time = time.monotonic()
        print(f"Time taken: {new_time - old_time} seconds")


def remove_empty_parquets(samples_dir, year):
    from os import listdir

    full_samples_list = listdir(f"{samples_dir}/{year}")
    print("Checking for empty parquets")

    for sample in full_samples_list:
        if sample == ".DS_Store":
            continue
        parquet_files = listdir(f"{samples_dir}/{year}/{sample}/parquet")
        for f in parquet_files:
            file_path = Path(f"{samples_dir}/{year}/{sample}/parquet/{f}")
            if not len(pd.read_parquet(file_path)):
                print("Removing: ", f"{sample}/{f}")
                file_path.unlink()


def remove_variation_suffix(var: str):
    """removes the variation suffix from the variable name"""
    if var.endswith("Down"):
        return var.split("Down")[0]
    elif var.endswith("Up"):
        return var.split("Up")[0]
    return var


def get_nevents(pickles_path, year, sample_name):
    """Adds up nevents over all pickles in ``pickles_path`` directory"""
    out_pickles = listdir(pickles_path)

    file_name = out_pickles[0]
    with Path(f"{pickles_path}/{file_name}").open("rb") as file:
        out_dict = pickle.load(file)
        nevents = out_dict[year][sample_name]["nevents"]  # index by year, then sample name

    for file_name in out_pickles[1:]:
        with Path(f"{pickles_path}/{file_name}").open("rb") as file:
            out_dict = pickle.load(file)
            nevents += out_dict[year][sample_name]["nevents"]

    return nevents


def get_cutflow(pickles_path, year, sample_name):
    """Accumulates cutflow over all pickles in ``pickles_path`` directory"""
    from coffea.processor.accumulator import accumulate

    out_pickles = [f for f in listdir(pickles_path) if f != ".DS_Store"]

    file_name = out_pickles[0]
    with Path(f"{pickles_path}/{file_name}").open("rb") as file:
        out_dict = pickle.load(file)
        cutflow = out_dict[year][sample_name]["cutflow"]  # index by year, then sample name

    for file_name in out_pickles[1:]:
        with Path(f"{pickles_path}/{file_name}").open("rb") as file:
            out_dict = pickle.load(file)
            cutflow = accumulate([cutflow, out_dict[year][sample_name]["cutflow"]])

    return cutflow


def get_pickles(pickles_path, year, sample_name):
    """Accumulates all pickles in ``pickles_path`` directory"""
    from coffea.processor.accumulator import accumulate

    out_pickles = [f for f in listdir(pickles_path) if f != ".DS_Store"]

    file_name = out_pickles[0]
    with Path(f"{pickles_path}/{file_name}").open("rb") as file:
        # out = pickle.load(file)[year][sample_name]  # TODO: uncomment and delete below
        out = pickle.load(file)[year]
        sample_name = next(iter(out.keys()))
        out = out[sample_name]

    for file_name in out_pickles[1:]:
        with Path(f"{pickles_path}/{file_name}").open("rb") as file:
            out_dict = pickle.load(file)[year][sample_name]
            out = accumulate([out, out_dict])

    return out


def check_selector(sample: str, selector: str | list[str]):
    if not (isinstance(selector, (list, tuple))):
        selector = [selector]

    for s in selector:
        if s.startswith("*"):
            if s[1:] in sample:
                return True
        else:
            if sample.startswith(s):
                return True

    return False


def _hem_cleaning(sample, events):
    if "ak8FatJetEta" not in events:
        warnings.warn("Can't do HEM cleaning!", stacklevel=2)
        return events

    if sample.startswith(("JetHT", "SingleMuon")):
        if sample.endswith(("2018C", "2018D")):
            hem_cut = np.any(
                (events["ak8FatJetEta"] > -3.2)
                & (events["ak8FatJetEta"] < -1.3)
                & (events["ak8FatJetPhi"] > -1.57)
                & (events["ak8FatJetPhi"] < -0.87),
                axis=1,
            )
            print(f"Removing {np.sum(hem_cut)} events")
            return events[~hem_cut]
        else:
            return events
    else:
        hem_cut = np.any(
            (events["ak8FatJetEta"] > -3.2)
            & (events["ak8FatJetEta"] < -1.3)
            & (events["ak8FatJetPhi"] > -1.57)
            & (events["ak8FatJetPhi"] < -0.87),
            axis=1,
        ) & (np.random.rand(len(events)) < 0.632)
        print(f"Removing {np.sum(hem_cut)} events")
        return events[~hem_cut]


def load_samples(
    data_dir: str,
    samples: dict[str, str],
    year: str,
    filters: list = None,
    columns: list = None,
    hem_cleaning: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Loads events with an optional filter.
    Reweights samples by nevents.

    Args:
        data_dir (str): path to data directory.
        samples (Dict[str, str]): dictionary of samples and selectors to load.
        year (str): year.
        filters (List): Optional filters when loading data.
        columns (List): Optional columns to load.
        hem_cleaning (bool): Whether to apply HEM cleaning to 2018 data.

    Returns:
        Dict[str, pd.DataFrame]: ``events_dict`` dictionary of events dataframe for each sample.

    """

    from os import listdir

    full_samples_list = listdir(f"{data_dir}/{year}")
    events_dict = {}

    for label, selector in samples.items():
        events_dict[label] = []
        for sample in full_samples_list:
            if not check_selector(sample, selector):
                continue

            # print(sample)
            # if sample.startswith("QCD") and not sample.endswith("_PSWeights_madgraph"):
            #     continue

            if not Path(f"{data_dir}/{year}/{sample}/parquet").exists():
                print(f"No parquet file for {sample}")
                continue

            # print(f"Loading {sample}")
            events = pd.read_parquet(
                f"{data_dir}/{year}/{sample}/parquet", filters=filters, columns=columns
            )
            not_empty = len(events) > 0
            pickles_path = f"{data_dir}/{year}/{sample}/pickles"

            if label != data_key:
                if label in nonres_sig_keys + res_sig_keys:
                    n_events = get_cutflow(pickles_path, year, sample)["has_4q"]
                else:
                    n_events = get_nevents(pickles_path, year, sample)

                if not_empty:
                    if "weight_noxsec" in events and np.all(
                        events["weight"] == events["weight_noxsec"]
                    ):
                        print(f"WARNING: {sample} has not been scaled by its xsec and lumi")

                    events["weight_nonorm"] = events["weight"]

                    if "weight_noTrigEffs" in events and not np.all(
                        np.isclose(events["weight"], events["weight_noTrigEffs"], rtol=1e-5)
                    ):
                        events["finalWeight"] = events["weight"] / n_events
                        events["finalWeight_noTrigEffs"] = events["weight_noTrigEffs"] / n_events
                    else:
                        events["weight"] /= n_events
            else:
                events["finalWeight"] = events["weight"]

            if year == "2018" and hem_cleaning:
                events = _hem_cleaning(sample, events)

            if not_empty:
                events_dict[label].append(events)

            print(f"Loaded {sample: <50}: {len(events)} entries")

        if len(events_dict[label]):
            events_dict[label] = pd.concat(events_dict[label])
        else:
            del events_dict[label]

    return events_dict


def add_to_cutflow(
    events_dict: dict[str, pd.DataFrame], key: str, weight_key: str, cutflow: pd.DataFrame
):
    cutflow[key] = [
        np.sum(events_dict[sample][weight_key]).squeeze() for sample in list(cutflow.index)
    ]


def getParticles(particle_list, particle_type):
    """
    Finds particles in `particle_list` of type `particle_type`

    Args:
        particle_list: array of particle pdgIds
        particle_type: can be 1) string: 'b', 'V' currently, or TODO: 2) pdgID, 3) list of pdgIds
    """

    B_PDGID = 5
    Z_PDGID = 23
    W_PDGID = 24

    if particle_type == "b":
        return abs(particle_list) == B_PDGID
    elif particle_type == "V":
        return (abs(particle_list) == W_PDGID) + (abs(particle_list) == Z_PDGID)


# check if string is an int
def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_feat(events: pd.DataFrame, feat: str, bb_mask: pd.DataFrame = None):
    if feat in events:
        return events[feat].to_numpy().squeeze()
    elif feat.startswith(("bb", "VV")):
        assert bb_mask is not None, "No bb mask given!"
        return events["ak8" + feat[2:]].to_numpy()[bb_mask ^ feat.startswith("VV")].squeeze()
    elif _is_int(feat[-1]):
        return events[feat[:-1]].to_numpy()[:, int(feat[-1])].squeeze()


def get_feat_first(events: pd.DataFrame, feat: str):
    return events[feat][0].to_numpy().squeeze()


def make_vector(events: dict, name: str, bb_mask: pd.DataFrame = None, mask=None):
    """
    Creates Lorentz vector from input events and beginning name, assuming events contain
      {name}Pt, {name}Phi, {name}Eta, {Name}Msd variables
    Optional input mask to select certain events

    Args:
        events (dict): dict of variables and corresponding numpy arrays
        name (str): object string e.g. ak8FatJet
        mask (bool array, optional): array selecting desired events
    """
    import vector

    if mask is None:
        return vector.array(
            {
                "pt": get_feat(events, f"{name}Pt", bb_mask),
                "phi": get_feat(events, f"{name}Phi", bb_mask),
                "eta": get_feat(events, f"{name}Eta", bb_mask),
                "M": (
                    get_feat(events, f"{name}Msd", bb_mask)
                    if f"{name}Msd" in events or f"ak8{name[2:]}Msd" in events
                    else get_feat(events, f"{name}Mass", bb_mask)
                ),
            }
        )
    else:
        return vector.array(
            {
                "pt": get_feat(events, f"{name}Pt", bb_mask)[mask],
                "phi": get_feat(events, f"{name}Phi", bb_mask)[mask],
                "eta": get_feat(events, f"{name}Eta", bb_mask)[mask],
                "M": (
                    get_feat(events, f"{name}Msd", bb_mask)[mask]
                    if f"{name}Msd" in events or f"ak8{name[2:]}Msd" in events
                    else get_feat(events, f"{name}Mass", bb_mask)[mask]
                ),
            }
        )


def get_key_index(h: Hist, axis_name: str):
    """Get the index of a key in a Hist's first axis"""
    return np.where(np.array(list(h.axes[0])) == axis_name)[0][0]


# TODO: extend to multi axis using https://stackoverflow.com/a/47859801/3759946 for 2D blinding
def blindBins(h: Hist, blind_region: list, blind_sample: str = None, axis=0):
    """
    Blind (i.e. zero) bins in histogram ``h``.
    If ``blind_sample`` specified, only blind that sample, else blinds all.
    """
    if axis > 0:
        raise Exception("not implemented > 1D blinding yet")

    bins = h.axes[axis + 1].edges
    lv = int(np.searchsorted(bins, blind_region[0], "right"))
    rv = int(np.searchsorted(bins, blind_region[1], "left") + 1)

    if blind_sample is not None:
        data_key_index = get_key_index(h, blind_sample)
        h.view(flow=True)[data_key_index][lv:rv].value = 0
        h.view(flow=True)[data_key_index][lv:rv].variance = 0
    else:
        h.view(flow=True)[:, lv:rv].value = 0
        h.view(flow=True)[:, lv:rv].variance = 0


def singleVarHist(
    events_dict: dict[str, pd.DataFrame],
    shape_var: ShapeVar,
    bb_masks: dict[str, pd.DataFrame],
    weight_key: str = "finalWeight",
    selection: dict = None,
) -> Hist:
    """
    Makes and fills a histogram for variable `var` using data in the `events` dict.

    Args:
        events (dict): a dict of events of format
          {sample1: {var1: np.array, var2: np.array, ...}, sample2: ...}
        shape_var (ShapeVar): ShapeVar object specifying the variable, label, binning, and (optionally) a blinding window.
        weight_key (str, optional): which weight to use from events, if different from 'weight'
        blind_region (list, optional): region to blind for data, in format [low_cut, high_cut].
          Bins in this region will be set to 0 for data.
        selection (dict, optional): if performing a selection first, dict of boolean arrays for
          each sample
    """
    samples = list(events_dict.keys())

    h = Hist(
        hist.axis.StrCategory(samples, name="Sample"),
        shape_var.axis,
        storage="weight",
    )

    var = shape_var.var

    for sample in samples:
        events = events_dict[sample]
        if sample == data_key and (var.endswith(("_up", "_down"))):
            fill_var = "_".join(var.split("_")[:-2])
        else:
            fill_var = var

        fill_data = {var: get_feat(events, fill_var, bb_masks[sample])}
        weight = events[weight_key].to_numpy().squeeze()

        if selection is not None:
            sel = selection[sample]
            fill_data[var] = fill_data[var][sel]
            weight = weight[sel]

        if len(fill_data[var]):
            h.fill(Sample=sample, **fill_data, weight=weight)

    if shape_var.blind_window is not None:
        blindBins(h, shape_var.blind_window, data_key)

    return h


def singleVarHistNoMask(
    events_dict: dict[str, pd.DataFrame],
    var: str,
    bins: list,
    label: str,
    weight_key: str = "finalWeight",
    blind_region: list = None,
    selection: dict = None,
) -> Hist:
    """
    Makes and fills a histogram for variable `var` using data in the `events` dict.

    Args:
        events (dict): a dict of events of format
          {sample1: {var1: np.array, var2: np.array, ...}, sample2: ...}
        var (str): variable inside the events dict to make a histogram of
        bins (list): bins in Hist format i.e. [num_bins, min_value, max_value]
        label (str): label for variable (shows up when plotting)
        weight_key (str, optional): which weight to use from events, if different from 'weight'
        blind_region (list, optional): region to blind for data, in format [low_cut, high_cut].
          Bins in this region will be set to 0 for data.
        selection (dict, optional): if performing a selection first, dict of boolean arrays for
          each sample
    """
    samples = list(events_dict.keys())

    h = Hist.new.StrCat(samples, name="Sample").Reg(*bins, name=var, label=label).Weight()

    for sample in samples:
        events = events_dict[sample]
        fill_data = {var: get_feat_first(events, var)}
        weight = events[weight_key].to_numpy().squeeze()

        if selection is not None:
            sel = selection[sample]
            fill_data[var] = fill_data[var][sel]
            weight = weight[sel]

        h.fill(Sample=sample, **fill_data, weight=weight)

    if blind_region is not None:
        blindBins(h, blind_region, data_key)

    return h


def add_selection(name, sel, selection, cutflow, events, weight_key):
    """Adds selection to PackedSelection object and the cutflow"""
    selection.add(name, sel)
    cutflow[name] = np.sum(events[weight_key][selection.all(*selection.names)])


def check_get_jec_var(var, jshift):
    """Checks if var is affected by the JEC / JMSR and if so, returns the shifted var name"""

    if jshift in jec_shifts and var in jec_vars:
        return var + "_" + jshift

    if jshift in jmsr_shifts and var in jmsr_vars:
        return var + "_" + jshift

    return var


def _var_selection(
    events: pd.DataFrame,
    bb_mask: pd.DataFrame,
    var: str,
    brange: list[float],
    MAX_VAL: float = CUT_MAX_VAL,
):
    """get selection for a single cut, including logic for OR-ing cut on two vars"""
    rmin, rmax = brange
    cut_vars = var.split("+")

    sels = []
    selstrs = []

    # OR the different vars
    for var in cut_vars:
        vals = get_feat(events, var, bb_mask)

        if rmin == -MAX_VAL:
            sels.append(vals < rmax)
            selstrs.append(f"{var} < {rmax}")
        elif rmax == MAX_VAL:
            sels.append(vals >= rmin)
            selstrs.append(f"{var} >= {rmin}")
        else:
            sels.append((vals >= rmin) & (vals < rmax))
            selstrs.append(f"{rmin} ≤ {var} < {rmax}")

    sel = np.sum(sels, axis=0).astype(bool)
    selstr = " or ".join(selstrs)

    return sel, selstr


def make_selection(
    var_cuts: dict[str, list[float]],
    events_dict: dict[str, pd.DataFrame],
    bb_masks: dict[str, pd.DataFrame],
    weight_key: str = "finalWeight",
    prev_cutflow: dict = None,
    selection: dict[str, np.ndarray] = None,
    jshift: str = "",
    MAX_VAL: float = CUT_MAX_VAL,
):
    """
    Makes cuts defined in `var_cuts` for each sample in `events`.

    Selection syntax:

    Simple cut:
    "var": [lower cut value, upper cut value]

    OR cut on `var`:
    "var": [[lower cut1 value, upper cut1 value], [lower cut2 value, upper cut2 value]] ...

    OR same cut(s) on multiple vars:
    "var1+var2": [lower cut value, upper cut value]

    TODO: OR more general cuts

    Args:
        var_cuts (dict): a dict of cuts, with each (key, value) pair = {var: [lower cut value, upper cut value], ...}.
        events (dict): a dict of events of format {sample1: {var1: np.array, var2: np.array, ...}, sample2: ...}
        weight_key (str): key to use for weights. Defaults to 'finalWeight'.
        prev_cutflow (dict): cutflow from previous cuts, if any. Defaults to None.
        selection (dict): previous selection, if any. Defaults to None.
        MAX_VAL (float): if abs of one of the cuts equals or exceeds this value it will be ignored. Defaults to 9999.

    Returns:
        selection (dict): dict of each sample's cut boolean arrays.
        cutflow (dict): dict of each sample's yields after each cut.
    """
    from coffea.analysis_tools import PackedSelection

    selection = {} if selection is None else deepcopy(selection)

    cutflow = {}

    for sample, events in events_dict.items():
        bb_mask = bb_masks[sample]
        if sample not in cutflow:
            cutflow[sample] = {}

        if sample in selection:
            new_selection = PackedSelection()
            new_selection.add("Previous selection", selection[sample])
            selection[sample] = new_selection
        else:
            selection[sample] = PackedSelection()

        for cutvar, branges in var_cuts.items():
            if jshift != "" and sample != data_key:
                var = check_get_jec_var(cutvar, jshift)
            else:
                var = cutvar

            if isinstance(branges[0], list):
                # OR the cuts
                sels = []
                selstrs = []
                for brange in branges:
                    sel, selstr = _var_selection(events, bb_mask, var, brange, MAX_VAL)
                    sels.append(sel)
                    selstrs.append(selstr)

                sel = np.sum(sels, axis=0).astype(bool)
                selstr = " or ".join(selstrs)

                add_selection(
                    selstr,
                    sel,
                    selection[sample],
                    cutflow[sample],
                    events,
                    weight_key,
                )
            else:
                sel, selstr = _var_selection(events, bb_mask, var, branges, MAX_VAL)
                add_selection(
                    selstr,
                    sel,
                    selection[sample],
                    cutflow[sample],
                    events,
                    weight_key,
                )

        selection[sample] = selection[sample].all(*selection[sample].names)

    cutflow = pd.DataFrame.from_dict(list(cutflow.values()))
    cutflow.index = list(events_dict.keys())

    if prev_cutflow is not None:
        cutflow = pd.concat((prev_cutflow, cutflow), axis=1)

    return selection, cutflow


def getSigSidebandBGYields(
    mass_key: str,
    sig_key: str,
    mass_cuts: list[int],
    events_dict: dict[str, pd.DataFrame],
    bb_masks: dict[str, pd.DataFrame],
    weight_key: str = "finalWeight",
    selection: dict = None,
):
    """
    Get signal and background yields in the `mass_cuts` range ([mass_cuts[0], mass_cuts[1]]),
    using the data in the sideband regions as the bg estimate
    """

    # get signal features
    sig_mass = get_feat(events_dict[sig_key], mass_key, bb_masks[sig_key])
    sig_weight = get_feat(events_dict[sig_key], weight_key, bb_masks[sig_key])

    if selection is not None:
        sig_mass = sig_mass[selection[sig_key]]
        sig_weight = sig_weight[selection[sig_key]]

    # get data features
    data_mass = get_feat(events_dict[data_key], mass_key, bb_masks[data_key])
    data_weight = get_feat(events_dict[data_key], weight_key, bb_masks[data_key])

    if selection is not None:
        data_mass = data_mass[selection[data_key]]
        data_weight = data_weight[selection[data_key]]

    # signal yield
    sig_cut = (sig_mass > mass_cuts[0]) * (sig_mass < mass_cuts[1])
    sig_yield = np.sum(sig_weight[sig_cut])

    # sideband regions
    mass_range = mass_cuts[1] - mass_cuts[0]
    low_mass_range = [mass_cuts[0] - mass_range / 2, mass_cuts[0]]
    high_mass_range = [mass_cuts[1], mass_cuts[1] + mass_range / 2]

    # get data yield in sideband regions
    low_data_cut = (data_mass > low_mass_range[0]) * (data_mass < low_mass_range[1])
    high_data_cut = (data_mass > high_mass_range[0]) * (data_mass < high_mass_range[1])
    bg_yield = np.sum(data_weight[low_data_cut]) + np.sum(data_weight[high_data_cut])

    return sig_yield, bg_yield


def getSignalPlotScaleFactor(
    events_dict: dict[str, pd.DataFrame],
    sig_keys: list[str],
    weight_key: str = "finalWeight",
    selection: dict = None,
):
    """Get scale factor for signals in histogram plots"""
    sig_scale_dict = {}

    if selection is None:
        data_sum = np.sum(events_dict[data_key][weight_key])
        for sig_key in sig_keys:
            sig_scale_dict[sig_key] = data_sum / np.sum(events_dict[sig_key][weight_key])
    else:
        data_sum = np.sum(events_dict[data_key][weight_key][selection[data_key]])
        for sig_key in sig_keys:
            sig_scale_dict[sig_key] = (
                data_sum / events_dict[sig_key][weight_key][selection[sig_key]]
            )

    return sig_scale_dict


def mxmy(sample):
    mY = int(sample.split("-")[-1])
    mX = int(sample.split("NMSSM_XToYHTo2W2BTo4Q2B_MX-")[1].split("_")[0])

    return (mX, mY)


def merge_dictionaries(dict1, dict2):
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    return merged_dict
