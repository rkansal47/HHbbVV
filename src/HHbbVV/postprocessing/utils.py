"""
General utilities for postprocessing.

Author: Raghav Kansal
"""

import time
import contextlib
from os import listdir
from os.path import exists
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd

from typing import Dict, List, Union

from coffea.analysis_tools import PackedSelection
from hist import Hist

from hh_vars import sig_key, data_key, jec_shifts, jmsr_shifts, jec_vars, jmsr_vars

MAIN_DIR = "./"
CUT_MAX_VAL = 9999.0


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
    from os import listdir, remove

    full_samples_list = listdir(f"{samples_dir}/{year}")
    print("Checking for empty parquets")

    for sample in full_samples_list:
        if sample == ".DS_Store":
            continue
        parquet_files = listdir(f"{samples_dir}/{year}/{sample}/parquet")
        for f in parquet_files:
            file_path = f"{samples_dir}/{year}/{sample}/parquet/{f}"
            if not len(pd.read_parquet(file_path)):
                print("Removing: ", f"{sample}/{f}")
                remove(file_path)


def get_xsecs():
    """Load cross sections json file and evaluate if necessary"""
    import json

    with open(f"{MAIN_DIR}/data/xsecs.json") as f:
        xsecs = json.load(f)

    for key, value in xsecs.items():
        if type(value) == str:
            xsecs[key] = eval(value)

    return xsecs


def get_nevents(pickles_path, year, sample_name):
    """Adds up nevents over all pickles in ``pickles_path`` directory"""
    out_pickles = listdir(pickles_path)

    file_name = out_pickles[0]
    with open(f"{pickles_path}/{file_name}", "rb") as file:
        out_dict = pickle.load(file)
        nevents = out_dict[year][sample_name]["nevents"]  # index by year, then sample name

    for file_name in out_pickles[1:]:
        with open(f"{pickles_path}/{file_name}", "rb") as file:
            out_dict = pickle.load(file)
            nevents += out_dict[year][sample_name]["nevents"]

    return nevents


def get_cutflow(pickles_path, year, sample_name):
    """Accumulates cutflow over all pickles in ``pickles_path`` directory"""
    from coffea.processor.accumulator import accumulate

    out_pickles = listdir(pickles_path)

    file_name = out_pickles[0]
    with open(f"{pickles_path}/{file_name}", "rb") as file:
        out_dict = pickle.load(file)
        cutflow = out_dict[year][sample_name]["cutflow"]  # index by year, then sample name

    for file_name in out_pickles[1:]:
        with open(f"{pickles_path}/{file_name}", "rb") as file:
            out_dict = pickle.load(file)
            cutflow = accumulate([cutflow, out_dict[year][sample_name]["cutflow"]])

    return cutflow


def check_selector(sample: str, selector: Union[str, List[str]]):
    if isinstance(selector, list) or isinstance(selector, tuple):
        for s in selector:
            if sample.startswith(s):
                return True
    else:
        if sample.startswith(selector):
            return True

    return False


def load_samples(
    data_dir: str, samples: Dict[str, str], year: str, filters: List = None
) -> Dict[str, pd.DataFrame]:
    """
    Loads events with an optional filter.
    Reweights samples by nevents.

    Args:
        data_dir (str): path to data directory.
        samples (Dict[str, str]): dictionary of samples and selectors to load.
        year (str): year.
        filters (List): Optional filters when loading data.

    Returns:
        Dict[str, pd.DataFrame]: ``events_dict`` dictionary of events dataframe for each sample.

    """

    from os import listdir

    full_samples_list = listdir(f"{data_dir}/{year}")
    events_dict = {}

    for label, selector in samples.items():
        # print(f"Finding {label} samples")
        events_dict[label] = []
        for sample in full_samples_list:
            if not check_selector(sample, selector):
                continue

            # print(sample)
            # if sample.startswith("QCD") and not sample.endswith("_PSWeights_madgraph"):
            #     continue

            if not exists(f"{data_dir}/{year}/{sample}/parquet"):
                print(f"No parquet file for {sample}")
                continue

            if sample in ["QCD_HT200to300", "WJetsToQQ_HT-200to400"]:
                print(f"WARNING: IGNORING {sample} because of empty df bug")
                continue

            # print(f"Loading {sample}")
            events = pd.read_parquet(f"{data_dir}/{year}/{sample}/parquet", filters=filters)
            not_empty = len(events) > 0
            pickles_path = f"{data_dir}/{year}/{sample}/pickles"

            if sample == "ZJetsToQQ_HT-200to400":
                print(f"WARNING: Normalising {sample} by hand")
                events["weight"] *= 1012.0 * 41480.0

            if label != data_key:
                if label == sig_key:
                    n_events = get_cutflow(pickles_path, year, sample)["has_4q"]
                else:
                    n_events = get_nevents(pickles_path, year, sample)

                if not_empty:
                    if "weight_noxsec" in events:
                        if np.all(events["weight"] == events["weight_noxsec"]):
                            print(f"WARNING: {sample} has not been scaled by its xsec and lumi")
                        else:
                            print("xsec check passed")

                    events["weight_nonorm"] = events["weight"]
                    events["weight"] /= n_events

            if not_empty:
                events_dict[label].append(events)

            print(f"Loaded {sample: <50}: {len(events)} entries")

        events_dict[label] = pd.concat(events_dict[label])

    return events_dict


def add_to_cutflow(
    events_dict: Dict[str, pd.DataFrame], key: str, weight_key: str, cutflow: pd.DataFrame
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


def get_feat(events: pd.DataFrame, feat: str, bb_mask: pd.DataFrame = None):
    if feat in events:
        return events[feat].values.squeeze()
    elif feat.startswith("bb") or feat.startswith("VV"):
        assert bb_mask is not None, "No bb mask given!"
        return events["ak8" + feat[2:]].values[bb_mask ^ feat.startswith("VV")].squeeze()


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
                "M": get_feat(events, f"{name}Msd", bb_mask)
                if f"{name}Msd" in events or f"ak8{name[2:]}Msd" in events
                else get_feat(events, f"{name}Mass", bb_mask),
            }
        )
    else:
        return vector.array(
            {
                "pt": get_feat(events, f"{name}Pt", bb_mask)[mask],
                "phi": get_feat(events, f"{name}Phi", bb_mask)[mask],
                "eta": get_feat(events, f"{name}Eta", bb_mask)[mask],
                "M": get_feat(events, f"{name}Msd", bb_mask)[mask]
                if f"{name}Msd" in events or f"ak8{name[2:]}Msd" in events
                else get_feat(events, f"{name}Mass", bb_mask)[mask],
            }
        )


def blindBins(h: Hist, blind_region: List, blind_sample: str = None):
    """
    Blind (i.e. zero) bins in histogram ``h``.
    If ``blind_sample`` specified, only blind that sample, else blinds all.
    """
    bins = h.axes[1].edges
    lv = int(np.searchsorted(bins, blind_region[0], "right"))
    rv = int(np.searchsorted(bins, blind_region[1], "left") + 1)

    if blind_sample is not None:
        data_key_index = np.where(np.array(list(h.axes[0])) == blind_sample)[0][0]
        h.view(flow=True)[data_key_index][lv:rv].value = 0
        h.view(flow=True)[data_key_index][lv:rv].variance = 0
    else:
        h.view(flow=True)[:, lv:rv].value = 0
        h.view(flow=True)[:, lv:rv].variance = 0


def singleVarHist(
    events_dict: Dict[str, pd.DataFrame],
    var: str,
    bins: list,
    label: str,
    bb_masks: Dict[str, pd.DataFrame],
    weight_key: str = "finalWeight",
    blind_region: List = None,
    selection: Dict = None,
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
        fill_data = {var: get_feat(events, var, bb_masks[sample])}
        weight = events[weight_key].values.squeeze()

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
    """Checks if var is affected by the JEC / JMSR and if so, returns the shfited var name"""

    if jshift in jec_shifts and var in jec_vars:
        return var + "_" + jshift

    if jshift in jmsr_shifts and var in jmsr_vars:
        return var + "_" + jshift

    return var


def make_selection(
    var_cuts: Dict[str, List[float]],
    events_dict: Dict[str, pd.DataFrame],
    bb_masks: Dict[str, pd.DataFrame],
    weight_key: str = "finalWeight",
    prev_cutflow: dict = None,
    selection: dict = None,
    jshift: str = "",
    MAX_VAL: float = CUT_MAX_VAL,
):
    """
    Makes cuts defined in `var_cuts` for each sample in `events`.

    Args:
        var_cuts (dict): a dict of cuts, with each (key, value) pair = (var, [lower cut value, upper cut value]).
        events (dict): a dict of events of format {sample1: {var1: np.array, var2: np.array, ...}, sample2: ...}
        weight_key (str): key to use for weights. Defaults to 'finalWeight'.
        prev_cutflow (dict): cutflow from previous cuts, if any. Defaults to None.
        selection (dict): cutflow from previous selection, if any. Defaults to None.
        MAX_VAL (float): if abs of one of the cuts equals or exceeds this value it will be ignored. Defaults to 9999.

    Returns:
        selection (dict): dict of each sample's cut boolean arrays.
        cutflow (dict): dict of each sample's yields after each cut.
    """

    if selection is None:
        selection = {}
    else:
        selection = deepcopy(selection)

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

        for var, brange in var_cuts.items():
            if jshift != "" and sample != data_key:
                var = check_get_jec_var(var, jshift)

            if brange[0] > -MAX_VAL:
                add_selection(
                    f"{var} > {brange[0]}",
                    get_feat(events, var, bb_mask) > brange[0],
                    selection[sample],
                    cutflow[sample],
                    events,
                    weight_key,
                )
            if brange[1] < MAX_VAL:
                add_selection(
                    f"{var} < {brange[1]}",
                    get_feat(events, var, bb_mask) < brange[1],
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
    mass_cuts: List[int],
    events_dict: Dict[str, pd.DataFrame],
    bb_masks: Dict[str, pd.DataFrame],
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
    events_dict: Dict[str, pd.DataFrame], weight_key: str = "finalWeight", selection: dict = None
):
    """Get scale factor for signal in histogram plots"""
    if selection is None:
        return np.sum(events_dict[data_key][weight_key]) / np.sum(events_dict[sig_key][weight_key])
    else:
        return np.sum(events_dict[data_key][weight_key][selection[data_key]]) / np.sum(
            events_dict[sig_key][weight_key][selection[sig_key]]
        )
