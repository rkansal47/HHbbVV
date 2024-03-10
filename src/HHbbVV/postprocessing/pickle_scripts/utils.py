"""
Common functions for the analysis.

Author(s): Raghav Kansal
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from hist import Hist

background_keys = ["V", "Top", "QCD"]
sig_key = "HHbbVV4q"
data_key = "Data"
all_keys = background_keys + [data_key, sig_key]

background_labels = ["VV/V+jets", "ST/TT", "QCD"]
sig_label = "HHbbVV4q"
data_label = "Data"
all_labels = background_labels + [data_label, sig_label]


def getAllKeys():
    return all_keys


def getSigKey():
    return sig_key


def getBackgroundKeys():
    return background_keys


def getSimKeys():
    return background_keys + [sig_key]


def getAllLabels():
    return all_labels


def getSigLabel():
    return sig_label


def getBackgroundLabels():
    return background_labels


def getSimLabels():
    return background_labels + [sig_label]


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


def load_events(
    data_path: str = "../../data/2017_combined/", keys: list = all_keys, do_print: bool = True
):
    """Load events for samples in `keys` from pickles in `data_path`, which must be named `key`.pkl"""
    import pickle

    events = {}

    for key in keys:
        if do_print:
            print(f"Loading {key} events")
        with open(f"{data_path}{key}.pkl", "rb") as file:
            events[key] = pickle.load(file)["skimmed_events"]

    # Just for checking
    if do_print:
        for key in keys:
            print(f"{key} events: {np.sum(events[key]['finalWeight']):.2f}")

    return events


def make_vector(events: dict, name: str, mask=None):
    """
    Creates Lorentz vector from input events and beginning name, assuming events contain {name}Pt, {name}Phi, {name}Eta, {Name}Msd variables
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
                "pt": events[f"{name}Pt"],
                "phi": events[f"{name}Phi"],
                "eta": events[f"{name}Eta"],
                "M": events[f"{name}Msd"] if f"{name}Msd" in events else events[f"{name}Mass"],
            }
        )
    else:
        return vector.array(
            {
                "pt": events[f"{name}Pt"][mask],
                "phi": events[f"{name}Phi"][mask],
                "eta": events[f"{name}Eta"][mask],
                "M": (
                    events[f"{name}Msd"][mask]
                    if f"{name}Msd" in events
                    else events[f"{name}Mass"][mask]
                ),
            }
        )


def getParticles(particle_list, particle_type):
    """
    Finds particles in `particle_list` of type `particle_type`

    Args:
        particle_list: array of particle pdgIds
        particle_type: can be 1) string: 'b', 'V' or 'H' currently, or TODO: 2) pdgID, 3) list of pdgIds
    """

    B_PDGID = 5
    Z_PDGID = 23
    W_PDGID = 24

    if particle_type == "b":
        return abs(particle_list) == B_PDGID
    elif particle_type == "V":
        return (abs(particle_list) == W_PDGID) + (abs(particle_list) == Z_PDGID)


def singleVarHist(
    events: dict,
    var: str,
    bins: list,
    label: str,
    weight_key: str = "finalWeight",
    blind_region: list = None,
    selection: dict = None,
):
    """
    Makes and fills a histogram for variable `var` using data in the `events` dict.

    Args:
        events (dict): a dict of events of format {sample1: {var1: np.array, var2: np.array, ...}, sample2: ...}
        var (str): variable inside the events dict to make a histogram of
        bins (list): bins in Hist format i.e. [num_bins, min_value, max_value]
        label (str): label for variable (shows up when plotting)
        weight_key (str, optional): which weight to use from events, if different from 'weight'
        blind_region (list, optional): region to blind for data, in format [low_cut, high_cut]. Bins in this region will be set to 0 for data.
        selection (dict, optional): if performing a selection first, dict of boolean arrays for each sample
    """
    keys = list(events.keys())

    h = Hist.new.StrCat(keys, name="Sample").Reg(*bins, name=var, label=label).Double()

    for key in keys:
        if selection is None:
            fill_data = {var: events[key][var]}
            weight = events[key][weight_key]
        else:
            fill_data = {var: events[key][var][selection[key]]}
            weight = events[key][weight_key][selection[key]]

        h.fill(Sample=key, **fill_data, weight=weight)

    if blind_region is not None:
        bins = h.axes[1].edges
        lv = int(np.searchsorted(bins, blind_region[0], "right"))
        rv = int(np.searchsorted(bins, blind_region[1], "left") + 1)

        data_key_index = np.where(np.array(list(h.axes[0])) == "Data")[0][0]
        h.view(flow=True)[data_key_index][lv:rv] = 0

    return h


def getSignalPlotScaleFactor(events: dict, weight_key: str = "finalWeight", selection: dict = None):
    """Get scale factor for signal in histogram plots"""
    if selection is None:
        return np.sum(events[data_key][weight_key]) / np.sum(events[sig_key][weight_key])
    else:
        return np.sum(events[data_key][weight_key][selection[data_key]]) / np.sum(
            events[sig_key][weight_key][selection[sig_key]]
        )


def add_selection(name, sel, selection, cutflow, events, weight_key):
    """Adds selection to PackedSelection object and the cutflow"""
    selection.add(name, sel)
    cutflow[name] = np.sum(events[weight_key][selection.all(*selection.names)])


def make_selection(
    var_cuts: dict,
    events: dict,
    weight_key: str = "finalWeight",
    cutflow: dict = None,
    selection: dict = None,
    MAX_VAL: float = 9999.0,
):
    """
    Makes cuts defined in `var_cuts` for each sample in `events`.

    Args:
        var_cuts (dict): a dict of cuts, with each (key, value) pair = (var, [lower cut value, upper cut value]).
        events (dict): a dict of events of format {sample1: {var1: np.array, var2: np.array, ...}, sample2: ...}
        weight_key (str): key to use for weights. Defaults to 'finalWeight'.
        cutflow (dict): cutflow from previous cuts, if any. Defaults to None.
        selection (dict): cutflow from previous selection, if any. Defaults to None.
        MAX_VAL (float): if abs of one of the cuts equals or exceeds this value it will be ignored. Defaults to 9999.

    Returns:
        selection (dict): dict of each sample's cut boolean arrays.
        cutflow (dict): dict of each sample's yields after each cut.
    """
    from coffea.processor import PackedSelection

    selection = {} if selection is None else deepcopy(selection)

    if cutflow is None:
        cutflow = {}

    for s, evts in events.items():
        if s not in cutflow:
            cutflow[s] = {}

        if s in selection:
            new_selection = PackedSelection()
            new_selection.add("Previous selection", selection[s])
            selection[s] = new_selection
        else:
            selection[s] = PackedSelection()

        for var, brange in var_cuts.items():
            if "+" in var:  # means OR-ing these cuts
                vars = var.split("+")

                if brange[0] > -MAX_VAL:
                    cut1 = evts[vars[0]] > brange[0]
                    for tvars in vars[1:]:
                        cut1 = cut1 + (evts[tvars] > brange[0])
                    add_selection(
                        f"{' or '.join(vars[:])} > {brange[0]}",
                        cut1,
                        selection[s],
                        cutflow[s],
                        evts,
                        weight_key,
                    )

                if brange[1] < MAX_VAL:
                    cut2 = evts[vars[0]] < brange[1]
                    for tvars in vars[1:]:
                        cut2 = cut2 + (evts[tvars] < brange[1])
                    add_selection(
                        f"{' or '.join(vars[:])} < {brange[1]}",
                        cut2,
                        selection[s],
                        cutflow[s],
                        evts,
                        weight_key,
                    )
            else:
                if brange[0] > -MAX_VAL:
                    add_selection(
                        f"{var} > {brange[0]}",
                        evts[var] > brange[0],
                        selection[s],
                        cutflow[s],
                        evts,
                        weight_key,
                    )
                if brange[1] < MAX_VAL:
                    add_selection(
                        f"{var} < {brange[1]}",
                        evts[var] < brange[1],
                        selection[s],
                        cutflow[s],
                        evts,
                        weight_key,
                    )

        selection[s] = selection[s].all(*selection[s].names)

    return selection, cutflow


def getSigSidebandBGYields(
    mass_key: str,
    mass_cuts: list,
    events: dict,
    weight_key: str = "finalWeight",
    selection: dict = None,
):
    """Get signal and background yields in the `mass_cuts` range ([mass_cuts[0], mass_cuts[1]]), using the data in the sideband regions as the bg estimate"""
    sig_mass = events[sig_key][mass_key]
    sig_weight = events[sig_key][weight_key]

    if selection is not None:
        sig_mass = sig_mass[selection[sig_key]]
        sig_weight = sig_weight[selection[sig_key]]

    data_mass = events[data_key][mass_key]
    data_weight = events[data_key][weight_key]

    if selection is not None:
        data_mass = data_mass[selection[data_key]]
        data_weight = data_weight[selection[data_key]]

    sig_cut = (sig_mass > mass_cuts[0]) * (sig_mass < mass_cuts[1])
    sig_yield = np.sum(sig_weight[sig_cut])

    # calculate bg estimate from data in sideband regions
    mass_range = mass_cuts[1] - mass_cuts[0]
    low_mass_range = [mass_cuts[0] - mass_range / 2, mass_cuts[0]]
    high_mass_range = [mass_cuts[1], mass_cuts[1] + mass_range / 2]

    low_data_cut = (data_mass > low_mass_range[0]) * (data_mass < low_mass_range[1])
    high_data_cut = (data_mass > high_mass_range[0]) * (data_mass < high_mass_range[1])
    bg_yield = np.sum(data_weight[low_data_cut]) + np.sum(data_weight[high_data_cut])

    return sig_yield, bg_yield
