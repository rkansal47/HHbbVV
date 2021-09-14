import vector
from hist import Hist
import numpy as np


def add_bool_arg(parser, name, help, default=False, no_name=None):
    """ Add a boolean command line argument for argparse """
    varname = '_'.join(name.split('-'))  # change hyphens to underscores
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=varname, action='store_true', help=help)
    if(no_name is None):
        no_name = 'no-' + name
        no_help = "don't " + help
    else:
        no_help = help
    group.add_argument('--' + no_name, dest=varname, action='store_false', help=no_help)
    parser.set_defaults(**{varname: default})


def make_vector(events: dict, name: str, mask=None):
    """
    Creates Lorentz vector from input events and beginning name, assuming events contain {name}Pt, {name}Phi, {name}Eta, {Name}Msd variables
    Optional input mask to select certain events

    Args:
        events (dict): dict of variables and corresponding numpy arrays
        name (str): object string e.g. ak8FatJet
        mask (bool array, optional): array selecting desired events

    """

    if mask is None:
        return vector.array({
                                "pt": events[f'{name}Pt'],
                                "phi": events[f'{name}Phi'],
                                "eta": events[f'{name}Eta'],
                                "M": events[f'{name}Msd'] if f'{name}Msd' in events else events[f'{name}Mass'],
                            })
    else:
        return vector.array({
                                "pt": events[f'{name}Pt'][mask],
                                "phi": events[f'{name}Phi'][mask],
                                "eta": events[f'{name}Eta'][mask],
                                "M": events[f'{name}Msd'][mask] if f'{name}Msd' in events else events[f'{name}Mass'][mask],
                            })


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

    if particle_type == 'b':
        return abs(particle_list) == B_PDGID
    elif particle_type == 'V':
        return (abs(particle_list) == W_PDGID) + (abs(particle_list) == Z_PDGID)


def singleVarHist(events: dict, var: str, bins: list, label: str, weight_key: str = 'weight'):
    """
    Makes and fills a histogram for variable `var` using data in the `events` dict.

    Args:
        events (dict): a dict of events of format {sample1: {var1: np.array, var2: np.array, ...}, sample2: ...}
        var (str): variable inside the events dict to make a histogram of
        bins (list): bins in Hist format i.e. [num_bins, min_value, max_value]
        label (str): label for variable (shows up when plotting)
        weight_key (str, optional): which weight to use from events, if different from 'weight'
    """

    keys = list(events.keys())

    h = (
        Hist.new
        .StrCat(keys, name='Sample')
        .Reg(*bins, name=var, label=label)
        .Double()
    )

    for key in keys:
        h.fill(Sample=key, **{var: events[key][var]}, weight=events[key][weight_key])

    return h
