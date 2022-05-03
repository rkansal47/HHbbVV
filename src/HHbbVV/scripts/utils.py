import time
import contextlib
from os import listdir
import pickle

import numpy as np
import pandas as pd

MAIN_DIR = "./"


@contextlib.contextmanager
def timer():
    old_time = time.monotonic()
    try:
        yield
    finally:
        new_time = time.monotonic()
        print(f"Time taken: {new_time - old_time} seconds")


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


def get_key(events: pd.DataFrame, key: str):
    return events[key].values


def make_vector(
    events: pd.DataFrame,
    name: str,
    mask=None,
):
    """
    Creates Lorentz vector from input events and beginning name, assuming events contain
    {name}Pt, {name}Phi, {name}Eta, {name}Msd variables
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
                "pt": get_key(events, f"{name}Pt"),
                "phi": get_key(events, f"{name}Phi"),
                "eta": get_key(events, f"{name}Eta"),
                "M": get_key(events, f"{name}Msd")
                if f"{name}Msd" in events
                else get_key(events, f"{name}Mass"),
            }
        )
    else:
        return vector.array(
            {
                "pt": get_key(events, f"{name}Pt")[mask],
                "phi": get_key(events, f"{name}Phi")[mask],
                "eta": get_key(events, f"{name}Eta")[mask],
                "M": get_key(events, f"{name}Msd")[mask]
                if f"{name}Msd" in events
                else get_key(events, f"{name}Mass")[mask],
            }
        )
