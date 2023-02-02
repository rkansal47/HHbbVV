"""
Common functions for processors.

Author(s): Raghav Kansal
"""

import numpy as np
import awkward as ak

from coffea.analysis_tools import PackedSelection

from typing import List, Dict


P4 = {
    "eta": "Eta",
    "phi": "Phi",
    "mass": "Mass",
    "pt": "Pt",
}


def pad_val(
    arr: ak.Array,
    target: int,
    value: float,
    axis: int = 0,
    to_numpy: bool = True,
    clip: bool = True,
):
    """
    pads awkward array up to ``target`` index along axis ``axis`` with value ``value``,
    optionally converts to numpy array
    """
    ret = ak.fill_none(ak.pad_none(arr, target, axis=axis, clip=clip), value, axis=axis)
    return ret.to_numpy() if to_numpy else ret


def add_selection(
    name: str,
    sel: np.ndarray,
    selection: PackedSelection,
    cutflow: dict,
    isData: bool,
    signGenWeights: ak.Array,
):
    """adds selection to PackedSelection object and the cutflow dictionary"""
    if isinstance(sel, ak.Array):
        sel = sel.to_numpy()

    selection.add(name, sel.astype(bool))
    cutflow[name] = (
        np.sum(selection.all(*selection.names))
        if isData
        # add up sign of genWeights for MC
        else np.sum(signGenWeights[selection.all(*selection.names)])
    )


def add_selection_no_cutflow(
    name: str,
    sel: np.ndarray,
    selection: PackedSelection,
):
    """adds selection to PackedSelection object"""
    selection.add(name, ak.fill_none(sel, False))


def concatenate_dicts(dicts_list: List[Dict[str, np.ndarray]]):
    """given a list of dicts of numpy arrays, concatenates the numpy arrays across the lists"""
    if len(dicts_list) > 1:
        return {
            key: np.concatenate(
                [dicts_list[i][key][:, np.newaxis] for i in range(len(dicts_list))],
                axis=1,
            )
            for key in dicts_list[0]
        }
    else:
        return dicts_list[0]