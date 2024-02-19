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


PAD_VAL = -99999


def pad_val(
    arr: ak.Array,
    target: int,
    value: float = PAD_VAL,
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
    genWeights: ak.Array = None,
):
    """adds selection to PackedSelection object and the cutflow dictionary"""
    if isinstance(sel, ak.Array):
        sel = sel.to_numpy()

    selection.add(name, sel.astype(bool))
    cutflow[name] = (
        np.sum(selection.all(*selection.names))
        if isData
        # add up genWeights for MC
        else np.sum(genWeights[selection.all(*selection.names)])
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
                [
                    dicts_list[i][key].reshape(dicts_list[i][key].shape[0], -1)
                    for i in range(len(dicts_list))
                ],
                axis=1,
            )
            for key in dicts_list[0]
        }
    else:
        return dicts_list[0]


def select_dicts(dicts_list: List[Dict[str, np.ndarray]], sel: np.ndarray):
    """given a list of dicts of numpy arrays, select the entries per array across the lists according to ``sel``"""
    return {
        key: np.stack(
            [
                dicts_list[i][key].reshape(dicts_list[i][key].shape[0], -1)
                for i in range(len(dicts_list))
            ],
            axis=1,
        )[sel]
        for key in dicts_list[0]
    }


def remove_variation_suffix(var: str):
    """removes the variation suffix from the variable name"""
    if var.endswith("Down"):
        return var.split("Down")[0]
    elif var.endswith("Up"):
        return var.split("Up")[0]
    return var


class Weights:
    """Container for event weights and associated systematic shifts

    This container keeps track of correction factors and systematic
    effects that can be encoded as multiplicative modifiers to the event weight.
    All weights are stored in vector form.

    Parameters
    ----------
        size : int
            size of the weight arrays to be handled (i.e. the number of events / instances).
        storeIndividual : bool, optional
            store not only the total weight + variations, but also each individual weight.
            Default is false.
    """

    def __init__(self, size, storeIndividual=False):
        self._weight = np.ones(size)
        self._weights = {}
        self._modifiers = {}
        self._weightStats = {}
        self._storeIndividual = storeIndividual

    def add(self, name, weight, weightUp=None, weightDown=None, shift=False):
        """Add a new weight

        Adds a named correction to the event weight, and optionally also associated
        systematic uncertainties.

        Parameters
        ----------
            name : str
                name of correction
            weight : np.ndarray
                the nominal event weight associated with the correction
            weightUp : np.ndarray, optional
                weight with correction uncertainty shifted up (if available)
            weightDown : np.ndarray, optional
                weight with correction uncertainty shifted down. If ``weightUp`` is supplied, and
                the correction uncertainty is symmetric, this can be set to None to auto-calculate
                the down shift as ``1 / weightUp``.
            shift : bool, optional
                if True, interpret weightUp and weightDown as a relative difference (additive) to the
                nominal value

        .. note:: ``weightUp`` and ``weightDown`` are assumed to be rvalue-like and may be modified in-place by this function
        """
        if name.endswith("Up") or name.endswith("Down"):
            raise ValueError(
                "Avoid using 'Up' and 'Down' in weight names, instead pass appropriate shifts to add() call"
            )
        weight = np.array(weight)
        self._weight = self._weight * weight
        if self._storeIndividual:
            self._weights[name] = weight
        if weightUp is not None:
            weightUp = np.array(weightUp)
            if shift:
                weightUp += weight
            weightUp[weight != 0.0] /= weight[weight != 0.0]
            self._modifiers[name + "Up"] = weightUp
        if weightDown is not None:
            weightDown = np.array(weightDown)
            if shift:
                weightDown = weight - weightDown
            weightDown[weight != 0.0] /= weight[weight != 0.0]
            self._modifiers[name + "Down"] = weightDown
        self._weightStats[name] = {
            "sumw": weight.sum(),
            "sumw2": (weight**2).sum(),
            "min": weight.min(),
            "max": weight.max(),
            "n": weight.size,
        }

    def weight(self, modifier=None):
        """Current event weight vector

        Parameters
        ----------
            modifier : str, optional
                if supplied, provide event weight corresponding to a particular
                systematic uncertainty shift, of form ``str(name + 'Up')`` or (Down)

        Returns
        -------
            weight : np.ndarray
                The weight vector, possibly modified by the effect of a given systematic variation.
        """
        if modifier is None:
            return self._weight
        elif "Down" in modifier and modifier not in self._modifiers:
            return self._weight / self._modifiers[modifier.replace("Down", "Up")]
        return self._weight * self._modifiers[modifier]

    def partial_weight(self, include=[], exclude=[]):
        """Partial event weight vector

        Return a partial weight by multiplying a subset of all weights.
        Can be operated either by specifying weights to include or
        weights to exclude, but not both at the same time. The method
        can only be used if the individual weights are stored via the
        ``storeIndividual`` argument in the `Weights` initializer.

        Parameters
        ----------
            include : list
                Weight names to include, defaults to []
            exclude : list
                Weight names to exclude, defaults to []
        Returns
        -------
            weight : np.ndarray
                The weight vector, corresponding to only the effect of the
                corrections specified.
        """
        if not self._storeIndividual:
            raise ValueError(
                "To be able to request weight exclusion, use storeIndividual=True when creating Weights object."
            )
        if (include and exclude) or not (include or exclude):
            raise ValueError("Need to specify exactly one of the 'exclude' or 'include' arguments.")
        if include and not isinstance(include, list):
            raise ValueError("'include' should be a list of weight names")
        if exclude and not isinstance(exclude, list):
            raise ValueError("'exclude' should be a list of weight names")

        names = set(self._weights.keys())
        if include:
            names = names & set(include)
        if exclude:
            names = names - set(exclude)

        w = np.ones(self._weight.size)
        for name in names:
            w = w * self._weights[name]

        return w

    @property
    def variations(self):
        """List of available modifiers"""
        keys = set(self._modifiers.keys())
        # add any missing 'Down' variation
        for k in self._modifiers.keys():
            keys.add(k.replace("Up", "Down"))
        return keys
