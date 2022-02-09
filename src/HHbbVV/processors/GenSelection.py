"""
Gen selection functions for skimmer.

Author(s): Raghav Kansal
"""

import numpy as np
import awkward as ak

from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.analysis_tools import PackedSelection

from .utils import pad_val, add_selection


B_PDGID = 5
Z_PDGID = 23
W_PDGID = 24
HIGGS_PDGID = 25

HIGGS_FLAGS = ["fromHardProcess", "isLastCopy"]


def gen_selection_HHbbVV(
    self,
    events: NanoEventsArray,
    selection: PackedSelection,
    cutflow: dict,
    signGenWeights: ak.Array,
    skim_vars: dict,
):
    """Gets HH, bb, VV, and 4q 4-vectors + Higgs children information"""

    # finding the two gen higgs
    higgs = events.GenPart[
        (abs(events.GenPart.pdgId) == HIGGS_PDGID) * events.GenPart.hasFlags(HIGGS_FLAGS)
    ]

    # saving 4-vector info
    GenHiggsVars = {f"GenHiggs{key}": higgs[var].to_numpy() for (var, key) in skim_vars.items()}

    higgs_children = higgs.children

    # saving whether H->bb or H->VV
    GenHiggsVars["GenHiggsChildren"] = abs(higgs_children.pdgId[:, :, 0]).to_numpy()

    # finding bb and VV children
    is_bb = abs(higgs_children.pdgId) == B_PDGID
    is_VV = (abs(higgs_children.pdgId) == W_PDGID) + (abs(higgs_children.pdgId) == Z_PDGID)

    # checking that there are 2 b's and 2 V's
    has_bb = ak.sum(ak.flatten(is_bb, axis=2), axis=1) == 2
    has_VV = ak.sum(ak.flatten(is_VV, axis=2), axis=1) == 2

    # only select events with 2 b's and 2 V's
    add_selection("has_bbVV", has_bb * has_VV, selection, cutflow, False, signGenWeights)

    # saving bb and VV 4-vector info
    bb = ak.flatten(higgs_children[is_bb], axis=2)
    VV = ak.flatten(higgs_children[is_VV], axis=2)

    # have to pad to 2 because of some 4V events
    GenbbVars = {
        f"Genbb{key}": pad_val(bb[var], 2, -99999, axis=1) for (var, key) in skim_vars.items()
    }

    # selecting only up to the 2nd index because of some 4V events
    # (doesn't matter which two are selected since these events will be excluded anyway)
    GenVVVars = {f"GenVV{key}": VV[var][:, :2].to_numpy() for (var, key) in skim_vars.items()}

    # checking that each V has 2 q children
    VV_children = VV.children
    V_has_2q = ak.count(VV_children.pdgId, axis=2) == 2
    has_4q = ak.values_astype(ak.prod(V_has_2q, axis=1), np.bool)
    add_selection("has_4q", has_4q, selection, cutflow, False, signGenWeights)

    # saving 4q 4-vector info
    Gen4qVars = {
        f"Gen4q{key}": ak.to_numpy(
            ak.fill_none(
                ak.pad_none(
                    ak.pad_none(VV_children[var], 2, axis=1, clip=True), 2, axis=2, clip=True
                ),
                -99999,
            )
        )
        for (var, key) in skim_vars.items()
    }

    return {**GenHiggsVars, **GenbbVars, **GenVVVars, **Gen4qVars}


def gen_selection_HH4V(
    self,
    events: NanoEventsArray,
    selection: PackedSelection,
    cutflow: dict,
    signGenWeights: ak.Array,
    skim_vars: dict,
):
    """Gets HH, bb, VV, and 4q 4-vectors + Higgs children information"""

    # finding the two gen higgs
    higgs = events.GenPart[
        (abs(events.GenPart.pdgId) == HIGGS_PDGID) * events.GenPart.hasFlags(HIGGS_FLAGS)
    ]
    higgs_children = higgs.children

    GenHiggsVars = {f"GenHiggs{key}": higgs[var].to_numpy() for (var, key) in skim_vars.items()}
    GenHiggsVars["GenHiggsChildren"] = abs(higgs_children.pdgId[:, :, 0]).to_numpy()

    is_VV = (abs(higgs_children.pdgId) == W_PDGID) + (abs(higgs_children.pdgId) == Z_PDGID)
    has_2_VV = ak.sum(ak.sum(is_VV, axis=2) == 2, axis=1) == 2
    add_selection("has_2_VV", has_2_VV, selection, cutflow, False, signGenWeights)

    VV = ak.flatten(higgs_children[is_VV], axis=2)

    GenVVars = {f"GenV{key}": VV[var].to_numpy() for (var, key) in skim_vars.items()}

    VV_children = VV.children

    V_has_2q = ak.count(VV_children.pdgId, axis=2) == 2
    has_2_4q = ak.all(V_has_2q, axis=1)

    add_selection("has_2_4q", has_2_4q, selection, cutflow, False, signGenWeights)

    GenqVars = {
        f"Genq{key}": ak.to_numpy(
            ak.fill_none(
                ak.pad_none(
                    ak.pad_none(VV_children[var], 2, axis=1, clip=True), 2, axis=2, clip=True
                ),
                -99999,
            )
        )
        for (var, key) in skim_vars.items()
    }

    return {**GenHiggsVars, **GenVVars, **GenqVars}
