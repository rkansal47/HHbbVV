"""
Gen selection functions for skimmer.

Author(s): Raghav Kansal, Cristina Mantilla Suarez
"""

import numpy as np
import awkward as ak

from coffea.analysis_tools import PackedSelection
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.nanoevents.methods.nanoaod import FatJetArray, GenParticleArray

from typing import List, Dict, Tuple, Union

from .utils import pad_val, add_selection


B_PDGID = 5
Z_PDGID = 23
W_PDGID = 24
HIGGS_PDGID = 25
ELE_PDGID = 11
MU_PDGID = 13
TAU_PDGID = 15
B_PDGID = 5

GEN_FLAGS = ["fromHardProcess", "isLastCopy"]


def gen_selection_HHbbVV(
    events: NanoEventsArray,
    selection: PackedSelection,
    cutflow: dict,
    signGenWeights: ak.Array,
    skim_vars: dict,
):
    """Gets HH, bb, VV, and 4q 4-vectors + Higgs children information"""

    # finding the two gen higgs
    higgs = events.GenPart[
        (abs(events.GenPart.pdgId) == HIGGS_PDGID) * events.GenPart.hasFlags(GEN_FLAGS)
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

    quarks = abs(VV_children.pdgId) <= B_PDGID
    all_q = ak.all(ak.all(quarks, axis=2), axis=1)
    add_selection("all_q", all_q, selection, cutflow, False, signGenWeights)

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
    events: NanoEventsArray,
    selection: PackedSelection,
    cutflow: dict,
    signGenWeights: ak.Array,
    skim_vars: dict,
):
    """Gets HH, VV, and 4q 4-vectors + Higgs children information"""

    # finding the two gen higgs
    higgs = events.GenPart[
        (abs(events.GenPart.pdgId) == HIGGS_PDGID) * events.GenPart.hasFlags(GEN_FLAGS)
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

    quarks = abs(VV_children.pdgId) <= B_PDGID
    all_q = ak.all(ak.all(quarks, axis=2), axis=1)
    add_selection("all_q", all_q, selection, cutflow, False, signGenWeights)

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


def gen_selection_HVV(
    events: NanoEventsArray,
    selection: PackedSelection,
    cutflow: dict,
    signGenWeights: ak.Array,
    skim_vars: dict,
):
    """Gets H, VV, and 4q 4-vectors + Higgs children information"""

    # finding the two gen higgs
    higgs = events.GenPart[
        (abs(events.GenPart.pdgId) == HIGGS_PDGID) * events.GenPart.hasFlags(GEN_FLAGS)
    ]
    higgs_children = higgs.children

    GenHiggsVars = {f"GenHiggs{key}": higgs[var].to_numpy() for (var, key) in skim_vars.items()}
    GenHiggsVars["GenHiggsChildren"] = abs(higgs_children.pdgId[:, :, 0]).to_numpy()

    is_VV = (abs(higgs_children.pdgId) == W_PDGID) + (abs(higgs_children.pdgId) == Z_PDGID)
    has_VV = ak.flatten(ak.sum(is_VV, axis=2) == 2)
    add_selection("has_VV", has_VV, selection, cutflow, False, signGenWeights)

    VV = ak.flatten(higgs_children[is_VV], axis=2)
    GenVVars = {f"GenV{key}": VV[var].to_numpy() for (var, key) in skim_vars.items()}

    VV_children = VV.children

    quarks = abs(VV_children.pdgId) <= B_PDGID
    all_q = ak.all(ak.all(quarks, axis=2), axis=1)
    add_selection("all_q", all_q, selection, cutflow, False, signGenWeights)

    V_has_2q = ak.count(VV_children.pdgId, axis=2) == 2
    V_has_2q = ak.all(V_has_2q, axis=1)
    add_selection("V_has_2q", V_has_2q, selection, cutflow, False, signGenWeights)

    GenqVars = {
        f"Genq{key}": pad_val(ak.flatten(VV_children.pt, axis=2), 4, 0, axis=1)
        for (var, key) in skim_vars.items()
    }

    return {**GenHiggsVars, **GenVVars, **GenqVars}


def get_pid_mask(
    genparts: GenParticleArray, pdgids: Union[int, list], ax: int = 2, byall: bool = True
) -> ak.Array:
    """
    Get selection mask for gen particles matching any of the pdgIds in ``pdgids``.
    If ``byall``, checks all particles along axis ``ax`` match.
    """
    gen_pdgids = abs(genparts.pdgId)

    if type(pdgids) == list:
        mask = gen_pdgids == pdgids[0]
        for pdgid in pdgids[1:]:
            mask = mask | (gen_pdgids == pdgid)
    else:
        mask = gen_pdgids == pdgids

    return ak.all(mask, axis=ax) if byall else mask


def to_label(array: ak.Array) -> ak.Array:
    return ak.values_astype(array, np.int32)


P4 = {
    "eta": "eta",
    "phi": "phi",
    "mass": "mass",
    "pt": "pt",
}


def tagger_gen_H_matching(
    genparts: GenParticleArray,
    fatjets: FatJetArray,
    genlabels: List[str],
    jet_dR: float,
    match_dR: float = 1.0,
    decays: str = "VV",
) -> Tuple[np.array, Dict[str, np.array]]:
    """Gen matching for Higgs samples, arguments as defined in ``tagger_gen_matching``"""

    higgs = genparts[
        get_pid_mask(genparts, HIGGS_PDGID, byall=False) * genparts.hasFlags(GEN_FLAGS)
    ]

    # find closest higgs
    matched_higgs = higgs[ak.argmin(fatjets.delta_r(higgs), axis=1, keepdims=True)]
    # select event only if distance to closest higgs is < ``match_dR``
    matched_higgs_mask = ak.any(fatjets.delta_r(matched_higgs) < match_dR, axis=1)
    # higgs kinematics
    genResVars = {
        f"fj_genRes_{key}": ak.fill_none(matched_higgs[var], -99999) for (var, key) in P4.items()
    }
    # Higgs parent kinematics
    bulkg = matched_higgs.distinctParent
    genXVars = {f"fj_genX_{key}": ak.fill_none(bulkg[var], -99999) for (var, key) in P4.items()}

    genVars = {**genResVars, **genXVars}

    matched_higgs_children = matched_higgs.children

    if "VV" in decays:
        # select only VV children
        children_mask = get_pid_mask(matched_higgs_children, [W_PDGID, Z_PDGID], byall=False)
        matched_higgs_children = matched_higgs_children[children_mask]

        children_mass = matched_higgs_children.mass

        # select lower mass child as V* and higher as V
        v_star = ak.firsts(matched_higgs_children[ak.argmin(children_mass, axis=2, keepdims=True)])
        v = ak.firsts(matched_higgs_children[ak.argmax(children_mass, axis=2, keepdims=True)])

        genVars["fj_dR_V"] = fatjets.delta_r(v)
        genVars["fj_dR_Vstar"] = fatjets.delta_r(v_star)

        # select event only if VV are within jet radius
        matched_Vs_mask = ak.any(fatjets.delta_r(v) < jet_dR, axis=1) & ak.any(
            fatjets.delta_r(v_star) < jet_dR, axis=1
        )

        # I think this will find all VV daughters - not just the ones from the Higgs matched to the fatjet? (e.g. for HH4W it'll find all 8 daughters?)
        # daughter_mask = get_pid_mask(genparts.distinctParent, [W_PDGID, Z_PDGID], ax=1, byall=False)
        # daughters = genparts[daughter_mask & genparts.hasFlags(GEN_FLAGS)]

        # get VV daughters
        daughters = ak.flatten(ak.flatten(matched_higgs_children.distinctChildren, axis=2), axis=2)
        daughters = daughters[daughters.hasFlags(GEN_FLAGS)]
        daughters_pdgId = abs(daughters.pdgId)

        nprongs = ak.sum(fatjets.delta_r(daughters) < jet_dR, axis=1)

        # why checking pT > 0?
        decay = (
            # 2 quarks * 1
            (ak.sum(daughters_pdgId <= B_PDGID, axis=1) == 2) * 1
            # 1 electron * 3
            + (ak.sum(daughters_pdgId == ELE_PDGID, axis=1) == 1) * 3
            # 1 muon * 5
            + (ak.sum(daughters_pdgId == MU_PDGID, axis=1) == 1) * 5
            # 1 tau * 7
            + (ak.sum(daughters_pdgId == TAU_PDGID, axis=1) == 1) * 7
            # 4 quarks * 11
            + (ak.sum(daughters_pdgId <= B_PDGID, axis=1) == 4) * 11
        )

        matched_mask = matched_higgs_mask & matched_Vs_mask

        genVVars = {f"fj_genV_{key}": ak.fill_none(v[var], -99999) for (var, key) in P4.items()}
        genVstarVars = {
            f"fj_genVstar_{key}": ak.fill_none(v_star[var], -99999) for (var, key) in P4.items()
        }
        genLabelVars = {
            "fj_nprongs": nprongs,
            "fj_H_VV_4q": to_label(decay == 11),
            "fj_H_VV_elenuqq": to_label(decay == 4),
            "fj_H_VV_munuqq": to_label(decay == 6),
            "fj_H_VV_taunuqq": to_label(decay == 8),
            "fj_H_VV_unmatched": to_label(~matched_mask),
        }
        genVars = {**genVars, **genVVars, **genVstarVars, **genLabelVars}

    return matched_mask, genVars


def tagger_gen_QCD_matching(
    genparts: GenParticleArray,
    fatjets: FatJetArray,
    genlabels: List[str],
    jet_dR: float,
    match_dR: float = 1.0,
) -> Tuple[np.array, Dict[str, np.array]]:

    """Gen matching for QCD samples, arguments as defined in ``tagger_gen_matching``"""
    partons = genparts[get_pid_mask(genparts, [21, 1, 2, 3, 4, 5], ax=1, byall=False)]
    matched_mask = ak.any(fatjets.delta_r(partons) < match_dR, axis=1)
    btoleptons = genparts[
        get_pid_mask(genparts, [511, 521, 523], ax=1, byall=False)
        & get_pid_mask(genparts.children, [11, 13])
    ]
    matched_b = ak.any(fatjets.delta_r(btoleptons) < 0.5, axis=1)
    genLabelVars = {
        "fj_isQCDb": to_label(matched_mask & (fatjets.nBHadrons == 1) & ~matched_b),
        "fj_isQCDbb": to_label(matched_mask & (fatjets.nBHadrons > 1) & ~matched_b),
        "fj_isQCDc": to_label(matched_mask & (fatjets.nCHadrons == 1)),
        "fj_isQCDcc": to_label(matched_mask & (fatjets.nCHadrons > 1)),
        "fj_isQCDlep": to_label(matched_mask & matched_b),
    }
    genLabelVars["fj_isQCDothers"] = to_label(
        matched_mask & (fatjets.nBHadrons == 0) & (fatjets.nBHadrons == 0) & ~matched_b
    )

    return matched_mask, genLabelVars


def get_genjet_vars(
    events: NanoEventsArray, fatjets: FatJetArray, ak15: bool = True, match_dR: float = 1.0
):
    """Matched fat jet to gen-level jet and gets gen jet vars"""
    GenJetVars = {}

    if ak15:
        sdgen_dr = fatjets.delta_r(events.SoftDropGenJetAK15)
        # get closest gen jet
        matched_sdgen_jet = events.SoftDropGenJetAK15[ak.argmin(sdgen_dr, axis=1, keepdims=True)]
        # check that it is within ``match_dR`` of fat jet
        matched_sdgen_jet_mask = fatjets.delta_r(matched_sdgen_jet) < match_dR
        GenJetVars["fj_genjetmsd"] = matched_sdgen_jet.mass * ak.values_astype(
            matched_sdgen_jet_mask, int
        )

        gen_dr = fatjets.delta_r(events.GenJetAK15)
        matched_gen_jet = events.GenJetAK15[ak.argmin(gen_dr, axis=1, keepdims=True)]
        matched_gen_jet_mask = fatjets.delta_r(matched_gen_jet) < match_dR
        GenJetVars["fj_genjetmass"] = matched_gen_jet.mass * ak.values_astype(
            matched_gen_jet_mask, int
        )
    else:
        # NanoAOD automatically matched ak8 fat jets
        # No soft dropped gen jets however
        GenJetVars["fj_genjetmass"] = fatjets.matched_gen.mass

    return GenJetVars


def tagger_gen_matching(
    events: NanoEventsArray,
    genparts: GenParticleArray,
    fatjets: FatJetArray,
    genlabels: List[str],
    label: str,
    match_dR: float = 1.0,
) -> Tuple[np.array, Dict[str, np.array]]:
    """Does fatjet -> gen-level matching and derives gen-level variables.

    Args:
        events (NanoEventsArray): events.
        genparts (GenParticleArray): event gen particles.
        fatjets (FatJetArray): event fat jets (should be only one fat jet per event!).
        genlabels (List[str]): gen variables to return.
        label (str): dataset label, formatted as
          ``{AK15 or AK8}_{H or QCD}_{(optional) H decay products}``.
        match_dR (float): max distance between fat jet and gen particle for matching.
          Defaults to 1.0.

    Returns:
        np.array: Boolean selection array of shape ``[len(fatjets)]``.
        Dict[str, np.array]: dict of gen variables.

    """

    jet_dR = 1.5 if "AK15" in label else 0.8
    matched_mask = np.ones(len(genparts), dtype="bool")

    if "H_" in label:
        matched_mask, GenVars = tagger_gen_H_matching(
            genparts, fatjets, genlabels, jet_dR, match_dR, decays=label.split("_")[-1]
        )
    elif "QCD" in label:
        matched_mask, GenVars = tagger_gen_QCD_matching(
            genparts, fatjets, genlabels, jet_dR, match_dR
        )

    GenVars = {**GenVars, **get_genjet_vars(events, fatjets, label.startswith("AK15"))}

    # if ``GenVars`` doesn't contain a gen var, that var is not applicable to this sample so fill with 0s
    GenVars = {
        key: GenVars[key] if key in GenVars.keys() else np.zeros(len(genparts)) for key in genlabels
    }
    return matched_mask, GenVars
