"""
Gen selection functions for skimmer.

Author(s): Raghav Kansal, Cristina Mantilla Suarez, Melissa Quinnan
"""

import numpy as np
import awkward as ak

from coffea.analysis_tools import PackedSelection
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.nanoevents.methods.nanoaod import FatJetArray, GenParticleArray

from typing import List, Dict, Tuple, Union

from .utils import pad_val, add_selection, PAD_VAL

d_PDGID = 1
u_PDGID = 2
s_PDGID = 3
c_PDGID = 4
b_PDGID = 5
g_PDGID = 21
TOP_PDGID = 6

ELE_PDGID = 11
vELE_PDGID = 12
MU_PDGID = 13
vMU_PDGID = 14
TAU_PDGID = 15
vTAU_PDGID = 16

G_PDGID = 22
Z_PDGID = 23
W_PDGID = 24
HIGGS_PDGID = 25
Y_PDGID = 35

b_PDGIDS = [511, 521, 523]

GRAV_PDGID = 39

GEN_FLAGS = ["fromHardProcess", "isLastCopy"]


def gen_selection_HYbbVV(
    events: NanoEventsArray,
    fatjets: FatJetArray,
    selection: PackedSelection,
    cutflow: dict,
    signGenWeights: ak.Array,
    skim_vars: dict,
):
    """Gets HY, bb, VV, and 4q 4-vectors"""

    # gen higgs and kids
    higgs = events.GenPart[
        (abs(events.GenPart.pdgId) == HIGGS_PDGID) * events.GenPart.hasFlags(GEN_FLAGS)
    ]
    GenHiggsVars = {f"GenHiggs{key}": higgs[var].to_numpy() for (var, key) in skim_vars.items()}
    is_bb = abs(higgs.children.pdgId) == b_PDGID
    has_bb = ak.sum(ak.flatten(is_bb, axis=2), axis=1) == 2

    bb = ak.flatten(higgs.children[is_bb], axis=2)
    GenbbVars = {f"Genbb{key}": pad_val(bb[var], 2, axis=1) for (var, key) in skim_vars.items()}

    # gen Y and kids
    Ys = events.GenPart[(abs(events.GenPart.pdgId) == Y_PDGID) * events.GenPart.hasFlags(GEN_FLAGS)]
    GenYVars = {f"GenY{key}": Ys[var].to_numpy() for (var, key) in skim_vars.items()}
    is_VV = (abs(Ys.children.pdgId) == W_PDGID) + (abs(Ys.children.pdgId) == Z_PDGID)
    has_VV = ak.sum(ak.flatten(is_VV, axis=2), axis=1) == 2

    add_selection("has_bbVV", has_bb * has_VV, selection, cutflow, False, signGenWeights)

    VV = ak.flatten(Ys.children[is_VV], axis=2)
    GenVVVars = {f"GenVV{key}": VV[var][:, :2].to_numpy() for (var, key) in skim_vars.items()}

    VV_children = VV.children

    # iterate through the children in photon scattering events to get final daughter quarks
    for i in range(5):
        photon_mask = ak.any(ak.flatten(abs(VV_children.pdgId), axis=2) == G_PDGID, axis=1)
        if not np.any(photon_mask):
            break

        # use a where condition to get next layer of children for photon scattering events
        VV_children = ak.where(photon_mask, ak.flatten(VV_children.children, axis=3), VV_children)

    quarks = abs(VV_children.pdgId) <= b_PDGID
    all_q = ak.all(ak.all(quarks, axis=2), axis=1)
    add_selection("all_q", all_q, selection, cutflow, False, signGenWeights)

    V_has_2q = ak.count(VV_children.pdgId, axis=2) == 2
    has_4q = ak.values_astype(ak.prod(V_has_2q, axis=1), bool)
    add_selection("has_4q", has_4q, selection, cutflow, False, signGenWeights)

    Gen4qVars = {
        f"Gen4q{key}": ak.to_numpy(
            ak.fill_none(
                ak.pad_none(
                    ak.pad_none(VV_children[var], 2, axis=1, clip=True), 2, axis=2, clip=True
                ),
                PAD_VAL,
            )
        )
        for (var, key) in skim_vars.items()
    }

    # fatjet gen matching
    Hbb = ak.pad_none(higgs, 1, axis=1, clip=True)[:, 0]
    HVV = ak.pad_none(Ys, 1, axis=1, clip=True)[:, 0]

    bbdr = fatjets[:, :2].delta_r(Hbb)
    vvdr = fatjets[:, :2].delta_r(HVV)

    match_dR = 0.8
    Hbb_match = bbdr <= match_dR
    HVV_match = vvdr <= match_dR

    # overlap removal - in the case where fatjet is matched to both, match it only to the closest Higgs
    Hbb_match = (Hbb_match * ~HVV_match) + (bbdr <= vvdr) * (Hbb_match * HVV_match)
    HVV_match = (HVV_match * ~Hbb_match) + (bbdr > vvdr) * (Hbb_match * HVV_match)

    VVJets = ak.pad_none(fatjets[HVV_match], 1, axis=1)[:, 0]
    quarkdrs = ak.flatten(VVJets.delta_r(VV_children), axis=2)
    num_prongs = ak.sum(quarkdrs < match_dR, axis=1)

    GenMatchingVars = {
        "ak8FatJetHbb": pad_val(Hbb_match, 2, axis=1),
        "ak8FatJetHVV": pad_val(HVV_match, 2, axis=1),
        "ak8FatJetHVVNumProngs": ak.fill_none(num_prongs, PAD_VAL).to_numpy(),
    }

    return {**GenHiggsVars, **GenYVars, **GenbbVars, **GenVVVars, **Gen4qVars, **GenMatchingVars}, (
        bb,
        ak.flatten(VV_children, axis=2),
    )


def gen_selection_HHbbVV(
    events: NanoEventsArray,
    fatjets: FatJetArray,
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
    is_bb = abs(higgs_children.pdgId) == b_PDGID
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
    GenbbVars = {f"Genbb{key}": pad_val(bb[var], 2, axis=1) for (var, key) in skim_vars.items()}

    # selecting only up to the 2nd index because of some 4V events
    # (doesn't matter which two are selected since these events will be excluded anyway)
    GenVVVars = {f"GenVV{key}": VV[var][:, :2].to_numpy() for (var, key) in skim_vars.items()}

    # checking that each V has 2 q children
    VV_children = VV.children

    # iterate through the children in photon scattering events to get final daughter quarks
    for i in range(5):
        photon_mask = ak.any(ak.flatten(abs(VV_children.pdgId), axis=2) == G_PDGID, axis=1)
        if not np.any(photon_mask):
            break

        # use a where condition to get next layer of children for photon scattering events
        VV_children = ak.where(photon_mask, ak.flatten(VV_children.children, axis=3), VV_children)

    quarks = abs(VV_children.pdgId) <= b_PDGID
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
                PAD_VAL,
            )
        )
        for (var, key) in skim_vars.items()
    }

    # fatjet gen matching
    Hbb = higgs[ak.sum(is_bb, axis=2) == 2]
    Hbb = ak.pad_none(Hbb, 1, axis=1, clip=True)[:, 0]

    HVV = higgs[ak.sum(is_VV, axis=2) == 2]
    HVV = ak.pad_none(HVV, 1, axis=1, clip=True)[:, 0]

    bbdr = fatjets[:, :2].delta_r(Hbb)
    vvdr = fatjets[:, :2].delta_r(HVV)

    match_dR = 0.8
    Hbb_match = bbdr <= match_dR
    HVV_match = vvdr <= match_dR

    # overlap removal - in the case where fatjet is matched to both, match it only to the closest Higgs
    Hbb_match = (Hbb_match * ~HVV_match) + (bbdr <= vvdr) * (Hbb_match * HVV_match)
    HVV_match = (HVV_match * ~Hbb_match) + (bbdr > vvdr) * (Hbb_match * HVV_match)

    VVJets = ak.pad_none(fatjets[HVV_match], 1, axis=1)[:, 0]
    quarkdrs = ak.flatten(VVJets.delta_r(VV_children), axis=2)
    num_prongs = ak.sum(quarkdrs < match_dR, axis=1)

    GenMatchingVars = {
        "ak8FatJetHbb": pad_val(Hbb_match, 2, axis=1),
        "ak8FatJetHVV": pad_val(HVV_match, 2, axis=1),
        "ak8FatJetHVVNumProngs": ak.fill_none(num_prongs, PAD_VAL).to_numpy(),
    }

    return {**GenHiggsVars, **GenbbVars, **GenVVVars, **Gen4qVars, **GenMatchingVars}, (
        bb,
        ak.flatten(VV_children, axis=2),
    )


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

    quarks = abs(VV_children.pdgId) <= b_PDGID
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
                PAD_VAL,
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

    quarks = abs(VV_children.pdgId) <= b_PDGID
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
    h_pdgid: int = HIGGS_PDGID,
) -> Tuple[np.array, Dict[str, np.array]]:
    """Gen matching for Higgs samples, arguments as defined in ``tagger_gen_matching``"""

    higgs = genparts[get_pid_mask(genparts, h_pdgid, byall=False) * genparts.hasFlags(GEN_FLAGS)]

    # find closest higgs
    matched_higgs = higgs[ak.argmin(fatjets.delta_r(higgs), axis=1, keepdims=True)]
    # select event only if distance to closest higgs is < ``match_dR``
    matched_higgs_mask = ak.any(fatjets.delta_r(matched_higgs) < match_dR, axis=1)
    # higgs kinematics
    genResVars = {
        f"fj_genRes_{key}": ak.fill_none(matched_higgs[var], PAD_VAL) for (var, key) in P4.items()
    }
    # Higgs parent kinematics
    bulkg = matched_higgs.distinctParent
    genXVars = {f"fj_genX_{key}": ak.fill_none(bulkg[var], PAD_VAL) for (var, key) in P4.items()}

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
        genVars["fj_dR_V_Vstar"] = v.delta_r(v_star)

        # select event only if VV are within jet radius
        matched_Vs_mask = ak.any(fatjets.delta_r(v) < jet_dR, axis=1) & ak.any(
            fatjets.delta_r(v_star) < jet_dR, axis=1
        )

        # get VV daughters
        daughters = ak.flatten(ak.flatten(matched_higgs_children.distinctChildren, axis=2), axis=2)
        daughters = daughters[daughters.hasFlags(GEN_FLAGS)]
        daughters_pdgId = abs(daughters.pdgId)

        # exclude neutrinos from nprongs count
        daughters_nov = daughters[
            (
                (daughters_pdgId != vELE_PDGID)
                & (daughters_pdgId != vMU_PDGID)
                & (daughters_pdgId != vTAU_PDGID)
            )
        ]
        # number of prongs inside the jet
        nprongs = ak.sum(fatjets.delta_r(daughters_nov) < jet_dR, axis=1)

        lepdaughters = daughters[
            (
                (daughters_pdgId == ELE_PDGID)
                | (daughters_pdgId == MU_PDGID)
                | (daughters_pdgId == TAU_PDGID)
            )
        ]
        lepinprongs = 0
        if len(lepdaughters) > 0:
            lepinprongs = ak.sum(fatjets.delta_r(lepdaughters) < jet_dR, axis=1)  # should be 0 or 1

        decay = (
            # 2 quarks * 1
            (ak.sum(daughters_pdgId <= b_PDGID, axis=1) == 2) * 1
            # 1 electron * 3
            + (ak.sum(daughters_pdgId == ELE_PDGID, axis=1) == 1) * 3
            # 1 muon * 5
            + (ak.sum(daughters_pdgId == MU_PDGID, axis=1) == 1) * 5
            # 1 tau * 7
            + (ak.sum(daughters_pdgId == TAU_PDGID, axis=1) == 1) * 7
            # 4 quarks * 11
            + (ak.sum(daughters_pdgId <= b_PDGID, axis=1) == 4) * 11
        )

        # print("daughters " ,daughters_pdgId)
        # print("decay ",decay)

        # get tau decays from daughters
        taudaughters = daughters[(daughters_pdgId == TAU_PDGID)].children
        taudaughters = taudaughters[taudaughters.hasFlags(["isLastCopy"])]
        taudaughters_pdgId = abs(taudaughters.pdgId)

        taudecay = (
            # pions/kaons (hadronic tau) * 1
            (
                ak.sum((taudaughters_pdgId == ELE_PDGID) | (taudaughters_pdgId == MU_PDGID), axis=2)
                == 0
            )
            * 1
            # 1 electron * 3
            + (ak.sum(taudaughters_pdgId == ELE_PDGID, axis=2) == 1) * 3
            # 1 muon * 5
            + (ak.sum(taudaughters_pdgId == MU_PDGID, axis=2) == 1) * 5
        )
        # flatten taudecay - so painful
        taudecay = ak.sum(taudecay, axis=-1)

        matched_mask = matched_higgs_mask & matched_Vs_mask

        genVVars = {f"fj_genV_{key}": ak.fill_none(v[var], PAD_VAL) for (var, key) in P4.items()}
        genVstarVars = {
            f"fj_genVstar_{key}": ak.fill_none(v_star[var], PAD_VAL) for (var, key) in P4.items()
        }

        # number of c quarks in V decay inside jet
        cquarks = daughters_nov[abs(daughters_nov.pdgId) == c_PDGID]
        ncquarks = ak.sum(fatjets.delta_r(cquarks) < jet_dR, axis=1)

        genLabelVars = {
            "fj_nprongs": nprongs,
            "fj_ncquarks": ncquarks,
            "fj_lepinprongs": lepinprongs,
            "fj_H_VV_4q": to_label(decay == 11),
            "fj_H_VV_elenuqq": to_label(decay == 4),
            "fj_H_VV_munuqq": to_label(decay == 6),
            "fj_H_VV_leptauelvqq": to_label((decay == 8) & (taudecay == 3)),
            "fj_H_VV_leptaumuvqq": to_label((decay == 8) & (taudecay == 5)),
            "fj_H_VV_hadtauvqq": to_label((decay == 8) & (taudecay == 1)),
            "fj_H_VV_unmatched": to_label(~matched_mask),
        }
        genVars = {**genVars, **genVVars, **genVstarVars, **genLabelVars}

    elif "qq" in decays:
        print("qq")
        children_mask = get_pid_mask(
            matched_higgs_children,
            [g_PDGID, b_PDGID, c_PDGID, s_PDGID, d_PDGID, u_PDGID],
            byall=False,
        )
        daughters = ak.firsts(matched_higgs_children[children_mask])
        daughters_pdgId = abs(daughters.pdgId)

        nprongs = ak.sum(fatjets.delta_r(daughters) < jet_dR, axis=1)

        # higgs decay
        decay = (
            # 2 b quarks * 1
            ((ak.sum(daughters_pdgId == b_PDGID, axis=1) == 2)) * 1
            # 2 c quarks * 3
            + ((ak.sum(daughters_pdgId == c_PDGID, axis=1) == 2)) * 3
            # 2 light quarks * 5
            + ((ak.sum(daughters_pdgId < c_PDGID, axis=1) == 2)) * 5
            # 2 gluons * 7
            + ((ak.sum(daughters_pdgId == g_PDGID, axis=1) == 2)) * 7
        )

        bs_decay = (ak.sum(daughters_pdgId == b_PDGID, axis=1) == 1) * (
            ak.sum(daughters_pdgId == s_PDGID, axis=1) == 1
        )

        print(bs_decay)
        print(np.sum(bs_decay))

        genLabelVars = {
            "fj_nprongs": nprongs,
            "fj_H_bb": to_label(decay == 1),
            "fj_H_cc": to_label(decay == 3),
            "fj_H_qq": to_label(decay == 5),
            "fj_H_gg": to_label(decay == 7),
            "fj_H_bs": to_label(bs_decay),
        }

        print(genLabelVars)

        genVars = {**genVars, **genLabelVars}

        # select event only if any of the q/g decays are within jet radius
        matched_qs_mask = ak.any(fatjets.delta_r(daughters) < jet_dR, axis=1)
        matched_mask = matched_higgs_mask & matched_qs_mask

        breakpoint()

    return matched_mask, genVars


def tagger_gen_QCD_matching(
    genparts: GenParticleArray,
    fatjets: FatJetArray,
    genlabels: List[str],
    jet_dR: float,
    match_dR: float = 1.0,
) -> Tuple[np.array, Dict[str, np.array]]:
    """Gen matching for QCD samples, arguments as defined in ``tagger_gen_matching``"""
    partons = genparts[
        get_pid_mask(genparts, [g_PDGID] + list(range(1, b_PDGID + 1)), ax=1, byall=False)
    ]
    matched_mask = ak.any(fatjets.delta_r(partons) < match_dR, axis=1)

    genLabelVars = {
        "fj_isQCDb": (fatjets.nBHadrons == 1),
        "fj_isQCDbb": (fatjets.nBHadrons > 1),
        "fj_isQCDc": (fatjets.nCHadrons == 1) * (fatjets.nBHadrons == 0),
        "fj_isQCDcc": (fatjets.nCHadrons > 1) * (fatjets.nBHadrons == 0),
        "fj_isQCDothers": (fatjets.nBHadrons == 0) & (fatjets.nCHadrons == 0),
    }

    genLabelVars = {key: to_label(var) for key, var in genLabelVars.items()}

    return matched_mask, genLabelVars


def tagger_gen_VJets_matching(
    genparts: GenParticleArray,
    fatjets: FatJetArray,
    genlabels: List[str],
    jet_dR: float,
    match_dR: float = 1.0,
) -> Tuple[np.array, Dict[str, np.array]]:
    """Gen matching for VJets samples"""

    vs = genparts[
        get_pid_mask(genparts, [W_PDGID, Z_PDGID], byall=False) * genparts.hasFlags(GEN_FLAGS)
    ]
    matched_vs = vs[ak.argmin(fatjets.delta_r(vs), axis=1, keepdims=True)]
    matched_vs_mask = ak.any(fatjets.delta_r(matched_vs) < match_dR, axis=1)
    genResVars = {
        f"fj_genRes_{key}": ak.fill_none(matched_vs[var], PAD_VAL) for (var, key) in P4.items()
    }

    daughters = ak.flatten(matched_vs.distinctChildren, axis=2)
    daughters = daughters[daughters.hasFlags(GEN_FLAGS)]
    daughters_pdgId = abs(daughters.pdgId)

    # exclude neutrinos from nprongs count
    daughters_nov = daughters[
        (
            (daughters_pdgId != vELE_PDGID)
            & (daughters_pdgId != vMU_PDGID)
            & (daughters_pdgId != vTAU_PDGID)
        )
    ]
    nprongs = ak.sum(fatjets.delta_r(daughters_nov) < jet_dR, axis=1)

    # number of c quarks in V decay inside jet
    cquarks = daughters_nov[abs(daughters_nov.pdgId) == c_PDGID]
    ncquarks = ak.sum(fatjets.delta_r(cquarks) < jet_dR, axis=1)

    lepdaughters = daughters[
        (
            (daughters_pdgId == ELE_PDGID)
            | (daughters_pdgId == MU_PDGID)
            | (daughters_pdgId == TAU_PDGID)
        )
    ]
    lepinprongs = 0
    if len(lepdaughters) > 0:
        lepinprongs = ak.sum(fatjets.delta_r(lepdaughters) < jet_dR, axis=1)  # should be 0 or 1

    decay = (
        # 2 quarks * 1
        (ak.sum(daughters_pdgId < b_PDGID, axis=1) == 2) * 1
        # 1 electron * 3
        + (ak.sum(daughters_pdgId == ELE_PDGID, axis=1) == 1) * 3
        # 1 muon * 5
        + (ak.sum(daughters_pdgId == MU_PDGID, axis=1) == 1) * 5
        # 1 tau * 7
        + (ak.sum(daughters_pdgId == TAU_PDGID, axis=1) == 1) * 7
    )
    matched_vdaus_mask = ak.any(fatjets.delta_r(daughters) < match_dR, axis=1)

    matched_mask = matched_vs_mask & matched_vdaus_mask

    genLabelVars = {
        "fj_nprongs": nprongs,
        "fj_lepinprongs": lepinprongs,
        "fj_ncquarks": ncquarks,
        "fj_V_2q": to_label(decay == 1),
        "fj_V_elenu": to_label(decay == 3),
        "fj_V_munu": to_label(decay == 5),
        "fj_V_taunu": to_label(decay == 7),
    }

    genVars = {**genResVars, **genLabelVars}

    return matched_mask, genVars


def tagger_gen_Top_matching(
    genparts: GenParticleArray,
    fatjets: FatJetArray,
    genlabels: List[str],
    jet_dR: float,
    match_dR: float = 1.0,
) -> Tuple[np.array, Dict[str, np.array]]:
    """Gen matching for TT samples"""

    tops = genparts[get_pid_mask(genparts, TOP_PDGID, byall=False) * genparts.hasFlags(GEN_FLAGS)]
    matched_tops = tops[ak.argmin(fatjets.delta_r(tops), axis=1, keepdims=True)]
    matched_tops_mask = ak.any(fatjets.delta_r(matched_tops) < match_dR, axis=1)
    genResVars = {
        f"fj_genRes_{key}": ak.fill_none(matched_tops[var], PAD_VAL) for (var, key) in P4.items()
    }

    daughters = ak.flatten(matched_tops.distinctChildren, axis=2)
    daughters = daughters[daughters.hasFlags(GEN_FLAGS)]
    daughters_pdgId = abs(daughters.pdgId)

    wboson_daughters = ak.flatten(daughters[(daughters_pdgId == 24)].distinctChildren, axis=2)
    wboson_daughters = wboson_daughters[wboson_daughters.hasFlags(GEN_FLAGS)]
    wboson_daughters_pdgId = abs(wboson_daughters.pdgId)
    decay = (
        # 2 quarks
        (ak.sum(wboson_daughters_pdgId < b_PDGID, axis=1) == 2) * 1
        # 1 electron * 3
        + (ak.sum(wboson_daughters_pdgId == ELE_PDGID, axis=1) == 1) * 3
        # 1 muon * 5
        + (ak.sum(wboson_daughters_pdgId == MU_PDGID, axis=1) == 1) * 5
        # 1 tau * 7
        + (ak.sum(wboson_daughters_pdgId == TAU_PDGID, axis=1) == 1) * 7
    )

    bquark = daughters[(daughters_pdgId == 5)]
    matched_b = ak.sum(fatjets.delta_r(bquark) < jet_dR, axis=1)

    # exclude neutrinos from nprongs count
    wboson_daughters_nov = wboson_daughters[
        (
            (wboson_daughters_pdgId != vELE_PDGID)
            & (wboson_daughters_pdgId != vMU_PDGID)
            & (wboson_daughters_pdgId != vTAU_PDGID)
        )
    ]
    # nprongs only includes the number of quarks from W decay (not b!)
    nprongs = ak.sum(fatjets.delta_r(wboson_daughters_nov) < jet_dR, axis=1)

    # number of c quarks in V decay inside jet
    cquarks = wboson_daughters_nov[abs(wboson_daughters_nov.pdgId) == c_PDGID]
    ncquarks = ak.sum(fatjets.delta_r(cquarks) < jet_dR, axis=1)

    lepdaughters = wboson_daughters[
        (
            (wboson_daughters_pdgId == ELE_PDGID)
            | (wboson_daughters_pdgId == MU_PDGID)
            | (wboson_daughters_pdgId == TAU_PDGID)
        )
    ]

    lepinprongs = 0
    if len(lepdaughters) > 0:
        lepinprongs = ak.sum(fatjets.delta_r(lepdaughters) < jet_dR, axis=1)  # should be 0 or 1

    # get tau decays from V daughters
    taudaughters = wboson_daughters[(wboson_daughters_pdgId == TAU_PDGID)].children
    taudaughters = taudaughters[taudaughters.hasFlags(["isLastCopy"])]
    taudaughters_pdgId = abs(taudaughters.pdgId)

    taudecay = (
        # pions/kaons (hadronic tau) * 1
        (ak.sum((taudaughters_pdgId == ELE_PDGID) | (taudaughters_pdgId == MU_PDGID), axis=2) == 0)
        * 1
        # 1 electron * 3
        + (ak.sum(taudaughters_pdgId == ELE_PDGID, axis=2) == 1) * 3
        # 1 muon * 5
        + (ak.sum(taudaughters_pdgId == MU_PDGID, axis=2) == 1) * 5
    )
    # flatten taudecay - so painful
    taudecay = ak.sum(taudecay, axis=-1)

    genLabelVars = {
        "fj_nprongs": nprongs,
        "fj_lepinprongs": lepinprongs,
        "fj_ncquarks": ncquarks,
        "fj_Top_bmerged": to_label(matched_b == 1),
        "fj_Top_2q": to_label(decay == 1),
        "fj_Top_elenu": to_label(decay == 3),
        "fj_Top_munu": to_label(decay == 5),
        "fj_Top_hadtauvqq": to_label((decay == 7) & (taudecay == 1)),
        "fj_Top_leptauelvnu": to_label((decay == 7) & (taudecay == 3)),
        "fj_Top_leptaumuvnu": to_label((decay == 7) & (taudecay == 5)),
    }

    matched_topdaus_mask = ak.any(fatjets.delta_r(daughters) < match_dR, axis=1)
    matched_mask = matched_tops_mask & matched_topdaus_mask

    genVars = {**genResVars, **genLabelVars}

    return matched_mask, genVars


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
        matched_sdgen_jet_mask = ak.flatten(
            ak.fill_none(fatjets.delta_r(matched_sdgen_jet) < match_dR, [False], axis=None),
            axis=None,
        )

        GenJetVars["fj_genjetmsd"] = matched_sdgen_jet.mass

        gen_dr = fatjets.delta_r(events.GenJetAK15)
        matched_gen_jet = events.GenJetAK15[ak.argmin(gen_dr, axis=1, keepdims=True)]
        matched_gen_jet_mask = ak.flatten(
            ak.fill_none(fatjets.delta_r(matched_gen_jet) < match_dR, [False], axis=None), axis=None
        )
        GenJetVars["fj_genjetmass"] = matched_gen_jet.mass

        matched_gen_jet_mask = matched_sdgen_jet_mask * matched_gen_jet_mask
    else:
        # NanoAOD automatically matched ak8 fat jets
        # No soft dropped gen jets however
        GenJetVars["fj_genjetmass"] = fatjets.matched_gen.mass
        matched_gen_jet_mask = np.ones(len(events), dtype="bool")

    return matched_gen_jet_mask, GenJetVars


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
    elif "Y_" in label:
        matched_mask, GenVars = tagger_gen_H_matching(
            genparts, fatjets, genlabels, jet_dR, match_dR, decays=label.split("_")[-1], h_pdgid=35
        )
    elif "QCD" in label:
        matched_mask, GenVars = tagger_gen_QCD_matching(
            genparts, fatjets, genlabels, jet_dR, match_dR
        )
    elif "VJets" in label:
        matched_mask, GenVars = tagger_gen_VJets_matching(
            genparts, fatjets, genlabels, jet_dR, match_dR
        )
    elif "Top" in label:
        matched_mask, GenVars = tagger_gen_Top_matching(
            genparts, fatjets, genlabels, jet_dR, match_dR
        )

    matched_gen_jet_mask, genjet_vars = get_genjet_vars(events, fatjets, label.startswith("AK15"))

    GenVars = {**GenVars, **genjet_vars}

    # if ``GenVars`` doesn't contain a gen var, that var is not applicable to this sample so fill with 0s
    GenVars = {
        key: GenVars[key] if key in GenVars.keys() else np.zeros(len(genparts)) for key in genlabels
    }
    for key, item in GenVars.items():
        try:
            GenVars[key] = GenVars[key].to_numpy()
        except:
            continue

    return matched_mask * matched_gen_jet_mask, GenVars


def ttbar_scale_factor_matching(
    events: NanoEventsArray, leading_fatjet: FatJetArray, selection_args: Tuple
):
    """
    Classifies jets as top-matched, w-matched, or un-matched using gen info, as defined in
    https://indico.cern.ch/event/1101433/contributions/4775247/

    Returns gen quarks as well for systematic uncertainties
    """
    # finding the two gen tops
    tops = events.GenPart[
        (abs(events.GenPart.pdgId) == TOP_PDGID) * events.GenPart.hasFlags(GEN_FLAGS)
    ]

    tops_children = tops.distinctChildren
    tops_children = tops_children[tops_children.hasFlags(GEN_FLAGS)]

    ws = ak.flatten(tops_children[np.abs(tops_children.pdgId) == W_PDGID], axis=2)

    # get hadronic W and top
    had_top_sel = np.all(np.abs(ws.children.pdgId) <= 5, axis=2)
    had_ws = ak.flatten(ws[had_top_sel])
    had_ws_children = had_ws.children
    had_tops = ak.flatten(tops[had_top_sel])

    # check for b's from top
    had_top_children = ak.flatten(tops_children[had_top_sel], axis=1)
    had_bs = had_top_children[np.abs(had_top_children.pdgId) == 5]
    add_selection("top_has_bs", np.any(had_bs.pdgId, axis=1), *selection_args)

    gen_quarks = ak.concatenate([had_bs[:, :1], had_ws_children[:, :2]], axis=1)

    deltaR = 0.8

    had_w_jet_match = ak.fill_none(
        ak.all(had_ws_children.delta_r(leading_fatjet) < deltaR, axis=1), False
    )
    had_b_jet_match = ak.flatten(
        pad_val(
            ak.fill_none(had_bs.delta_r(leading_fatjet) < deltaR, [], axis=0),
            1,
            False,
            axis=1,
            to_numpy=False,
        )
    )

    top_match_dict = {
        "top_matched": had_w_jet_match * had_b_jet_match,
        "w_matched": had_w_jet_match * ~had_b_jet_match,
        "unmatched": ~had_w_jet_match,
    }

    top_match_dict = {key: val.to_numpy().astype(int) for key, val in top_match_dict.items()}

    return top_match_dict, gen_quarks
