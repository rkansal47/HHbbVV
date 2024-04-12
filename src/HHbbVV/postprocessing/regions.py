"""
Defines all the analysis regions.
****Important****: Region names used in the analysis cannot have underscores because of a rhalphalib convention.
Author(s): Raghav Kansal
"""

from __future__ import annotations

from dataclasses import dataclass

from utils import CUT_MAX_VAL

from HHbbVV.hh_vars import txbb_wps


@dataclass
class Region:
    cuts: dict = None
    label: str = None
    signal: bool = False  # is this a signal region?
    lpsf: bool = False  # is this a region for LP SF calculation?
    lpsf_region: str = None  # if so, name of region for which LP SF is calculated
    cutstr: str = None  # optional label for the region's cuts e.g. when scanning cuts


def get_nonres_selection_regions(
    year: str,
    region: str = "all",
    ggf_txbb_wp: str = "MP",
    ggf_bdt_wp: float = 0.998,
    vbf_txbb_wp: str = "HP",
    vbf_bdt_wp: float = 0.999,
    lepton_veto_wp="None",
):
    """
    Args:
        year (str): year of data taking
        region (str): "ggf", "ggf_no_vbf", "vbf", or "all". "ggf_no_vbf" means without the VBF veto.
        ggf_txbb_wp (str): "LP", "HP", "MP"
        ggf_bdt_wp (float): ggF BDT WP
        vbf_txbb_wp (str): "LP", "HP", "MP"
        vbf_bdt_wp (float): VBF BDT WP
        lepton_veto_wp (str): "None", "Hbb", "HH"
    """
    pt_cuts = [300, CUT_MAX_VAL]

    fail_txbb_wp = "MP" if ggf_txbb_wp == "MP" or vbf_txbb_wp == "MP" else "HP"
    ggf_txbb_cut = txbb_wps[year][ggf_txbb_wp]
    vbf_txbb_cut = txbb_wps[year][vbf_txbb_wp]
    fail_txbb_cut = txbb_wps[year][fail_txbb_wp]

    if lepton_veto_wp == "None":
        lepton_cuts = {}
    elif lepton_veto_wp == "Hbb":
        lepton_cuts = {
            "nGoodElectronsHbb": [0, 0.9],
            "nGoodMuonsHbb": [0, 0.9],
        }
    elif lepton_veto_wp == "HH":
        lepton_cuts = {
            "nGoodElectronsHH": [0, 0.9],
            "nGoodMuonsHH": [0, 0.9],
        }
    else:
        raise ValueError(f"Invalid lepton veto: {lepton_veto_wp}")

    regions = {
        # {label: {cutvar: [min, max], ...}, ...}
        "passvbf": Region(
            cuts={
                "bbFatJetPt": pt_cuts,
                "VVFatJetPt": pt_cuts,
                "BDTScoreVBF": [vbf_bdt_wp, CUT_MAX_VAL],
                "bbFatJetParticleNetMD_Txbb": [vbf_txbb_cut, CUT_MAX_VAL],
                **lepton_cuts,
            },
            signal=True,
            label="VBF",
        ),
        "passggf": Region(
            cuts={
                "bbFatJetPt": pt_cuts,
                "VVFatJetPt": pt_cuts,
                "BDTScore": [ggf_bdt_wp, CUT_MAX_VAL],
                "bbFatJetParticleNetMD_Txbb": [ggf_txbb_cut, CUT_MAX_VAL],
                # veto VBF BDT or TXbb cuts
                "BDTScoreVBF+bbFatJetParticleNetMD_Txbb": [
                    [-CUT_MAX_VAL, vbf_bdt_wp],
                    [-CUT_MAX_VAL, vbf_txbb_cut],
                ],
                **lepton_cuts,
            },
            signal=True,
            label="ggF",
        ),
        "fail": Region(
            cuts={
                "bbFatJetPt": pt_cuts,
                "VVFatJetPt": pt_cuts,
                "bbFatJetParticleNetMD_Txbb": [0.8, fail_txbb_cut],
                **lepton_cuts,
            },
            label="Fail",
        ),
        # cuts for which LP SF is calculated
        "lpsf_passvbf": Region(
            cuts={
                "BDTScoreVBF": [vbf_bdt_wp, CUT_MAX_VAL],
            },
            lpsf=True,
            lpsf_region="passvbf",
            label="LP SF VBF Cut",
        ),
        "lpsf_passggf": Region(
            cuts={
                "BDTScore": [ggf_bdt_wp, CUT_MAX_VAL],
                # veto VBF BDT or TXbb cuts
                "BDTScoreVBF+bbFatJetParticleNetMD_Txbb": [
                    [-CUT_MAX_VAL, vbf_bdt_wp],
                    [-CUT_MAX_VAL, vbf_txbb_cut],
                ],
            },
            lpsf=True,
            lpsf_region="passggf",
            label="LP SF ggF Cut",
        ),
    }

    if region == "ggf":
        regions.pop("passvbf")
        regions.pop("lpsf_passvbf")
    elif region == "vbf":
        regions.pop("passggf")
        regions.pop("lpsf_passggf")
    elif region == "ggf_no_vbf":
        # old version without any VBF category
        lpregion = regions["lpsf_passggf"]
        lpregion.lpsf_region = "pass"
        regions = {
            "pass": regions["passggf"],
            "fail": regions["fail"],
            "lpsf": lpregion,
        }
        regions["pass"].cuts.pop("BDTScoreVBF")
        regions["lpsf"].cuts.pop("BDTScoreVBF")
    elif region != "all":
        raise ValueError(f"Invalid region: {region}")

    return regions


def get_nonres_vbf_selection_regions(
    year: str,
    txbb_wp: str = "HP",
    thww_wp: float = 0.6,
):
    # edit
    pt_cuts = [300, CUT_MAX_VAL]
    txbb_cut = txbb_wps[year][txbb_wp]

    return {
        # {label: {cutvar: [min, max], ...}, ...}
        "pass": Region(
            cuts={
                "bbFatJetPt": pt_cuts,
                "VVFatJetPt": pt_cuts,
                "bbFatJetParticleNetMD_Txbb": [txbb_cut, CUT_MAX_VAL],
                "VVFatJetParTMD_THWWvsT": [thww_wp, CUT_MAX_VAL],
                "vbf_Mass_jj": [500, 10000],
                "vbf_dEta_jj": [4, 10000],
                "ak8FatJetEta0": [-2.4, 2.4],
                "ak8FatJetEta1": [-2.4, 2.4],
                "DijetdEta": [0, 2.0],
                "DijetdPhi": [2.6, 10000],
                "bbFatJetParticleNetMass": [50, 250],
                "nGoodElectronsHbb": [0, 0.9],
                "nGoodMuonsHbb": [0, 0.9],
            },
            signal=True,
            label="Pass",
        ),
        "fail": Region(
            cuts={
                "bbFatJetPt": pt_cuts,
                "VVFatJetPt": pt_cuts,
                "bbFatJetParticleNetMD_Txbb": [0.8, txbb_cut],
                "VVFatJetParTMD_THWWvsT": [thww_wp, CUT_MAX_VAL],
                "vbf_Mass_jj": [500, 10000],
                "vbf_dEta_jj": [4, 10000],
                "ak8FatJetEta0": [-2.4, 2.4],
                "ak8FatJetEta1": [-2.4, 2.4],
                "DijetdEta": [0, 2.0],
                "DijetdPhi": [2.6, 10000],
                "bbFatJetParticleNetMass": [50, 250],
                "nGoodElectronsHbb": [0, 0.9],
                "nGoodMuonsHbb": [0, 0.9],
            },
            label="Fail",
        ),
        "lpsf_pass": Region(
            cuts={  # cut for which LP SF is calculated
                "bbFatJetPt": pt_cuts,
                "VVFatJetPt": pt_cuts,
                "VVFatJetParTMD_THWWvsT": [thww_wp, CUT_MAX_VAL],
            },
            lpsf=True,
            lpsf_region="pass",
            label="LP SF Cut",
        ),
    }


def get_res_selection_regions(
    year: str,
    mass_window: list[float] = None,
    txbb_wp: str = "HP",
    thww_wp: float = 0.6,
    leadingpt_wp: float = 400,
    subleadingpt_wp: float = 300,
):
    if mass_window is None:
        mass_window = [110, 145]
    mwsize = mass_window[1] - mass_window[0]
    mw_sidebands = [
        [mass_window[0] - mwsize / 2, mass_window[0]],
        [mass_window[1], mass_window[1] + mwsize / 2],
    ]
    txbb_cut = txbb_wps[year][txbb_wp]

    # first define without pT cuts
    regions = {
        # "unblinded" regions:
        "pass": Region(
            cuts={
                "bbFatJetParticleNetMass": mass_window,
                "bbFatJetParticleNetMD_Txbb": [txbb_cut, CUT_MAX_VAL],
                "VVFatJetParTMD_THWWvsT": [thww_wp, CUT_MAX_VAL],
            },
            signal=True,
            label="Pass",
        ),
        "fail": Region(
            cuts={
                "bbFatJetParticleNetMass": mass_window,
                "bbFatJetParticleNetMD_Txbb": [0.8, txbb_cut],
                "VVFatJetParTMD_THWWvsT": [-CUT_MAX_VAL, thww_wp],
            },
            label="Fail",
        ),
        # "blinded" validation regions:
        "passBlinded": Region(
            cuts={
                "bbFatJetParticleNetMass": mw_sidebands,
                "bbFatJetParticleNetMD_Txbb": [txbb_cut, CUT_MAX_VAL],
                "VVFatJetParTMD_THWWvsT": [thww_wp, CUT_MAX_VAL],
            },
            label="Validation Pass",
        ),
        "failBlinded": Region(
            cuts={
                "bbFatJetParticleNetMass": mw_sidebands,
                "bbFatJetParticleNetMD_Txbb": [0.8, txbb_cut],
                "VVFatJetParTMD_THWWvsT": [-CUT_MAX_VAL, thww_wp],
            },
            label="Validation Fail",
        ),
        # cut for which LP SF is calculated
        "lpsf_pass": Region(
            cuts={"VVFatJetParTMD_THWWvsT": [thww_wp, CUT_MAX_VAL]},
            lpsf=True,
            lpsf_region="pass",
            label="LP SF Cut",
        ),
    }

    # add pT cuts
    leading_pt_cut = [leadingpt_wp, CUT_MAX_VAL]
    subleading_pt_cut = [subleadingpt_wp, CUT_MAX_VAL]

    for _key, region in regions.items():
        cuts = {
            "bbFatJetPt": subleading_pt_cut,
            "VVFatJetPt": subleading_pt_cut,
            # '+' means OR
            "bbFatJetPt+VVFatJetPt": leading_pt_cut,
            **region.cuts,
        }
        region.cuts = cuts

    return regions
