"""
Collection of variables useful for the nonresonant analysis.

Author: Raghav Kansal
"""

from __future__ import annotations

from collections import OrderedDict

years = ["2016APV", "2016", "2017", "2018"]

LUMI = {  # in pb^-1
    "2016": 16830.0,
    "2016APV": 19500.0,
    "2017": 41480.0,
    "2018": 59830.0,
}

# order is important for loading BDT preds
# label: selector
# when loading samples, assigns all sample names which start with `selector` to the `label` dataframe
# if `selector` starts with `*`, assigns all sample names which have `selector` in it
samples = OrderedDict(
    [
        ("QCD", "QCD"),
        ("TT", "TT"),
        ("ST", "ST"),
        ("W+Jets", "WJets"),
        ("Z+Jets", "ZJets"),
        ("Diboson", ("WW", "WZ", "ZZ")),
        # breaking single Higgs down into production modes for combination
        # https://gitlab.cern.ch/hh/naming-conventions#single-h-backgrounds
        ("ggFHbb", "GluGluHToBB"),
        ("VBFHbb", "VBFHToBB"),
        ("ZHbb", "ZH_HToBB"),
        ("WHbb", ("WminusH_HToBB", "WplusH_HToBB")),
        ("ggZHbb", "ggZH_HToBB"),
        ("ttHbb", "ttHToBB"),
        ("HWW", ("*HToWW", "*HToNonbb")),
        # ("HH", ("GluGluToHHTo4B_node_cHHH1_preUL")),
        ("Data", "JetHT"),
    ]
)


data_key = "Data"
qcd_key = "QCD"
bg_keys = [key for key in samples if key != data_key]
hbb_bg_keys = [key for key in samples if "Hbb" in key]

# change HHbbVV to hbbhVV
nonres_samples = OrderedDict(
    [
        ("HHbbVV", "GluGluToHHTobbVV_node_cHHH1"),
        ("ggHH_kl_2p45_kt_1_HHbbVV", "GluGluToHHTobbVV_node_cHHH2p45"),
        ("ggHH_kl_5_kt_1_HHbbVV", "GluGluToHHTobbVV_node_cHHH5"),
        ("ggHH_kl_0_kt_1_HHbbVV", "GluGluToHHTobbVV_node_cHHH0"),  # not used in combination
        ("VBFHHbbVV", "VBF_HHTobbVV_CV_1_C2V_1_C3_1"),
        ("qqHH_CV_1_C2V_0_kl_1_HHbbVV", "VBF_HHTobbVV_CV_1_C2V_0_C3_1"),
        ("qqHH_CV_1p5_C2V_1_kl_1_HHbbVV", "VBF_HHTobbVV_CV_1_5_C2V_1_C3_1"),
        ("qqHH_CV_1_C2V_1_kl_2_HHbbVV", "VBF_HHTobbVV_CV_1_C2V_1_C3_2"),
        ("qqHH_CV_1_C2V_2_kl_1_HHbbVV", "VBF_HHTobbVV_CV_1_C2V_2_C3_1"),
        ("qqHH_CV_1_C2V_1_kl_0_HHbbVV", "VBF_HHTobbVV_CV_1_C2V_1_C3_0"),
        (
            "qqHH_CV_0p5_C2V_1_kl_1_HHbbVV",
            "VBF_HHTobbVV_CV_0_5_C2V_1_C3_1",
        ),  # not used in combination
    ]
)
nonres_sig_keys = list(nonres_samples.keys())

res_samples = OrderedDict([])

res_mps = [
    (1000, 100),
    (1000, 125),
    (1000, 150),
    (1000, 190),
    (1000, 250),
    (1000, 60),
    (1000, 80),
    (1200, 100),
    (1200, 150),
    (1200, 190),
    (1200, 250),
    (1200, 60),
    (1200, 80),
    (1400, 100),
    (1400, 125),
    (1400, 150),
    (1400, 190),
    (1400, 250),
    (1400, 60),
    (1400, 80),
    (1600, 100),
    (1600, 125),
    (1600, 150),
    (1600, 190),
    (1600, 250),
    (1600, 60),
    (1600, 80),
    (1800, 100),
    (1800, 125),
    (1800, 150),
    (1800, 190),
    (1800, 250),
    (1800, 60),
    (1800, 80),
    (2000, 100),
    (2000, 125),
    (2000, 150),
    (2000, 190),
    (2000, 250),
    (2000, 60),
    (2000, 80),
    (2200, 100),
    (2200, 125),
    (2200, 150),
    (2200, 190),
    (2200, 250),
    (2200, 60),
    (2200, 80),
    (2400, 100),
    (2400, 125),
    (2400, 150),
    (2400, 190),
    (2400, 250),
    (2400, 60),
    (2400, 80),
    (2600, 100),
    (2600, 125),
    (2600, 150),
    (2600, 190),
    (2600, 250),
    (2600, 80),
    (2800, 100),
    (2800, 125),
    (2800, 150),
    (2800, 190),
    (2800, 250),
    (2800, 60),
    (2800, 80),
    (3000, 100),
    (3000, 125),
    (3000, 150),
    (3000, 190),
    (3000, 250),
    (3000, 60),
    (3000, 80),
    (3500, 100),
    (3500, 125),
    (3500, 150),
    (3500, 190),
    (3500, 250),
    (3500, 60),
    (3500, 80),
    (4000, 100),
    (4000, 125),
    (4000, 150),
    (4000, 190),
    (4000, 250),
    (4000, 60),
    (4000, 80),
    (600, 100),
    (600, 125),
    (600, 150),
    (600, 250),
    (600, 60),
    (600, 80),
    (700, 100),
    (700, 125),
    (700, 150),
    (700, 250),
    (700, 60),
    (700, 80),
    (800, 100),
    (800, 125),
    (800, 150),
    (800, 250),
    (800, 60),
    (800, 80),
    (900, 100),
    (900, 125),
    (900, 150),
    (900, 250),
    (900, 60),
    (900, 80),
]

for mX, mY in res_mps:
    res_samples[f"X[{mX}]->H(bb)Y[{mY}](VV)"] = f"NMSSM_XToYHTo2W2BTo4Q2B_MX-{mX}_MY-{mY}"

res_sig_keys = list(res_samples.keys())

all_mc = nonres_sig_keys + res_sig_keys + bg_keys
BDT_sample_order = nonres_sig_keys + bg_keys + [data_key]

# from https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2021/005
txbb_wps = {
    "2016APV": {"HP": 0.9883, "MP": 0.9737, "LP": 0.9088},
    "2016": {"HP": 0.9883, "MP": 0.9735, "LP": 0.9137},
    "2017": {"HP": 0.987, "MP": 0.9714, "LP": 0.9105},
    "2018": {"HP": 0.988, "MP": 0.9734, "LP": 0.9172},
}

norm_preserving_weights = ["genweight", "pileup", "ISRPartonShower", "FSRPartonShower"]

jecs = {
    "JES": "JES_jes",
    "JER": "JER",
}

jmsr = {
    "JMS": "JMS",
    "JMR": "JMR",
}

jec_shifts = []
for key in jecs:
    for shift in ["up", "down"]:
        jec_shifts.append(f"{key}_{shift}")

jmsr_shifts = []
for key in jmsr:
    for shift in ["up", "down"]:
        jmsr_shifts.append(f"{key}_{shift}")

# variables affected by JECs
jec_vars = [
    "bbFatJetPt",
    "VVFatJetPt",
    "DijetEta",
    "DijetPt",
    "DijetMass",
    "bbFatJetPtOverDijetPt",
    "VVFatJetPtOverDijetPt",
    "VVFatJetPtOverbbFatJetPt",
    "BDTScore",
    "VBFJetPt",
    "vbf_Mass_jj",
]


# variables affected by JMS/R
jmsr_vars = [
    "bbFatJetMsd",
    "bbFatJetParticleNetMass",
    "VVFatJetMsd",
    "VVFatJetParticleNetMass",
    "DijetEta",
    "DijetPt",
    "DijetMass",
    "bbFatJetPtOverDijetPt",
    "VVFatJetPtOverDijetPt",
    "VVFatJetPtOverbbFatJetPt",
    "BDTScore",
]

lp_sf_vars = [
    ("lp_sf_lnN", 101),
    ("lp_sf_pt_extrap_vars", 100),
    ("lp_sf_sys_down", 1),
    ("lp_sf_sys_up", 1),
    ("lp_sf_dist", 1),
    ("lp_sf_np_down", 1),
    ("lp_sf_np_up", 1),
    ("lp_sf_double_matched_event", 1),
    ("lp_sf_inside_boundary_quarks", 1),
    ("lp_sf_outside_boundary_quarks", 1),
    ("lp_sf_unmatched_quarks", 1),
    ("lp_sf_rc_unmatched_quarks", 1),
]
