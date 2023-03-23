"""
Collection of variables useful for the nonresonant analysis.

Author: Raghav Kansal
"""


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
samples = OrderedDict(
    [
        ("HHbbVV", "GluGluToHHTobbVV"),
        ("QCD", "QCD"),
        ("TT", "TT"),
        ("ST", "ST"),
        ("V+Jets", ("WJets", "ZJets")),
        ("Diboson", ("WW", "WZ", "ZZ")),
        ("Data", "JetHT"),
    ]
)

res_samples = OrderedDict([])

# res_mps = [
#     (600, 100),
#     (1000, 100),
#     # (1000, 450),
#     (2000, 100),
#     # (2000, 450),
#     (2000, 1000),
#     (3000, 100),
#     # (3000, 450),
#     # (3000, 1000),
# ]

res_mps = [
    (1000, 125),
    (1400, 125),
    (1400, 150),
    (1800, 125),
    (1800, 150),
    (1800, 190),
    (2200, 125),
    (2200, 150),
    (2200, 190),
    (2200, 250),
    (3000, 125),
    (3000, 150),
    (3000, 190),
    (3000, 250),
    (3000, 350),
]

for mX, mY in res_mps:
    res_samples[
        f"X[{mX}]->H(bb)Y[{mY}](VV)"
    ] = f"NMSSM_XToYH_MX{mX}_MY{mY}_HTo2bYTo2W_hadronicDecay"

nonres_sig_keys = ["HHbbVV"]
res_sig_keys = list(res_samples.keys())
data_key = "Data"
qcd_key = "QCD"
bg_keys = [key for key in samples.keys() if key not in nonres_sig_keys + res_sig_keys + [data_key]]


# from https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2021/005
txbb_wps = {
    "2016APV": {"HP": 0.9883, "MP": 0.9737, "LP": 0.9088},
    "2016": {"HP": 0.9883, "MP": 0.9735, "LP": 0.9137},
    "2017": {"HP": 0.987, "MP": 0.9714, "LP": 0.9105},
    "2018": {"HP": 0.988, "MP": 0.9734, "LP": 0.9172},
}


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
