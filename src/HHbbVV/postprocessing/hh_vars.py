"""
Collection of variables useful for the nonresonant analysis.

Author: Raghav Kansal
"""


from collections import OrderedDict


years = ["2016APV", "2016", "2017", "2018"]


# order is important for loading BDT preds
# label: selector
samples = OrderedDict(
    [
        ("HHbbVV", "GluGluToHHTobbVV"),
        ("QCD", "QCD"),
        ("TT", "TT"),
        ("W+Jets", "WJets"),
        ("Data", "JetHT"),
    ]
)

sig_key = "HHbbVV"
data_key = "Data"
qcd_key = "QCD"
bg_keys = [key for key in samples.keys() if key not in [sig_key, data_key]]


# from https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2021/005
txbb_wps = {
    "2016APV": {"HP": 0.9883, "MP": 0.9737, "LP": 0.9088},
    "2016": {"HP": 0.9883, "MP": 0.9735, "LP": 0.9137},
    "2017": {"HP": 0.987, "MP": 0.9714, "LP": 0.9105},
    "2018": {"HP": 0.988, "MP": 0.9734, "LP": 0.9172},
}