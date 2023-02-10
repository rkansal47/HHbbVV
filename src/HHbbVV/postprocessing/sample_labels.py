# label: selector
# samples = {"HHbbVV": "GluGluToHHTobbVV", "QCD": "QCD", "ST": "ST", "TT": "TT", "Data": "JetHT"}
from collections import OrderedDict

# order is important for loading BDT preds
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
bg_keys = list(samples.keys() - [sig_key, data_key])

# bdt_sample_order = [sig_key, qcd_key, "TT", data_key]
