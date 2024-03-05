# label: selector
# samples = {"HHbbVV": "GluGluToHHTobbVV", "QCD": "QCD", "ST": "ST", "TT": "TT", "Data": "JetHT"}
from __future__ import annotations

samples = {"HHbbVV": "GluGluToHHTobbVV", "QCD": "QCD", "TT": "TT", "Data": "JetHT"}
sig_key = "HHbbVV"
data_key = "Data"
qcd_key = "QCD"
ttsl_key = "TTToSemiLeptonic"
bg_keys = [qcd_key, "TT"]

bdt_sample_order = [sig_key, qcd_key, "TT", data_key]
