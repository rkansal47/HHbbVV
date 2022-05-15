# label: selector
samples = {"HHbbVV": "GluGluToHHTobbVV", "QCD": "QCD", "ST": "ST", "TT": "TT", "Data": "JetHT"}
sig_key = "HHbbVV"
data_key = "Data"
qcd_key = "QCD"
bg_keys = [qcd_key, "ST", "TT"]

bdt_sample_order = [sig_key, qcd_key, "ST", "TT", data_key]
