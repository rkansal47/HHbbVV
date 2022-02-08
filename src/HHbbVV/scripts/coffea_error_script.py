import awkward as ak
from coffea import nanoevents

nanoevents.NanoAODSchema.nested_index_items["FatJetAK15_pFCandsIdxG"] = (
    "FatJetAK15_nConstituents",
    "JetPFCandsAK15",
)
nanoevents.NanoAODSchema.mixins["FatJetAK15"] = "FatJet"
nanoevents.NanoAODSchema.mixins["PFCands"] = "PFCand"

events = nanoevents.NanoEventsFactory.from_root(
    "root://cmsxrootd.fnal.gov//store/user/lpchbb/cmantill/v2_2/2017v1/HWW/GluGluToHHTobbVV_node_cHHH1_TuneCP5_13TeV-powheg-pythia8/GluGluToHHTobbVV_node_cHHH1/220206_211217/0000/nano_mc_2017_ULv1_NANO_1-1.root",
    schemaclass=nanoevents.NanoAODSchema,
).events()

jet_idx = 0

jet = ak.pad_none(events.FatJetAK15, 2, axis=1)[:, jet_idx]
jet_pfcands = events.PFCands[
    events.FatJetAK15PFCands.pFCandsIdx[events.FatJetAK15PFCands.jetIdx == jet_idx]
]

# will give error
jet_pfcands.delta_phi(jet)
