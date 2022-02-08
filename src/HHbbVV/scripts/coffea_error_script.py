import awkward as ak
from coffea import nanoevents

nanoevents.NanoAODSchema.nested_index_items["FatJetAK15_pFCandsIdxG"] = (
    "FatJetAK15_nConstituents",
    "JetPFCandsAK15",
)
nanoevents.NanoAODSchema.mixins["FatJetAK15"] = "FatJet"
nanoevents.NanoAODSchema.mixins["PFCands"] = "PFCand"

events = nanoevents.NanoEventsFactory.from_root(
    "../../../../data/2017_UL_nano/GluGluToHHTobbVV_node_cHHH1/nano_mc2017_1-1.root",
    schemaclass=nanoevents.NanoAODSchema,
).events()

jet_idx = 0

jet = ak.pad_none(events.FatJetAK15, 2, axis=1)[:, jet_idx]
jet_pfcands = events.PFCands[
    events.FatJetAK15PFCands.pFCandsIdx[events.FatJetAK15PFCands.jetIdx == jet_idx]
]

# will give error
jet_pfcands.delta_phi(jet)
