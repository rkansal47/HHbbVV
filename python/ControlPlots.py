"""
Takes the skimmed pickles (output of bbVVSkimmer) and makes control plots.

Author(s): Raghav Kansal
"""

import numpy as np
import awkward as ak

import utils
import plotting

# load the data

import pickle

# backgrounds listed first and plotted in order
keys = ['V', 'Top', 'QCD', 'Data', 'HHbbVV4q']
labels = ['VV/V+jets', 'ST/TT', 'QCD', 'Data', 'HHbbVV4q']
num_bg = 3  # up to this label for bg
sig = 'HHbbVV4q'
data_path = '../../data/2017_combined/'

import os
plotdir = '../plots/ControlPlots/Sep3/'
os.system(f'mkdir -p {plotdir}')


# import importlib
# importlib.reload(utils)
# importlib.reload(plotting)


events = {}

for key in keys:
    # if key != sig: continue
    print(key)
    with open(f'{data_path}{key}.pkl', 'rb') as file:
        events[key] = pickle.load(file)['skimmed_events']

for key in keys:
    print(f"{key} events: {np.sum(events[key]['weight']):.2f}")


##################################################################################
# Define bb jet, VV jet according to hybrid policy
##################################################################################

dR = 1

for key in keys:
    # if key != sig: continue
    print(key)
    jet1_bb_leading = events[key]['ak8FatJetParticleNetMD_Txbb'][:, 0:1] >= events[key]['ak8FatJetParticleNetMD_Txbb'][:, 1:2]
    bb_mask = np.concatenate([jet1_bb_leading, ~jet1_bb_leading], axis=1)

    jet1_VV_leading = events[key]['ak15FatJetParticleNet_Th4q'][:, 0:1] >= events[key]['ak15FatJetParticleNet_Th4q'][:, 1:2]
    VV_mask = ak.concatenate([jet1_VV_leading, ~jet1_VV_leading], axis=1)

    print("prelim masks")

    ak8FatJet = utils.make_vector(events[key], 'ak8FatJet')
    ak15FatJet = utils.make_vector(events[key], 'ak15FatJet')

    print("fat jet arrays")

    # check if ak15 VV candidate jet is overlapping with the ak8 bb one  - 37.6% of bbVV jets, 6.8% with bb, VV tagger scores > 0.8
    bb_cand_VV_cand_dist = ak8FatJet[bb_mask].deltaR(ak15FatJet[VV_mask])
    VV_cand_overlap = bb_cand_VV_cand_dist < dR

    # overlap policy is: if bb and VV candidate jets overlap, use the ak15 jet which is farthest from the bb jet as the VV candidate
    bb_cand_VV_not_cand_dist = ak8FatJet[bb_mask].deltaR(ak15FatJet[~VV_mask])
    VV_not_cand_farther =  bb_cand_VV_not_cand_dist > bb_cand_VV_cand_dist

    # flip VV_mask only if (VV candidate jet is overlapping AND non-candidate jet is farther away)
    final_VV_mask = VV_mask ^ (VV_cand_overlap * VV_not_cand_farther)

    print("final masks")

    vars = events[key].keys()
    values = events[key].values()

    for var, value in list(zip(vars, values)):
        if var.startswith('ak8FatJet'):
            newvar = 'bb' + var.split('ak8')[1]
            events[key][newvar] = value[bb_mask]
        elif var.startswith('ak15FatJet'):
            newvar = 'VV' + var.split('ak15')[1]
            events[key][newvar] = value[final_VV_mask]


##################################################################################
# Derived variables
##################################################################################

for key in keys:
    print(key)
    bbFatJet = utils.make_vector(events[key], 'bbFatJet')
    VVFatJet = utils.make_vector(events[key], 'VVFatJet')
    Dijet = bbFatJet + VVFatJet

    events[key]['DijetPt'] = Dijet.pt
    events[key]['DijetMass'] = Dijet.M
    events[key]['DijetEta'] = Dijet.eta

    events[key]['bbFatJetPtOverDijetPt'] = events[key]['bbFatJetPt'] / events[key]['DijetPt']
    events[key]['VVFatJetPtOverDijetPt'] = events[key]['VVFatJetPt'] / events[key]['DijetPt']
    events[key]['VVFatJetPtOverbbFatJetPt'] = events[key]['VVFatJetPt'] / events[key]['bbFatJetPt']

    # ak8FatJet = utils.make_vector(events[key], 'ak8FatJet')
    # ak15FatJet = utils.make_vector(events[key], 'ak15FatJet')
    # ak8Dijet = ak8FatJet[:, 0] + ak8FatJet[:, 1]
    # ak15Dijet = ak15FatJet[:, 0] + ak15FatJet[:, 1]
    #
    # events[key]['ak8DijetPt'] = ak8Dijet.pt
    # events[key]['ak8DijetMass'] = ak8Dijet.M
    # events[key]['ak8DijetEta'] = ak8Dijet.eta
    #
    # events[key]['ak15DijetPt'] = ak15Dijet.pt
    # events[key]['ak15DijetMass'] = ak15Dijet.M
    # events[key]['ak15DijetEta'] = ak15Dijet.eta


##################################################################################
# Apply trigger efficiencies
##################################################################################

from coffea.lookup_tools.dense_lookup import dense_lookup

with open('../corrections/trigEffs/AK15JetHTTriggerEfficiency_2017.hist', 'rb') as filehandler:
    ak15TrigEffs = pickle.load(filehandler)

ak15TrigEffsLookup = dense_lookup(np.nan_to_num(ak15TrigEffs.view(flow=False), 0), np.squeeze(ak15TrigEffs.axes.edges))

with open('../corrections/trigEffs/AK8JetHTTriggerEfficiency_2017.hist', 'rb') as filehandler:
    ak8TrigEffs = pickle.load(filehandler)

ak8TrigEffsLookup = dense_lookup(np.nan_to_num(ak8TrigEffs.view(flow=False), 0), np.squeeze(ak8TrigEffs.axes.edges))

for key in keys:
    if key == 'Data':
        events[key]['finalWeight'] = events[key]['weight']
    else:
        bb_fj_trigEffs = ak8TrigEffsLookup(events[key]['bbFatJetPt'], events[key]['bbFatJetMsd'])
        VV_fj_trigEffs = ak15TrigEffsLookup(events[key]['VVFatJetPt'], events[key]['VVFatJetMsd'])
        combined_trigEffs = 1 - (1 - bb_fj_trigEffs) * (1 - VV_fj_trigEffs)
        events[key]['finalWeight'] = events[key]['weight'] * combined_trigEffs

QCD_SCALE_FACTOR =  (np.sum(events['Data']['finalWeight']) - np.sum(events['Top']['finalWeight']) - np.sum(events['V']['finalWeight'])) / (np.sum(events['QCD']['finalWeight']))
events['QCD']['finalWeight'] *= QCD_SCALE_FACTOR

for key in keys:
    print(f"Final {key} events: {np.sum(events[key]['finalWeight']):.2f}")


##################################################################################
# Control plots
##################################################################################

sig_scale = np.sum(events['Data']['finalWeight']) / np.sum(events['HHbbVV4q']['finalWeight'])


hists = {}

hist_vars = {  # (bins, labels)
    'MET_pt': ([50, 0, 250], r"$p^{miss}_T$ (GeV)"),

    'DijetEta': ([50, -8, 8], r"$\eta^{jj}$"),
    'DijetPt': ([50, 0, 750], r"$p_T^{jj}$ (GeV)"),
    'DijetMass': ([50, 0, 2500], r"$m^{jj}$ (GeV)"),

    'bbFatJetEta': ([50, -3, 3], r"$\eta^{bb}$"),
    'bbFatJetPt': ([50, 200, 1000], r"$p^{bb}_T$ (GeV)"),
    'bbFatJetMsd': ([50, 20, 250], r"$m^{bb}$ (GeV)"),
    'bbFatJetParticleNetMD_Txbb': ([50, 0, 1], r"$p^{bb}_{Txbb}$"),

    'VVFatJetEta': ([50, -3, 3], r"$\eta^{VV}$"),
    'VVFatJetPt': ([50, 200, 1000], r"$p^{VV}_T$ (GeV)"),
    'VVFatJetMsd': ([50, 20, 500], r"$m^{VV}$ (GeV)"),
    'VVFatJetParticleNet_Th4q': ([50, 0, 1], r"$p^{VV}_{Th4q}$"),

    'bbFatJetPtOverDijetPt': ([50, 0, 40], r"$p^{bb}_T / p_T^{jj}$"),
    'VVFatJetPtOverDijetPt': ([50, 0, 40], r"$p^{VV}_T / p_T^{jj}$"),
    'VVFatJetPtOverbbFatJetPt': ([50, 0.4, 2.5], r"$p^{VV}_T / p^{bb}_T$"),

    # 'ak8DijetEta': ([50, -5, 5], r"$\eta^{jj}$"),
    # 'ak8DijetPt': ([50, 0, 2000], r"$p_T^{jj}$ (GeV)"),
    # 'ak8DijetMass': ([50, 0, 2000], r"$m^{jj}$ (GeV)"),
    #
    # 'ak15DijetEta': ([50, -5, 5], r"$\eta^{jj}$"),
    # 'ak15DijetPt': ([50, 0, 2000], r"$p_T^{jj}$ (GeV)"),
    # 'ak15DijetMass': ([50, 0, 2000], r"$m^{jj}$ (GeV)"),
}


for var, (bins, label) in hist_vars.items():
    hists[var] = utils.singleVarHist(events, var, bins, label, weight_key='finalWeight')

# var = 'MET_pt'

for var in hist_vars.keys():
    plotting.ratioHistPlot(hists[var], keys[:num_bg], sig, bg_labels=labels[:num_bg], plotdir=plotdir, name=var, sig_scale=sig_scale)
