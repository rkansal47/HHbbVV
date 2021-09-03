"""
Analyzes and makes plots of the hybrid policy.

Author(s): Raghav Kansal
"""

import numpy as np
import awkward as ak

import plotting
import utils

# import importlib
# importlib.reload(utils)

# load the data

import pickle

sig = "HHbbVV4q"
sig_pickle_path = f'../../data/2017_combined/{sig}.pkl'
plotdir = '../plots/HybridPolicyAnalysis/'

with open(sig_pickle_path, 'rb') as file:
    out_pickle = pickle.load(file)
    events = out_pickle['skimmed_events']

frac_not_bbVV_events = (out_pickle['cutflow']['all'] - out_pickle['cutflow']['has_bbVV']) / out_pickle['cutflow']['all']
frac_not_4q_events = (out_pickle['cutflow']['has_bbVV'] - out_pickle['cutflow']['has_4q']) / out_pickle['cutflow']['has_bbVV']

print(f"{frac_not_bbVV_events = }")
print(f"{frac_not_4q_events = }")

print(f"events: {np.sum(events['weight']):.2f}")


##################################################################################
# Define bb jet, VV jet according to hybrid policy
##################################################################################

dR = 1

jet1_bb_leading = events['ak8FatJetParticleNetMD_Txbb'][:, 0:1] >= events['ak8FatJetParticleNetMD_Txbb'][:, 1:2]
bb_mask = np.concatenate([jet1_bb_leading, ~jet1_bb_leading], axis=1)

jet1_VV_leading = events['ak15FatJetParticleNet_Th4q'][:, 0:1] >= events['ak15FatJetParticleNet_Th4q'][:, 1:2]
VV_mask = ak.concatenate([jet1_VV_leading, ~jet1_VV_leading], axis=1)

print("prelim masks")

ak8FatJet = utils.make_vector(events, 'ak8FatJet')
ak15FatJet = utils.make_vector(events, 'ak15FatJet')

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

vars = events.keys()
values = events.values()

for var, value in list(zip(vars, values)):
    if var.startswith('ak8FatJet'):
        newvar = 'bb' + var.split('ak8')[1]
        events[newvar] = value[bb_mask]
    elif var.startswith('ak15FatJet'):
        newvar = 'VV' + var.split('ak15')[1]
        events[newvar] = value[final_VV_mask]


##################################################################################
# Hybrid gen matching analysis
##################################################################################

# arbitrary tagger cut
TAGGER_CUT = 0.8
bb_cut = events['ak8FatJetParticleNetMD_Txbb'][bb_mask] > TAGGER_CUT
VV_cut = events['ak15FatJetParticleNet_Th4q'][VV_mask] > TAGGER_CUT
bbVV_cut = bb_cut * VV_cut

# get 4-vectors
vec_keys = ['ak8FatJet', 'ak15FatJet', 'GenHiggs', 'Genbb', 'GenVV', 'Gen4q']
vectors = {vec_key: utils.make_vector(events, vec_key) for vec_key in vec_keys}

bbLeadingFatJet = utils.make_vector(events, 'ak8FatJet', bb_mask)
VVLeadingFatJet = utils.make_vector(events, 'ak15FatJet', VV_mask)
VVCandFatJet = utils.make_vector(events, 'ak15FatJet', final_VV_mask)

# get gen H->VV and H->bb higgs
is_HVV = utils.getParticles(events['GenHiggsChildren'], 'V')
is_Hbb = utils.getParticles(events['GenHiggsChildren'], 'b')

genHVV = vectors['GenHiggs'][is_HVV]
genHbb = vectors['GenHiggs'][is_Hbb]

# matching with gen higgs was correct if delta R between the chosen fat jet and gen higgs is < 1
bb_cand_correct = genHbb.deltaR(bbLeadingFatJet) < dR
VV_cand_correct = genHVV.deltaR(VVCandFatJet) < dR
VV_leading_correct = genHVV.deltaR(VVLeadingFatJet) < dR

tot_events = len(events['weight'])
print(f"fraction bb candidate correct: {np.sum(bb_cand_correct) / tot_events}")
print(f"fraction VV leading correct: {np.sum(VV_leading_correct) / tot_events}")
print(f"fraction VV candidate correct: {np.sum(VV_cand_correct) / tot_events}")

tot_bbVV_cut_events = np.sum(bbVV_cut)
print(f"out of events with both tagger scores > 0.8, fraction bb candidate correct: {np.sum(bb_cand_correct[bbVV_cut]) / tot_bbVV_cut_events}")
print(f"out of events with both tagger scores > 0.8, fraction VV candidate correct: {np.sum(VV_cand_correct[bbVV_cut]) / tot_bbVV_cut_events}")
print(f"out of events with both tagger scores > 0.8, fraction of both candidates correct: {np.sum(bb_cand_correct[bbVV_cut] * VV_cand_correct[bbVV_cut]) / tot_bbVV_cut_events}")



dRbins = np.linspace(0, 5, 101)

# gen higgs dR
plotting.singleHistPlot(genHVV.deltaR(genHbb), events['weight'], dRbins,
                xlabel='$\Delta R$ between gen HVV and gen Hbb',
                title='Gen HH $\Delta R$', plotdir=plotdir, name='genHiggsdR')

# leading VV to gen HVV matching
plotting.multiHistCutsPlot(genHVV.deltaR(VVLeadingFatJet), weights=events['weight'], bins=dRbins,
                cuts=[None, VV_cand_overlap, VV_cut * VV_cand_overlap, bbVV_cut * VV_cand_overlap],
                labels=['All events',
                        'Leading jet overlapping with AK8 bb candidate',
                        'Overlap and VV jet tagger score > 0.8',
                        'Overlap and both jets tagger scores > 0.8'],
                xlabel='$\Delta R$ between gen HVV and leading ak15 VV fat jet',
                title='AK15 VV FatJet Gen Matching', plotdir=plotdir, name='VVgenmatching', ylim=1.5)

# leading bb to gen Hbb matching
plotting.multiHistCutsPlot(genHbb.deltaR(bbLeadingFatJet), weights=events['weight'], bins=dRbins,
                cuts=[None, VV_cand_overlap, bb_cut * VV_cand_overlap, bbVV_cut * VV_cand_overlap],
                labels=['All events',
                        'Leading jet overlapping with AK15 VV candidate',
                        'Overlap and bb jet tagger score > 0.8',
                        'Overlap and both jets tagger scores > 0.8'],
                xlabel='$\Delta R$ between gen Hbb and leading ak8 bb fat jet',
                title='AK8 bb FatJet Gen Matching', plotdir=plotdir, name='bbgenmatching', ylim=1.5)

# candidate VV to gen HVV matching
plotting.multiHistCutsPlot(genHVV.deltaR(VVCandFatJet), weights=events['weight'], bins=dRbins,
                cuts=[None, VV_cand_overlap, VV_cut * VV_cand_overlap, bbVV_cut * VV_cand_overlap],
                labels=['All events',
                        'Leading jet overlapping with AK8 bb candidate',
                        'Overlap and VV jet tagger score > 0.8',
                        'Overlap and both jets tagger scores > 0.8'],
                xlabel='$\Delta R$ between gen HVV and candidate ak15 VV fat jet',
                title='AK15 VV FatJet Candidate Gen Matching', plotdir=plotdir, name='VVcandgenmatching', ylim=1.5)

# bb dR with leading and candidate VV
plotting.multiHistPlot([bbLeadingFatJet.deltaR(VVLeadingFatJet), bbLeadingFatJet.deltaR(VVCandFatJet)], weights=events['weight'], bins=dRbins,
                labels=['VV Leading FatJet',
                        'VV Candidate FatJet'],
                xlabel='$\Delta R$ between bb and VV fat jets',
                title='bb VV $\Delta R$', plotdir=plotdir, name='bbVVdR')


##################################################################################
# Tagger score analysis
##################################################################################

tagger_bins = np.linspace(0, 1, 51)

# ak8 Txbb tagger
plotting.multiHistCutsPlot(events['ak8FatJetParticleNetMD_Txbb'], weights=np.repeat(events['weight'][:, np.newaxis], 2, 1), bins=tagger_bins,
                cuts=[None, bb_mask, bb_mask * VV_cand_overlap[:, np.newaxis], bb_mask * (VV_cut * VV_cand_overlap)[:, np.newaxis]],
                labels=['All events', 'Leading by Txbb', 'Leading jet overlapping with AK15 VV candidate', 'Leading jet overlapping & VV cand Th4q > 0.8'],
                xlabel='Txbb Score', ylabel='# Jets',
                title='AK8 Fat Jets', plotdir=plotdir, name='ak8bbjets', ylim=5)

# ak15 Th4q tagger
plotting.multiHistCutsPlot(events['ak8FatJetParticleNet_Th4q'], weights=np.repeat(events['weight'][:, np.newaxis], 2, 1), bins=tagger_bins,
                cuts=[None, VV_mask, VV_mask * VV_cand_overlap[:, np.newaxis], VV_mask * (bb_cut * VV_cand_overlap)[:, np.newaxis]],
                labels=['All events', 'Leading by Th4q', 'Leading jet overlapping with AK8 bb candidate', 'Leading jet overlapping & bb cand Txbb > 0.8'],
                xlabel='Txbb Score', ylabel='# Jets',
                title='AK15 Fat Jets', plotdir=plotdir, name='ak15VVjets', ylim=5)
