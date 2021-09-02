"""
Takes the skimmed pickles (output of bbVVSkimmer) and makes control plots.

Author(s): Raghav Kansal
"""


import numpy as np
import awkward as ak
import vector

import matplotlib.pyplot as plt
import mplhep as hep

plt.rcParams.update({'font.size': 16})
plt.style.use(hep.style.CMS)

# load the data

import pickle

# backgrounds listed first and plotted in order
keys = ['V', 'Top', 'QCD', 'Data', 'HHbbVV4q']
labels = ['VV/V+jets', 'ST/TT', 'QCD', 'Data', 'HHbbVV4q']
num_bg = 3  # up to this label for bg
sig = 'HHbbVV4q'

events = {}
out_pickle = {}

for key in keys:
    print(key)
    with open(f'../../data/2017_combined/{key}.pkl', 'rb') as file:
        events[key] = pickle.load(file)['skimmed_events']
        # out_pickle[key] = pickle.load(file)

# out_pickle['HHbbVV4q']


list(events['HHbbVV4q'].keys())

np.sum(events['QCD']['weight'])
np.sum(events['Top']['weight'])
np.sum(events['V']['weight'])
np.sum(events['HHbbVV4q']['weight'])
np.sum(events['Data']['weight'])


# define bb jet, VV jet according to hybrid policy

dR = 0.8

for key in keys:
    if key != sig: continue
    print(key)
    jet1_bb_leading = events[key]['ak8FatJetParticleNetMD_Txbb'][:, 0:1] >= events[key]['ak8FatJetParticleNetMD_Txbb'][:, 1:2]
    bb_mask = np.concatenate([jet1_bb_leading, ~jet1_bb_leading], axis=1)

    jet1_VV_leading = events[key]['ak15FatJetParticleNet_Th4q'][:, 0:1] >= events[key]['ak15FatJetParticleNet_Th4q'][:, 1:2]
    VV_mask = ak.concatenate([jet1_VV_leading, ~jet1_VV_leading], axis=1)

    print("prelim masks")

    ak8FatJet = vector.array({
                        "pt": events[key]['ak8FatJetPt'],
                        "phi": events[key]['ak8FatJetPhi'],
                        "eta": events[key]['ak8FatJetEta'],
                        "M": events[key]['ak8FatJetMsd'],
    })

    ak15FatJet = vector.array({
                        "pt": events[key]['ak15FatJetPt'],
                        "phi": events[key]['ak15FatJetPhi'],
                        "eta": events[key]['ak15FatJetEta'],
                        "M": events[key]['ak15FatJetMsd'],
    })

    print("fat jet arrays)")

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
            events[key][newvar] = value[VV_mask]


# hybrid policy analysis plots

key = 'HHbbVV4q'

bb_cut = events[key]['ak8FatJetParticleNetMD_Txbb'][bb_mask] > 0.8
VV_cut = events[key]['ak15FatJetParticleNet_Th4q'][VV_mask] > 0.8

plotdir = '../plots/HybridPolicyAnalysis/'


_ = plt.hist(events[key]['ak8FatJetParticleNetMD_Txbb'].reshape(-1), weights=np.repeat(np.expand_dims(events[key]['weight'], 1), 2, 1).reshape(-1), bins=np.linspace(0, 1, 51), histtype='step', color='blue', linewidth=3, label='All')
_ = plt.hist(events[key]['ak8FatJetParticleNetMD_Txbb'][bb_mask], weights=events[key]['weight'], bins=np.linspace(0, 1, 51), histtype='step', color='red', linewidth=2, label='Leading by Txbb')
_ = plt.hist(events[key]['ak8FatJetParticleNetMD_Txbb'][bb_mask][VV_cand_overlap], weights=events[key]['weight'][VV_cand_overlap], bins=np.linspace(0, 1, 51), histtype='step', color='green', linewidth=2, label='Leading jet overlapping with AK15 VV candidate')
_ = plt.hist(events[key]['ak8FatJetParticleNetMD_Txbb'][bb_mask][VV_cut * VV_cand_overlap], weights=events[key]['weight'][VV_cut * VV_cand_overlap], bins=np.linspace(0, 1, 51), histtype='step', color='orange', linewidth=2, label='Leading jet overlapping & VV cand Th4q > 0.8')
# plt.ylim(0, 30000)
plt.xlabel('Txbb Score')
plt.ylabel('# Jets')
plt.title('AK8 Fat Jets')
plt.legend(prop={'size': 18})
plt.savefig(plotdir + 'dR15ak8bbjets.pdf', bbox_inches='tight')


_ = plt.hist(events[key]['ak15FatJetParticleNet_Th4q'].reshape(-1), weights=np.repeat(np.expand_dims(events[key]['weight'], 1), 2, 1).reshape(-1), bins=np.linspace(0, 1, 51), histtype='step', color='blue', linewidth=3, label='All')
_ = plt.hist(events[key]['ak15FatJetParticleNet_Th4q'][VV_mask], weights=events[key]['weight'], bins=np.linspace(0, 1, 51), histtype='step', color='red', linewidth=2, label='Leading by Th4q')
_ = plt.hist(events[key]['ak15FatJetParticleNet_Th4q'][VV_mask][VV_cand_overlap], weights=events[key]['weight'][VV_cand_overlap], bins=np.linspace(0, 1, 51), histtype='step', color='green', linewidth=2, label='Leading jet overlapping with AK8 bb candidate')
_ = plt.hist(events[key]['ak15FatJetParticleNet_Th4q'][VV_mask][bb_cut * VV_cand_overlap], weights=events[key]['weight'][bb_cut * VV_cand_overlap], bins=np.linspace(0, 1, 51), histtype='step', color='orange', linewidth=2, label='Leading jet overlapping & bb cand Txbb > 0.8')
# plt.ylim(0, 5000)
plt.xlabel('Th4q Score')
plt.ylabel('# Jets')
plt.title('AK15 Fat Jets')
plt.legend(prop={'size': 18})
plt.savefig(plotdir + 'dR15ak15VVjets.pdf', bbox_inches='tight')


bbFatJet = vector.array({
                    "pt": events[key]['bbFatJetPt'],
                    "phi": events[key]['bbFatJetPhi'],
                    "eta": events[key]['bbFatJetEta'],
                    "M": events[key]['bbFatJetMsd'],
})

VVFatJet = vector.array({
                    "pt": events[key]['VVFatJetPt'],
                    "phi": events[key]['VVFatJetPhi'],
                    "eta": events[key]['VVFatJetEta'],
                    "M": events[key]['VVFatJetMsd'],
})


plt.hist(bbFatJet.deltaR(VVFatJet), weights=events[key]['weight'], bins=np.linspace(0, 5, 51), histtype='step', label='All jets')
plt.hist(bbFatJet.deltaR(VVFatJet)[bb_cut * VV_cut], weights=events[key]['weight'][bb_cut * VV_cut], bins=np.linspace(0, 5, 51), histtype='step', label='Jets with tagger scores > 0.8')
plt.xlabel('$\Delta R$ between bb and VV fat jets')
plt.ylabel('# Jets')
plt.title('HHbbVV4q Hybrid Fat Jets')
plt.legend(prop={'size': 18})
plt.savefig(plotdir + 'bbVVdR.pdf', bbox_inches='tight')


# derived vars

key = 'HHbbVV4q'

for key in keys:
    print(key)
    bbFatJet = vector.array({
                        "pt": events[key]['bbFatJetPt'],
                        "phi": events[key]['bbFatJetPhi'],
                        "eta": events[key]['bbFatJetEta'],
                        "M": events[key]['bbFatJetMsd'],
    })

    VVFatJet = vector.array({
                        "pt": events[key]['VVFatJetPt'],
                        "phi": events[key]['VVFatJetPhi'],
                        "eta": events[key]['VVFatJetEta'],
                        "M": events[key]['VVFatJetMsd'],
    })

    Dijet = bbFatJet + VVFatJet

    events[key]['DijetPt'] = Dijet.pt
    events[key]['DijetMass'] = Dijet.M
    events[key]['DijetEta'] = Dijet.eta

    ak8FatJet = vector.array({
                        "pt": events[key]['ak8FatJetPt'],
                        "phi": events[key]['ak8FatJetPhi'],
                        "eta": events[key]['ak8FatJetEta'],
                        "M": events[key]['ak8FatJetMsd'],
    })

    ak8Dijet = ak8FatJet[:, 0] + ak8FatJet[:, 1]

    ak15FatJet = vector.array({
                        "pt": events[key]['ak15FatJetPt'],
                        "phi": events[key]['ak15FatJetPhi'],
                        "eta": events[key]['ak15FatJetEta'],
                        "M": events[key]['ak15FatJetMsd'],
    })

    ak15Dijet = ak15FatJet[:, 0] + ak15FatJet[:, 1]

    events[key]['ak8DijetPt'] = ak8Dijet.pt
    events[key]['ak8DijetMass'] = ak8Dijet.M
    events[key]['ak8DijetEta'] = ak8Dijet.eta

    events[key]['ak15DijetPt'] = ak15Dijet.pt
    events[key]['ak15DijetMass'] = ak15Dijet.M
    events[key]['ak15DijetEta'] = ak15Dijet.eta

    events[key]['bbFatJetPtOverDijetPt'] = events[key]['bbFatJetPt'] / events[key]['DijetPt']
    events[key]['VVFatJetPtOverDijetPt'] = events[key]['VVFatJetPt'] / events[key]['DijetPt']

    events[key]['VVFatJetPtOverbbFatJetPt'] = events[key]['VVFatJetPt'] / events[key]['bbFatJetPt']


# trigger efficiencies

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

np.sum(events['QCD']['finalWeight'])
np.sum(events['Top']['finalWeight'])
np.sum(events['V']['finalWeight'])
np.sum(events['HHbbVV4q']['finalWeight'])
np.sum(events['Data']['finalWeight'])



# plots

import os
plotdir = '../plots/ControlPlots/Sep1/'
os.system(f'mkdir -p {plotdir}')

hep.style.use("CMS")

colours = {
    'darkblue': '#1f78b4',
    'lightblue':'#a6cee3',
    'red': '#e31a1c',
    'orange': '#ff7f00'
}

bg_colours = [colours['lightblue'], colours['orange'], colours['darkblue']]
sig_colour = colours['red']

bg_scale = 1
sig_scale = 1.3e7 * bg_scale

from hist import Hist
from hist.intervals import ratio_uncertainty

hists = {}

hist_vars = {  # (bins, labels)
    # 'MET_pt': ([50, 0, 250], r"$p^{miss}_T$ (GeV)"),

    'DijetEta': ([50, -5, 5], r"$\eta^{jj}$"),
    'DijetPt': ([50, 0, 2000], r"$p_T^{jj}$ (GeV)"),
    'DijetMass': ([50, 0, 2000], r"$m^{jj}$ (GeV)"),

    # 'ak8DijetEta': ([50, -5, 5], r"$\eta^{jj}$"),
    # 'ak8DijetPt': ([50, 0, 2000], r"$p_T^{jj}$ (GeV)"),
    # 'ak8DijetMass': ([50, 0, 2000], r"$m^{jj}$ (GeV)"),
    #
    # 'ak15DijetEta': ([50, -5, 5], r"$\eta^{jj}$"),
    # 'ak15DijetPt': ([50, 0, 2000], r"$p_T^{jj}$ (GeV)"),
    # 'ak15DijetMass': ([50, 0, 2000], r"$m^{jj}$ (GeV)"),
    #
    # 'bbFatJetEta': ([50, -3, 3], r"$\eta^{bb}$"),
    # 'bbFatJetPt': ([50, 200, 1000], r"$p^{bb}_T$ (GeV)"),
    # 'bbFatJetMsd': ([50, 20, 250], r"$m^{bb}$ (GeV)"),
    # 'bbFatJetParticleNetMD_Txbb': ([50, 0, 1], r"$p^{bb}_{Txbb}$"),
    #
    # 'VVFatJetEta': ([50, -3, 3], r"$\eta^{VV}$"),
    # 'VVFatJetPt': ([50, 200, 1000], r"$p^{VV}_T$ (GeV)"),
    # 'VVFatJetMsd': ([50, 20, 400], r"$m^{VV}$ (GeV)"),
    # 'VVFatJetParticleNet_Th4q': ([50, 0, 1], r"$p^{VV}_{Th4q}$"),
    #
    # 'bbFatJetPtOverDijetPt': ([50, 0.3, 0.7], r"$p^{bb}_T / p_T^{jj}$"),
    # 'VVFatJetPtOverDijetPt': ([50, 0.3, 0.7], r"$p^{VV}_T / p_T^{jj}$"),
    # 'VVFatJetPtOverbbFatJetPt': ([50, 0.4, 2.5], r"$p^{VV}_T / p^{bb}_T$"),
}

for var, (bins, label) in hist_vars.items():
    hists[var] = (
        Hist.new
        .StrCat(keys, name='Sample')
        .Reg(*bins, name=var, label=label)
        .Double()
    )

    for key in keys:
        hists[var].fill(Sample=key, **{var: events[key][var]}, weight=events[key]['finalWeight'])


# var = 'MET_pt'

for var in hist_vars.keys():
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw=dict(height_ratios=[3, 1], hspace=0), sharex=True)

    ax.set_ylabel('Events')
    hep.histplot([hists[var][key, :] * bg_scale for key in keys[:num_bg]], ax=ax, histtype='fill', stack=True, label=[f"{label}" for label in labels[:num_bg]], color=bg_colours[:num_bg])
    hep.histplot(hists[var][sig, :] * sig_scale, ax=ax, histtype='step', label=f"{sig} $\\times$ {sig_scale:.1e}", color=sig_colour)
    hep.histplot(hists[var]['Data', :], ax=ax, histtype='errorbar', label="Data", color='black')
    ax.legend()
    ax.set_ylim(0)

    bg_tot = sum([hists[var][key, :] for key in keys[:num_bg]])
    yerr = ratio_uncertainty(hists[var]['Data', :].values(), bg_tot.values(), 'poisson')
    hep.histplot(hists[var]['Data', :] / bg_tot, yerr=yerr, ax=rax, histtype='errorbar', color='black', capsize=4)
    rax.set_ylabel('Data/MC')
    rax.grid()

    hep.cms.label('Preliminary', data=True, lumi=40, year=2017, ax=ax)
    plt.savefig(f"{plotdir}qcdscaleonly_{var}.pdf", bbox_inches='tight')
