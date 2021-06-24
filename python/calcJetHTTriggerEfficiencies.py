import uproot4
import awkward1 as ak
import numpy as np
import coffea.hist as hist
from copy import deepcopy
import matplotlib.pyplot as plt
import mplhep as hep

plt.rcParams.update({'font.size': 16})
plt.style.use(hep.style.CMS)

events = uproot4.concatenate("data/SingleMuon/Run2017*/21*/0000/nano_data2017_1.root:Events")

triggers17 = [
                'HLT_PFJet500',
                'HLT_AK8PFJet500',
                # 'HLT_AK8PFJet360_TrimMass30',
                # 'HLT_AK8PFJet380_TrimMass30',
                # 'HLT_AK8PFJet400_TrimMass30',
                # 'HLT_AK8PFHT800_TrimMass50',
                # 'HLT_AK8PFJet330_PFAK8BTagCSV_p17'
            ]

jet_pts = ak.pad_none(events['FatJetAK15_pt'], 2, axis=1)
jet_msds = ak.pad_none(events['FatJetAK15_msoftdrop'], 2, axis=1)

fatjet1pt = jet_pts[:, 0]
fatjet1msd = jet_msds[:, 0]

# denominator

fatjet1bool = ~ak.to_numpy(fatjet1pt).mask * ~ak.to_numpy(fatjet1msd).mask

ptbins = [250, 300, 350, 400, 550, 1000]
msdbins = [20, 40, 60, 100, 250, 500]

den = hist.Hist("Events",
                hist.Cat("sample", "Sample"),
                hist.Bin("pt", r"$p_T$ (GeV)", ptbins),
                hist.Bin("msd", r"massSD (GeV)", msdbins),
                )

den.fill(sample='data',
        pt=ak.to_numpy(fatjet1pt[fatjet1bool]),
        msd=ak.to_numpy(fatjet1msd[fatjet1bool]),
)

den.values()

# numerator

triggered = events[triggers17[0]]
for i in range(1, len(triggers17)): triggered = triggered + events[triggers17[i]]

fatjet1pt_triggered = jet_pts[:, 0][triggered]
fatjet1msd_triggered = jet_msds[:, 0][triggered]

num = hist.Hist("Events",
                hist.Cat("sample", "Sample"),
                hist.Bin("pt", r"$p_T$ (GeV)", ptbins),
                hist.Bin("msd", r"massSD (GeV)", msdbins),
                )

num.fill(sample='data',
        pt=ak.to_numpy(fatjet1pt_triggered),
        msd=ak.to_numpy(fatjet1msd_triggered),
)

num.values()

# efficiencies

effs = deepcopy(den)
effs.clear()

effs._sumw = {(): list(num._sumw.values())[0] / list(den._sumw.values())[0]}
effs.label = 'Efficiency'

effs.values()

fig, ax = plt.subplots(1, 1, figsize=(14, 14))
hist.plot2d(effs.sum('sample'), xaxis='msd', ax=ax)
for i in range(len(ptbins) - 1):
    for j in range(len(msdbins) - 1):
        ax.text((msdbins[j] + msdbins[j + 1]) / 2, (ptbins[i] + ptbins[i + 1]) / 2, effs.values()[()][i, j].round(2), color="black", ha="center", va="center", fontsize=12)

plt.savefig("AK15TriggerEfficiencies.pdf")
plt.show()
