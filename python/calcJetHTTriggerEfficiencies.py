import awkward as ak
import coffea
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, BaseSchema, NanoAODSchema
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep
from glob import glob
import pickle

plt.rcParams.update({'font.size': 16})
plt.style.use(hep.style.CMS)


class JetHTTriggerEfficienciesProcessor(processor.ProcessorABC):
    """ Accumulates two 2D (pT, msd) histograms from all input events: 1) before triggers, and 2) after triggers """

    def __init__(self):
        super(JetHTTriggerEfficienciesProcessor, self).__init__()
        self.triggers = {
            2017:   [
                        'HLT_PFJet500',
                        'HLT_AK8PFJet400',
                        'HLT_AK8PFJet500',
                        'HLT_AK8PFJet360_TrimMass30',
                        'HLT_AK8PFJet380_TrimMass30',
                        'HLT_AK8PFJet400_TrimMass30',
                        'HLT_AK8PFHT750_TrimMass50',
                        'HLT_AK8PFHT800_TrimMass50',
                        'HLT_PFHT1050',
                        # 'HLT_AK8PFJet330_PFAK8BTagCSV_p17'
                    ]
        }

        # self.ptbins = [250, 300, 350, 400, 550, 1000]
        # self.msdbins = [0, 20, 40, 60, 100, 250, 500]

        # 4b bins
#        self.ptbins = [i for i in range(250, 401, 25)] + [450, 500, 600, 700]
#        self.msdbins = [i for i in range(0, 241, 20)]

    def process(self, events):
        """ Returns pre- (den) and post- (num) trigger 2D (pT, msd) histograms from input NanoAOD events """

        jet_pts = ak.pad_none(events['FatJetAK15_pt'], 2, axis=1)
        jet_msds = ak.pad_none(events['FatJetAK15_msoftdrop'], 2, axis=1)

        fatjet1pt = jet_pts[:, 0]
        fatjet1msd = jet_msds[:, 0]

        fatjet1bool = ~ak.to_numpy(fatjet1pt).mask * ~ak.to_numpy(fatjet1msd).mask  # events with at least one fat jet

        # denominator
        den = (
            Hist.new
            .Reg(50, 0, 1000, name='pt', label="$p_T (GeV)$")
            .Reg(15, 0, 300, name='msd', label="MassSD (GeV)")
            .Double()
        ).fill(pt=ak.to_numpy(fatjet1pt[fatjet1bool]), msd=ak.to_numpy(fatjet1msd[fatjet1bool]))

        # numerator
        triggered = events[self.triggers[2017][0]]
        for i in range(1, len(self.triggers[2017])): triggered = triggered + events[self.triggers[2017][i]]

        fatjet1pt_triggered = jet_pts[:, 0][triggered * fatjet1bool]
        fatjet1msd_triggered = jet_msds[:, 0][triggered * fatjet1bool]

        num = (
            Hist.new
            .Reg(50, 0, 1000, name='pt', label="$p_T (GeV)$")
            .Reg(15, 0, 300, name='msd', label="MassSD (GeV)")
            .Double()
        ).fill(pt=ak.to_numpy(fatjet1pt_triggered), msd=ak.to_numpy(fatjet1msd_triggered))

        return {
            'den': den,
            'num': num
        }

    def postprocess(self, accumulator):
        return accumulator

import time
from distributed import Client
from lpcjobqueue import LPCCondorCluster

tic = time.time()
cluster = LPCCondorCluster()
# minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
cluster.adapt(minimum=1, maximum=3000000)
client = Client(cluster)

exe_args = {
    "client": client,
    "savemetrics": True,
    "schema": BaseSchema,
    "align_clusters": True,
}

with open('filelist.txt', 'r') as file:
    filelist = [f[:-1] for f in file.readlines()]

fileset = {'2017': filelist}

print("Waiting for at least one worker...")
client.wait_for_workers(1)
out, metrics = processor.run_uproot_job(
    fileset,
    treename="Events",
    processor_instance=JetHTTriggerEfficienciesProcessor(),
    executor=processor.dask_executor,
    executor_args=exe_args,
    maxchunks=10
)

elapsed = time.time() - tic

print(f"num: {out['num'].view(flow=True)}")
print(f"den: {out['den'].view(flow=True)}")

print(f"Metrics: {metrics}")
print(f"Finished in {elapsed:.1f}s")

# eff = num / den

effs = out['den'].copy()
effs[:, :] = out['num'].view(flow=True) / out['den'].view(flow=True)
effs.view()


# save effs (currently according to Henry the best way to save a hist object is via pickling)

filehandler = open('../data/AK15JetHTTriggerEfficiency_2017.hist', 'wb')
pickle.dump(effs, filehandler)
filehandler.close()


# plot

w, ptbins, msdbins = effs.to_numpy()

fig, ax = plt.subplots(figsize=(14, 14))
mesh = ax.pcolormesh(msdbins, ptbins, w, cmap="jet")
for i in range(len(ptbins) - 1):
    for j in range(len(msdbins) - 1):
        ax.text((msdbins[j] + msdbins[j + 1]) / 2, (ptbins[i] + ptbins[i + 1]) / 2, w[i, j].round(2), color="black", ha="center", va="center", fontsize=12)
ax.set_xlabel('MassSD (GeV)')
ax.set_ylabel('$p_T$ (GeV)')
fig.colorbar(mesh)
plt.savefig("AK15TriggerEfficiencies.pdf")
# plt.show()
