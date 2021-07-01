import awkward as ak
from coffea import processor
from hist import Hist
import matplotlib.pyplot as plt
import mplhep as hep

plt.rcParams.update({'font.size': 16})
plt.style.use(hep.style.CMS)


class JetHTTriggerEfficienciesProcessor(processor.ProcessorABC):
    """ Accumulates two 2D (pT, msd) histograms from all input events: 1) before triggers, and 2) after triggers """

    def __init__(self):
        super(JetHTTriggerEfficienciesProcessor, self).__init__()
        self.muon_triggers = {
            2017:   [
                        'HLT_IsoMu27',
                        'HLT_Mu50'
                    ]
        }

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

    def process(self, events):
        """ Returns pre- (den) and post- (num) trigger 2D (pT, msd) histograms from input NanoAOD events """

        jet_pts = ak.pad_none(events['FatJetAK15_pt'], 2, axis=1)
        jet_msds = ak.pad_none(events['FatJetAK15_msoftdrop'], 2, axis=1)

        fatjet1pt = jet_pts[:, 0]
        fatjet1msd = jet_msds[:, 0]

        fatjet1bool = ~ak.to_numpy(fatjet1pt).mask * ~ak.to_numpy(fatjet1msd).mask  # events with at least one fat jet

        # passing single-muon control region triggers
        muon_triggered = events[self.muon_triggers[2017][0]]
        for i in range(1, len(self.muon_triggers[2017])): muon_triggered = muon_triggered + events[self.muon_triggers[2017][i]]

        # denominator
        den = (
            Hist.new
            .Reg(50, 0, 1000, name='pt', label="$p_T (GeV)$")
            .Reg(15, 0, 300, name='msd', label="MassSD (GeV)")
            .Double()
        ).fill(pt=ak.to_numpy(fatjet1pt[fatjet1bool * muon_triggered]), msd=ak.to_numpy(fatjet1msd[fatjet1bool * muon_triggered]))

        # numerator
        triggered = events[self.triggers[2017][0]]
        for i in range(1, len(self.triggers[2017])): triggered = triggered + events[self.triggers[2017][i]]

        fatjet1pt_triggered = jet_pts[:, 0][triggered * fatjet1bool * muon_triggered]
        fatjet1msd_triggered = jet_msds[:, 0][triggered * fatjet1bool * muon_triggered]

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
