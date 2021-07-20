import awkward as ak
from coffea import processor
from hist import Hist
import numpy as np


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



class JetHT3DTriggerEfficienciesProcessor(processor.ProcessorABC):
    """ Accumulates two 2D (pT, msd) histograms from all input events: 1) before triggers, and 2) after triggers """

    def __init__(self):
        super(JetHT3DTriggerEfficienciesProcessor, self).__init__()

        self.muon_HLTs = {
            2017:   [
                        'IsoMu27',
                        'Mu50'
                    ]
        }

        self.HLTs = {
            2017:   [
                        'PFJet500',
                        'AK8PFJet400',
                        'AK8PFJet500',
                        'AK8PFJet360_TrimMass30',
                        'AK8PFJet380_TrimMass30',
                        'AK8PFJet400_TrimMass30',
                        'AK8PFHT750_TrimMass50',
                        'AK8PFHT800_TrimMass50',
                        'PFHT1050',
                        # 'AK8PFJet330_PFAK8BTagCSV_p17'
                    ]
        }


    def process(self, events):
        """ Returns pre- (den) and post- (num) trigger 3D (jet 2 pT, jet 1 pT, jet 1 msd) histograms from input NanoAOD events """

        # passing single-muon triggers
        muon_triggered = np.any(np.array([events.HLT[trigger] for trigger in self.muon_HLTs[2017]]), axis=0)

        # does event have a fat jet
        fatjet1bool = ak.any(events.FatJetAK15.pt, axis=1)

        # for denominator select events which pass the muon triggers and contain at least one fat jet
        den_selection = fatjet1bool * muon_triggered

        # denominator
        jet2varbins = [0, 250, 300, 350, 400, 500, 750, 1000]

        den = (
            Hist.new
            .Var(jet2varbins, name='jet2pt', label="AK15 Fat Jet 2 $p_T$ (GeV)")
            .Reg(50, 0, 1000, name='jet1pt', label="AK15 Fat Jet 1 $p_T$ (GeV)")
            .Reg(15, 0, 300, name='jet1msd', label="AK15 Fat Jet 1 MassSD (GeV)")
            .Double()
        ).fill( jet1pt=events.FatJetAK15.pt[den_selection][:, 0].to_numpy(),
                jet1msd=events.FatJetAK15.msoftdrop[den_selection][:, 0].to_numpy(),
                jet2pt=ak.fill_none(ak.pad_none(events.FatJetAK15.pt[den_selection][:, 1:2], 1, axis=1), 0).to_numpy()[:, 0],  # putting events with no fat jet 2 in the low pT bin
              )

        # numerator

        # passing all HLT triggers
        bbVV_triggered = np.any(np.array([events.HLT[trigger] for trigger in self.HLTs[2017]]), axis=0)

        num_selection = fatjet1bool * muon_triggered * bbVV_triggered

        num = (
            Hist.new
            .Var(jet2varbins, name='jet2pt', label="AK15 Fat Jet 2 $p_T$ (GeV)")
            .Reg(50, 0, 1000, name='jet1pt', label="AK15 Fat Jet 1 $p_T$ (GeV)")
            .Reg(15, 0, 300, name='jet1msd', label="AK15 Fat Jet 1 MassSD (GeV)")
            .Double()
        ).fill( jet1pt=events.FatJetAK15.pt[num_selection][:, 0].to_numpy(),
                jet1msd=events.FatJetAK15.msoftdrop[num_selection][:, 0].to_numpy(),
                jet2pt=ak.fill_none(ak.pad_none(events.FatJetAK15.pt[num_selection][:, 1:2], 1, axis=1), 0).to_numpy()[:, 0],  # putting events with no fat jet 2 in the low pT bin
              )
              
        return {
            'den': den,
            'num': num
        }

    def postprocess(self, accumulator):
        return accumulator
