import awkward as ak
from coffea import processor
from hist import Hist
import numpy as np


class JetHTTriggerEfficienciesProcessor(processor.ProcessorABC):
    """ Accumulates two 2D (pT, msd) histograms from all input events: 1) before triggers, and 2) after triggers """

    def __init__(self, ak15=True):
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

        self.ak15 = ak15

    def process(self, events):
        """ Returns pre- (den) and post- (num) trigger 2D (pT, msd) histograms from input NanoAOD events """

        # passing single-muon triggers
        muon_triggered = np.any(np.array([events.HLT[trigger] for trigger in self.muon_HLTs[2017]]), axis=0)

        fatjets = events.FatJetAK15 if self.ak15 else events.FatJet

        # does event have a fat jet
        fatjet1bool = ak.any(fatjets.pt, axis=1)

        # for denominator select events which pass the muon triggers and contain at least one fat jet
        den_selection = fatjet1bool * muon_triggered

        # denominator
        den = (
            Hist.new
            .Reg(50, 0, 1000, name='pt', label="$p_T (GeV)$")
            .Reg(15, 0, 300, name='msd', label="MassSD (GeV)")
            .Double()
        ).fill( jet1pt=fatjets.pt[den_selection][:, 0].to_numpy(),
                jet1msd=fatjets.msoftdrop[den_selection][:, 0].to_numpy()
            )

        # numerator

        # passing all HLT triggers
        bbVV_triggered = np.any(np.array([events.HLT[trigger] for trigger in self.HLTs[2017]]), axis=0)

        num_selection = fatjet1bool * muon_triggered * bbVV_triggered

        num = (
            Hist.new
            .Reg(50, 0, 1000, name='pt', label="$p_T (GeV)$")
            .Reg(15, 0, 300, name='msd', label="MassSD (GeV)")
            .Double()
        ).fill( jet1pt=fatjets.pt[num_selection][:, 0].to_numpy(),
                jet1msd=fatjets.msoftdrop[num_selection][:, 0].to_numpy(),
              )

        return {
            'den': den,
            'num': num
        }

    def postprocess(self, accumulator):
        return accumulator



class JetHT3DTriggerEfficienciesProcessor(processor.ProcessorABC):
    """ Accumulates two 2D (pT, msd) histograms from all input events: 1) before triggers, and 2) after triggers """

    def __init__(self, ak15=True):
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

        self.ak15 = ak15


    def process(self, events):
        """ Returns pre- (den) and post- (num) trigger 3D (jet 2 pT, jet 1 pT, jet 1 msd) histograms from input NanoAOD events """

        # passing single-muon triggers
        muon_triggered = np.any(np.array([events.HLT[trigger] for trigger in self.muon_HLTs[2017]]), axis=0)

        fatjets = events.FatJetAK15 if self.ak15 else events.FatJet

        # does event have a fat jet
        fatjet1bool = ak.any(fatjets.pt, axis=1)

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
        ).fill( jet1pt=fatjets.pt[den_selection][:, 0].to_numpy(),
                jet1msd=fatjets.msoftdrop[den_selection][:, 0].to_numpy(),
                jet2pt=ak.fill_none(ak.pad_none(fatjets.pt[den_selection][:, 1:2], 1, axis=1), 0).to_numpy()[:, 0],  # putting events with no fat jet 2 in the low pT bin
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
        ).fill( jet1pt=fatjets.pt[num_selection][:, 0].to_numpy(),
                jet1msd=fatjets.msoftdrop[num_selection][:, 0].to_numpy(),
                jet2pt=ak.fill_none(ak.pad_none(fatjets.pt[num_selection][:, 1:2], 1, axis=1), 0).to_numpy()[:, 0],  # putting events with no fat jet 2 in the low pT bin
              )

        return {
            'den': den,
            'num': num
        }

    def postprocess(self, accumulator):
        return accumulator
