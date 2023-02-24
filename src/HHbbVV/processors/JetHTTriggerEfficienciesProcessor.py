from collections import OrderedDict
import awkward as ak
from coffea import processor
from hist import Hist
import numpy as np

from coffea.analysis_tools import PackedSelection

from .utils import add_selection
from .common import HLTs


class JetHTTriggerEfficienciesProcessor(processor.ProcessorABC):
    """Accumulates two 2D (pT, msd) histograms from all input events: 1) before triggers, and 2) after triggers"""

    muon_HLTs = {
        "2016": ["IsoMu24", "IsoTkMu24", "Mu50"],
        "2017": ["IsoMu27", "Mu50"],
        "2018": ["IsoMu24", "Mu50"],
    }

    # same selection as in AN-2020-201

    muon_selection = {
        "Id": "tight",
        "pt": 30,
        "eta": 2.4,
        "pfIsoId": 4,  # tight PF isolation
        "count": 1,
    }

    ak8_jet_selection = {
        "eta": 2.4,
        "delta_r_muon": 1.5,
    }

    # # bins, min, max
    msd_bins = (15, 0, 300)

    # edges
    pt_bins = [250, 275, 300, 325, 350, 375, 400, 450, 500, 600, 800, 1000]
    tagger_bins = [0.0, 0.9, 0.95, 0.98, 1.0]

    def __init__(self, ak15=False):
        super(JetHTTriggerEfficienciesProcessor, self).__init__()

        self.ak15 = ak15

    def process(self, events):
        """Returns pre- (den) and post- (num) trigger 2D (pT, msd) histograms from input NanoAOD events"""

        year = events.metadata["dataset"][:4]

        selection = PackedSelection()

        cutflow = OrderedDict()
        cutflow["all"] = len(events)

        selection_args = (selection, cutflow, True)

        # objects
        num_jets = 1
        muon = events.Muon
        fatjets = events.FatJet

        # passing single-muon triggers
        muon_triggered = np.any(
            np.array([events.HLT[trigger] for trigger in self.muon_HLTs[year]]),
            axis=0,
        )

        # passing our triggers
        bbVV_triggered = np.any(
            np.array(
                [events.HLT[trigger] for trigger in HLTs[year] if trigger in events.HLT.fields]
            ),
            axis=0,
        )

        # muon
        muon_selector = (
            (muon[f"{self.muon_selection['Id']}Id"])
            * (muon.pt > self.muon_selection["pt"])
            * (np.abs(muon.eta) < self.muon_selection["eta"])
            * (muon.pfIsoId >= self.muon_selection["pfIsoId"])
        )

        muon_selector = muon_selector * (
            ak.count(events.Muon.pt[muon_selector], axis=1) == self.muon_selection["count"]
        )
        muon = ak.pad_none(muon[muon_selector], 1, axis=1)[:, 0]

        muon_selector = ak.any(muon_selector, axis=1)
        add_selection("muon", muon_selector, *selection_args)

        # ak8 jet selection
        fatjet_selector = (np.abs(fatjets.eta) < self.ak8_jet_selection["eta"]) * (
            np.abs(fatjets.delta_r(muon)) > self.ak8_jet_selection["delta_r_muon"]
        )

        fatjets = ak.pad_none(fatjets[fatjet_selector], num_jets, axis=1)[:, :num_jets]
        fatjet_idx = ak.argmax(fatjet_selector, axis=1)  # gets first index which is true
        fatjet_selector = ak.any(fatjet_selector, axis=1)

        add_selection("ak8_jet", fatjet_selector, *selection_args)

        fatjets.txbb = fatjets.particleNetMD_Xbb / (
            fatjets.particleNetMD_QCD + fatjets.particleNetMD_Xbb
        )

        # initialize histograms
        h = (
            Hist.new.Var(self.tagger_bins, name="jet1txbb", label="$T_{Xbb}$ Score")
            .Var(self.tagger_bins, name="jet1th4q", label="$T_{H4q}$ Score")
            .Var(self.pt_bins, name="jet1pt", label="$p_T$ (GeV)")
            .Reg(*self.msd_bins, name="jet1msd", label="$m_{SD}$ (GeV)")
            .Double()
        )

        select = selection.all(*selection.names)

        selections = {
            # select events which pass the muon triggers and selection
            "den": select * muon_triggered,
            # add our triggers
            "num": select * muon_triggered * bbVV_triggered,
        }

        hists = {}

        for key, selection in selections.items():
            hists[key] = h.copy().fill(
                jet1txbb=fatjets.txbb[selection][:, 0].to_numpy(),
                jet1th4q=fatjets.particleNet_H4qvsQCD[selection][:, 0].to_numpy(),
                jet1pt=fatjets.pt[selection][:, 0].to_numpy(),
                jet1msd=fatjets.msoftdrop[selection][:, 0].to_numpy(),
            )

        hists["cutflow"] = cutflow

        return hists

    def postprocess(self, accumulator):
        return accumulator


class JetHT3DTriggerEfficienciesProcessor(processor.ProcessorABC):
    """Accumulates two 2D (pT, msd) histograms from all input events: 1) before triggers, and 2) after triggers"""

    def __init__(self, ak15=True):
        super(JetHT3DTriggerEfficienciesProcessor, self).__init__()

        self.muon_HLTs = {2017: ["IsoMu27", "Mu50"]}

        self.HLTs = {
            2017: [
                "PFJet500",
                "AK8PFJet400",
                "AK8PFJet500",
                "AK8PFJet360_TrimMass30",
                "AK8PFJet380_TrimMass30",
                "AK8PFJet400_TrimMass30",
                "AK8PFHT750_TrimMass50",
                "AK8PFHT800_TrimMass50",
                "PFHT1050",
                # 'AK8PFJet330_PFAK8BTagCSV_p17'
            ]
        }

        self.ak15 = ak15

    def process(self, events):
        """Returns pre- (den) and post- (num) trigger 3D (jet 2 pT, jet 1 pT, jet 1 msd) histograms from input NanoAOD events"""

        # passing single-muon triggers
        muon_triggered = np.any(
            np.array([events.HLT[trigger] for trigger in self.muon_HLTs[2017]]), axis=0
        )

        fatjets = events.FatJetAK15 if self.ak15 else events.FatJet

        # does event have a fat jet
        fatjet1bool = ak.any(fatjets.pt, axis=1)

        # for denominator select events which pass the muon triggers and contain at least one fat jet
        den_selection = fatjet1bool * muon_triggered

        # denominator
        jet2varbins = [0, 250, 300, 350, 400, 500, 750, 1000]

        den = (
            Hist.new.Var(jet2varbins, name="jet2pt", label="AK15 Fat Jet 2 $p_T$ (GeV)")
            .Reg(50, 0, 1000, name="jet1pt", label="AK15 Fat Jet 1 $p_T$ (GeV)")
            .Reg(15, 0, 300, name="jet1msd", label="AK15 Fat Jet 1 MassSD (GeV)")
            .Double()
        ).fill(
            jet1pt=fatjets.pt[den_selection][:, 0].to_numpy(),
            jet1msd=fatjets.msoftdrop[den_selection][:, 0].to_numpy(),
            jet2pt=ak.fill_none(
                ak.pad_none(fatjets.pt[den_selection][:, 1:2], 1, axis=1), 0
            ).to_numpy()[
                :, 0
            ],  # putting events with no fat jet 2 in the low pT bin
        )

        # numerator

        # passing all HLT triggers
        bbVV_triggered = np.any(
            np.array([events.HLT[trigger] for trigger in self.HLTs[2017]]), axis=0
        )

        num_selection = fatjet1bool * muon_triggered * bbVV_triggered

        num = (
            Hist.new.Var(jet2varbins, name="jet2pt", label="AK15 Fat Jet 2 $p_T$ (GeV)")
            .Reg(50, 0, 1000, name="jet1pt", label="AK15 Fat Jet 1 $p_T$ (GeV)")
            .Reg(15, 0, 300, name="jet1msd", label="AK15 Fat Jet 1 MassSD (GeV)")
            .Double()
        ).fill(
            jet1pt=fatjets.pt[num_selection][:, 0].to_numpy(),
            jet1msd=fatjets.msoftdrop[num_selection][:, 0].to_numpy(),
            jet2pt=ak.fill_none(
                ak.pad_none(fatjets.pt[num_selection][:, 1:2], 1, axis=1), 0
            ).to_numpy()[
                :, 0
            ],  # putting events with no fat jet 2 in the low pT bin
        )

        return {"den": den, "num": num}

    def postprocess(self, accumulator):
        return accumulator


class JetHTHybrid3DTriggerEfficienciesProcessor(processor.ProcessorABC):
    """Accumulates two 2D (pT, msd) histograms from all input events: 1) before triggers, and 2) after triggers"""

    def __init__(self):
        super(JetHTHybrid3DTriggerEfficienciesProcessor, self).__init__()

        self.muon_HLTs = {2017: ["IsoMu27", "Mu50"]}

        self.HLTs = {
            2017: [
                "PFJet500",
                "AK8PFJet400",
                "AK8PFJet500",
                "AK8PFJet360_TrimMass30",
                "AK8PFJet380_TrimMass30",
                "AK8PFJet400_TrimMass30",
                "AK8PFHT750_TrimMass50",
                "AK8PFHT800_TrimMass50",
                "PFHT1050",
                # 'AK8PFJet330_PFAK8BTagCSV_p17'
            ]
        }

    def process(self, events):
        """Returns pre- (den) and post- (num) trigger 3D (jet 2 pT, jet 1 pT, jet 1 msd) histograms from input NanoAOD events"""

        try:
            # passing single-muon triggers
            muon_triggered = np.any(
                np.array([events.HLT[trigger] for trigger in self.muon_HLTs[2017]]), axis=0
            )

            ak8fatjet1bool = ak.any(events.FatJet.pt, axis=1)
            ak15fatjet1bool = ak.any(events.FatJetAK15.pt, axis=1)

            hybr_den_selection = ak8fatjet1bool * ak15fatjet1bool * muon_triggered

            events2 = events[hybr_den_selection]

            Txbb_scores = ak.nan_to_num(
                ak.fill_none(
                    ak.pad_none(
                        events2.FatJet.particleNetMD_Xbb
                        / (events2.FatJet.particleNetMD_QCD + events2.FatJet.particleNetMD_Xbb),
                        2,
                        axis=1,
                    ),
                    0,
                ),
                0,
            )
            jet1_bb_leading = Txbb_scores[:, 0:1] >= Txbb_scores[:, 1:2]
            bb_mask = ak.concatenate([jet1_bb_leading, ~jet1_bb_leading], axis=1)

            # check if either of the ak15 fat jets in the event do not overlap
            dR = 0.8
            no_both_overlap = ak.flatten(
                (events2.FatJet[bb_mask].delta_r(events2.FatJetAK15[:, 0]) > dR)
                + (
                    ak.fill_none(
                        events2.FatJet[bb_mask].delta_r(
                            ak.pad_none(events2.FatJetAK15, 2, axis=1)[:, 1]
                        )
                        > dR,
                        False,
                    )
                )
            )

            # only use events which have two non-overlapping fat jets
            events2 = events2[no_both_overlap]
            bb_mask = bb_mask[no_both_overlap]

            Th4q_scores = ak.nan_to_num(
                ak.fill_none(
                    ak.pad_none(
                        events2.FatJetAK15.ParticleNet_probHqqqq
                        / (
                            events2.FatJetAK15.ParticleNet_probHqqqq
                            + events2.FatJetAK15.ParticleNet_probQCDb
                            + events2.FatJetAK15.ParticleNet_probQCDbb
                            + events2.FatJetAK15.ParticleNet_probQCDc
                            + events2.FatJetAK15.ParticleNet_probQCDcc
                            + events2.FatJetAK15.ParticleNet_probQCDothers
                        ),
                        2,
                        axis=1,
                    ),
                    0,
                ),
                0,
            )

            jet1_VV_leading = Th4q_scores[:, 0:1] >= Th4q_scores[:, 1:2]
            VV_mask = ak.concatenate([jet1_VV_leading, ~jet1_VV_leading], axis=1)

            # make sure the leading VV fat jet is not overlapping with the bb one
            overlap = ak.flatten(events2.FatJet[bb_mask].delta_r(events2.FatJetAK15[VV_mask])) < dR

            # pick the other jet if there's overlap
            VV_mask = overlap ^ VV_mask

            bbFatJet = ak.flatten(events2.FatJet[bb_mask])
            VVFatJet = ak.flatten(events2.FatJetAK15[VV_mask])

            jet2varbins = [0, 250, 300, 350, 400, 500, 750, 1000]

            hybr_den = (
                Hist.new.Var(jet2varbins, name="VVjetpt", label="AK15 VV Fat Jet $p_T$ (GeV)")
                .Reg(50, 0, 1000, name="bbjetpt", label="AK8 bb Fat Jet $p_T$ (GeV)")
                .Reg(15, 0, 300, name="bbjetmsd", label="AK8 bb Fat Jet MassSD (GeV)")
                .Double()
            ).fill(
                bbjetpt=bbFatJet.pt.to_numpy(),
                bbjetmsd=bbFatJet.msoftdrop.to_numpy(),
                VVjetpt=VVFatJet.pt.to_numpy(),
            )

            # passing all HLT triggers
            bbVV_triggered = np.any(
                np.array([events2.HLT[trigger] for trigger in self.HLTs[2017]]), axis=0
            )

            bbFatJet = bbFatJet[bbVV_triggered]
            VVFatJet = VVFatJet[bbVV_triggered]

            hybr_num = (
                Hist.new.Var(jet2varbins, name="VVjetpt", label="AK15 VV Fat Jet $p_T$ (GeV)")
                .Reg(50, 0, 1000, name="bbjetpt", label="AK8 bb Fat Jet $p_T$ (GeV)")
                .Reg(15, 0, 300, name="bbjetmsd", label="AK8 bb Fat Jet MassSD (GeV)")
                .Double()
            ).fill(
                bbjetpt=bbFatJet.pt.to_numpy(),
                bbjetmsd=bbFatJet.msoftdrop.to_numpy(),
                VVjetpt=VVFatJet.pt.to_numpy(),
            )

            return {"den": hybr_den, "num": hybr_num}

        except IndexError:
            return {}

    def postprocess(self, accumulator):
        return accumulator
