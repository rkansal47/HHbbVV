import awkward as ak
from coffea.processor import ProcessorABC, column_accumulator
import numpy as np
from coffea.analysis_tools import PackedSelection
import pickle


class bbVVSkimmer(ProcessorABC):
    """
    Skims nanoaod files, saving selected branches and events passing preselection cuts (and triggers for data), for preliminary cut-based analysis and BDT studies

    Args:
        xsecs (dict, optional): sample cross sections, if sample not included no lumi and xsec will not be applied to weights
        condor (bool, optional): using normal condor or not - if not, post processing will not divide by original total events
    """
    # TODO: do ak8, ak15 sorting for hybrid case

    def __init__(self, xsecs = {}, condor: bool = False):
        super(bbVVSkimmer, self).__init__()

        # in pb^-1
        self.LUMI = {'2017': 40000}

        # in pb
        self.XSECS = xsecs

        self.condor = condor

        self.HLTs = {
            '2017':   [
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

        # key is name in nano files, value will be the name in the skimmed output
        self.skim_vars = {
            'FatJet': {
                'eta': 'Eta',
                'phi': 'Phi',
                'mass': 'Mass',
                'msoftdrop': 'Msd',
                'pt': 'Pt',

                'particleNetMD_QCD': 'ParticleNetMD_QCD',
                'particleNetMD_Xbb': 'ParticleNetMD_Xbb',
                'particleNetMD_Xcc': 'ParticleNetMD_Xcc',
                'particleNetMD_Xqq': 'ParticleNetMD_Xqq',
                'particleNet_H4qvsQCD': 'ParticleNet_Th4q',
            },
            'FatJetAK15': {
                'eta': 'Eta',
                'phi': 'Phi',
                'mass': 'Mass',
                'msoftdrop': 'Msd',
                'pt': 'Pt',

                'ParticleNetMD_probQCD': 'ParticleNetMD_probQCD',
                'ParticleNetMD_probQCDb': 'ParticleNetMD_probQCDb',
                'ParticleNetMD_probQCDbb': 'ParticleNetMD_probQCDbb',
                'ParticleNetMD_probQCDc': 'ParticleNetMD_probQCDc',
                'ParticleNetMD_probQCDcc': 'ParticleNetMD_probQCDcc',
                'ParticleNetMD_probXbb': 'ParticleNetMD_probXbb',
                'ParticleNetMD_probXcc': 'ParticleNetMD_probXcc',
                'ParticleNetMD_probXqq': 'ParticleNetMD_probXqq',
                'ParticleNet_probHbb': 'ParticleNet_probHbb',
                'ParticleNet_probHcc': 'ParticleNet_probHcc',
                'ParticleNet_probHqqqq': 'ParticleNet_probHqqqq',
                'ParticleNet_probQCDb': 'ParticleNet_probQCDb',
                'ParticleNet_probQCDbb': 'ParticleNet_probQCDbb',
                'ParticleNet_probQCDc': 'ParticleNet_probQCDc',
                'ParticleNet_probQCDcc': 'ParticleNet_probQCDcc',
                'ParticleNet_probQCDothers': 'ParticleNet_probQCDothers',
            },
            'other': {
                'MET_pt': 'MET_pt'
            }
        }

        self.preselection_cut_vals = {'pt': 250, 'msd': 20}

        with open('processors/2017_pileupweight.pkl', 'rb') as filehandler:
            self.pileupweight_lookup = pickle.load(filehandler)


    def process(self, events):
        """ Returns skimmed events which pass preselection cuts (and triggers if data) and with the branches listed in self.skim_vars """

        print("processing")

        year = events.metadata['dataset'][:4]
        dataset = events.metadata['dataset'][5:]

        n_events = len(events)
        isData = 'JetHT' in dataset
        selection = PackedSelection()


        def pad_val(arr, target, value, axis=0, to_numpy=True):
            ret = ak.fill_none(ak.pad_none(arr, target, axis=axis, clip=True), value)
            return ret.to_numpy() if to_numpy else ret


        # TODO: gen vars


        # triggers

        if isData:
            HLT_triggered = np.any(np.array([events.HLT[trigger] for trigger in self.HLTs[year] if trigger in events.HLT.fields]), axis=0)
            selection.add('trigger', HLT_triggered)

        # pre-selection cuts
        # ORing ak8 and ak15 cuts

        preselection_cut = np.logical_or(   np.prod(pad_val((events.FatJet.pt > self.preselection_cut_vals['pt']) * (events.FatJet.msoftdrop > self.preselection_cut_vals['msd']), 2, False, axis=1), axis=1),
                                            np.prod(pad_val((events.FatJetAK15.pt > self.preselection_cut_vals['pt']) * (events.FatJetAK15.msoftdrop > self.preselection_cut_vals['msd']), 2, False, axis=1), axis=1))
        selection.add('preselection', preselection_cut)

        # TODO: trigger SFs


        # select vars

        ak8FatJetVars = {f'ak8FatJet{key}': pad_val(events.FatJet[var], 2, -1, axis=1) for (var, key) in self.skim_vars['FatJet'].items()}
        ak15FatJetVars = {f'ak15FatJet{key}': pad_val(events.FatJetAK15[var], 2, -1, axis=1) for (var, key) in self.skim_vars['FatJetAK15'].items()}
        otherVars = {key: events[var.split('_')[0]]["_".join(var.split('_')[1:])].to_numpy() for (var, key) in self.skim_vars['other'].items()}

#         skimmed_events = {**ak8FatJetVars, **ak15FatJetVars, **otherVars}

        import sys
        if sys.version_info[1] < 9: skimmed_events = {**ak8FatJetVars, **ak15FatJetVars, **otherVars}
        else: skimmed_events = ak8FatJetVars | ak15FatJetVars | otherVars  # this is fancier

        # particlenet h4q vs qcd, xbb vs qcd

        skimmed_events['ak8FatJetParticleNetMD_Txbb'] = pad_val(events.FatJet.particleNetMD_Xbb / (events.FatJet.particleNetMD_QCD + events.FatJet.particleNetMD_Xbb), 2, -1, axis=1)
        skimmed_events['ak15FatJetParticleNetMD_Txbb'] = pad_val(events.FatJetAK15.ParticleNetMD_probXbb / (events.FatJetAK15.ParticleNetMD_probQCD + events.FatJetAK15.ParticleNetMD_probXbb), 2, -1, axis=1)
        skimmed_events['ak15FatJetParticleNet_Th4q'] = pad_val(events.FatJetAK15.ParticleNet_probHqqqq / (  events.FatJetAK15.ParticleNet_probHqqqq +
                                                                                                    events.FatJetAK15.ParticleNet_probQCDb
                                                                                                    + events.FatJetAK15.ParticleNet_probQCDbb
                                                                                                    + events.FatJetAK15.ParticleNet_probQCDc
                                                                                                    + events.FatJetAK15.ParticleNet_probQCDcc
                                                                                                    + events.FatJetAK15.ParticleNet_probQCDothers
                                                                                                    ), 2, -1, axis=1)

        # calc weights

        skimmed_events['weight'] = np.ones(n_events) if isData else (events.genWeight * self.pileupweight_lookup(events.Pileup.nPU)).to_numpy()

        # apply selections

        skimmed_events = {
            key: column_accumulator(value[selection.all(*selection.names)]) for (key, value) in skimmed_events.items()
        }

        return {
            year: {
                dataset: {
                        'nevents': n_events,
                        'skimmed_events': skimmed_events,
                }
            }
        }


    def postprocess(self, accumulator):
        """
        Multiplies weights by luminosity and cross sections (if specified in input)

        If not using normal condor, will also 1) divide by total events (pre-cuts) and 2) convert `column_accumulator`s to normal arrays
        If using normal condor, this will have to be done manually, along with combining all the output files
        """

        for year, datasets in accumulator.items():
            for dataset, output in datasets.items():
                output['skimmed_events'] = {
                    key: value.value for (key, value) in output['skimmed_events'].items()
                }

                if 'JetHT' not in dataset:
                    weight = 1 if self.condor else 1 / output['nevents']
                    if dataset in self.XSECS:
                        weight *= self.LUMI[year] * self.XSECS[dataset]
                    output['skimmed_events']['weight'] *= weight

                if self.condor: output['skimmed_events'] = {key: column_accumulator(value) for (key, value) in output['skimmed_events'].items()}

        return accumulator
