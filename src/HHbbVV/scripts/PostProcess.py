import utils

class PostProcess:
    def __init__(data_dir,year,filters=None):
        self.data_dir = data_dir
        self.year = year

        from sample_labels import samples, sig_key, data_key
        self.samples = samples
        self.sig_key = sig_key
        self.data_key = data_key
        self.events_dict = load_samples(filters)

        self.bb_masks = self.bb_VV_asignment()

        self.mass = "bbFatJetMsd"
        self.nbins = 17
        self.mass_bins = np.linspace(50, self.nbins*10.0+50.0, self.nbins+1)
        self.blind_bins = [100, 150]

        CUT_MAX_VAL = 9999
        self.pass_masks = {
            "BDTScore": [0.9602, CUT_MAX_VAL],
            "bbFatJetParticleNetMD_Txbb": [0.98, CUT_MAX_VAL],
        }
        self.fail_masks = {
            "bbFatJetParticleNetMD_Txbb": [0.8, 0.98],
        }


    def load_samples(self,filters=None):
        """
        Loads events after a filter.
        Reweights samples by nevents.
        """
        import os
        from os import listdir
        samples_list = listdir(f"{self.data_dir}/{self.year}")

        events_dict = {}
        for label, selector in self.samples.items():
            print(label)
            events_dict[label] = []
            for sample in self.samples_list:
                if not sample.startswith(selector):
                    continue
                
                print(sample)
                
                events = pd.read_parquet(f"{self.data_dir}/{self.year}/{sample}/parquet", filters=filters)
                pickles_path = f"{self.data_dir}/{self.year}/{sample}/pickles"
            
                if label != self.data_key:
                    if label == self.sig_key:
                        n_events = utils.get_cutflow(pickles_path, self.year, sample)["has_4q"]
                    else:
                        n_events = utils.get_nevents(pickles_path, self.year, sample)
                    
                    events["weight"] /= n_events
                    
                events_dict[label].append(events)
            
            events_dict[label] = pd.concat(events_dict[label])
        
        return events_dict

    def apply_weights(self):
        """
        Apply external weights.
        """
        from coffea.lookup_tools.dense_lookup import dense_lookup
        with open(f"../corrections/trigEffs/AK8JetHTTriggerEfficiency_{self.year}.hist", "rb") as filehandler:
            ak8TrigEffs = pickle.load(filehandler)
            
            ak8TrigEffsLookup = dense_lookup(
                np.nan_to_num(ak8TrigEffs.view(flow=False), 0), np.squeeze(ak8TrigEffs.axes.edges)
            )    

        for sample in self.events_dict:
            print(sample)
            events = self.events_dict[sample]
            if sample == "Data":
                events["finalWeight"] = events["weight"]
            else:
                fj_trigeffs = ak8TrigEffsLookup(events["ak8FatJetPt"].values, events["ak8FatJetMsd"].values)
                # combined eff = 1 - (1 - fj1_eff) * (1 - fj2_eff)
                combined_trigEffs = 1 - np.prod(1 - fj_trigeffs, axis=1, keepdims=True)
                events["finalWeight"] = events["weight"] * combined_trigEffs
                
    def bb_VV_asignment(self):
        bb_masks = {}
        for sample, events in self.events_dict.items():
            txbb = events["ak8FatJetParticleNetMD_Txbb"]
            thvv = events["ak8FatJetParticleNetHWWMD_THWW4q"]
            bb_mask = txbb[0] >= txbb[1]
            bb_masks[sample] = pd.concat((bb_mask, ~bb_mask), axis=1)
        return bb_masks

    def get_bdt(self):
        try:
            BDT_samples = [self.sig_key, "QCD", "TT", "Data"]
            bdt_preds = np.load(f"{self.data_dir}/absolute_weights_preds.npy")
            i = 0
            for sample in BDT_samples:
                events = events_dict[sample]
                num_events = len(events)
                events["BDTScore"] = bdt_preds[i : i + num_events]
                i += num_events
        except:
            print(f'No BDT weights saved in {self.data_dir}')
            # should probably evaluate bdt in that case
            # and have function that derives the extra variables needed in event dict
            #import xgboost as xgb
            #model = xgb.XGBClassifier()
            #model.load_model(f"{args.model_dir}/trained_bdt.model")

    def get_templates(self):
        pass_selection, pass_cutflow = utils.make_selection(
            self.pass_masks, self.events_dict, self.bb_masks
        )
        fail_selection, fail_cutflow = utils.make_selection(
            self.fail_masks, self.events_dict, self.bb_masks
        )

        hists = {}
        hists['pass'] = utils.singleVarHist(
            self.events_dict,
            self.mass,
            self.mass_bins,
            r"$m^{bb}$ (GeV)",
            self.bb_masks,
            selection=pass_selection,
            blind_region=blind_bins,
        )
        hists['fail'] = utils.singleVarHist(
            self.events_dict,
            self.mass,
            self.mass_bins,
            r"$m^{bb}$ (GeV)",
            self.bb_masks,
            selection=fail_selection
        )
        
        return hists
