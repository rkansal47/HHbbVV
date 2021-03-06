# Analysis Plan

### ~Completed:

#### Preliminary 2017 cut-based signal and background yields estimate 
 - using a coarse-grained grid search on pT, msd, and tagger scores
 - 2017 lumi + data only
 - measured for two AK8 fat jets, two AK15 fat jets, and a hybrid (AK8 for bb candidate, AK15 for VV candidate)
 - background estimation from data sidebands
 - with AK8 and AK15 mass-decorrelated ParticleNet Hbb taggers + NOT mass-decorrelated AK8 ParticleNet H4q tagger

#### Preliminary 2017 trigger scale factor measurements:
 - Measured for AK8, AK15, and hybrid jets, single-jet 2D (mass, pT binned) efficiencies (applied assuming prob. of each fat jet passing trigger is independent) and 3D (jet 1 mass, jet 1 pt, jet 2 pt binned)  efficiencies ([processors](https://github.com/rkansal47/HHbbVV/blob/main/processors/JetHTTriggerEfficienciesProcessor.py))
 - Tentatively decided to use hybrid case which gave the highest preselection yield and should increase with a better HWW tagger
 - TODO:
   - Repeat using 2017 run B data as well
   - Repeat with UL v2 samples

#### Processor for skimming nano files
https://github.com/rkansal47/HHbbVV/blob/main/processors/bbVVSkimmer.py

 - Currently includes:
   - Signal gen-matching cuts
   - Pre-selection kinematic cuts
   - Inference via triton server running on SDSC
   - Saving flat skimmed data to parquet files, and metadata (total events, cutflow) to pickles
 - TODO:
   - Add Txbb tagger pre-selection cut
   - Add pileup weights
   - Apply trigger SFs
   - Incorporate hybrid sorting

#### Triton/SONIC inference server
https://github.com/rkansal47/sonic-models

Server for running inference with our new HWW tagger on samples. 

#### BDT Training
https://github.com/rkansal47/HHbbVV/blob/main/src/HHbbVV/scripts/pickle_scripts/TrainBDT.py

     
### In progress

1) HWW tagger + mass regression development ([Zichun's tagger repo](https://github.com/zichunhao/weaver) and [Ish's regression repo](https://github.com/ikaul00/weaver))

2) AN write-up https://gitlab.cern.ch/tdr/notes/AN-21-126


#### Samples

Currently have for UL 2017:

 - Data: JetHT RunB-F
 - QCD HT 300-inf
 - TT hadronic, TT semileptonic, T Mtt
 - ST s-channel leptonic, ST t-channel, Inclusive, ST tW top + antitop no fully hadronic
 - ggF HWW inclusive
 - VV, V+jets

1) Requesting UL samples
    - Requesting HH signal samples ggF and VBF for bbWW all-hadronic
    - TODO: Request UL ggH (and VBF?) signal samples for HtoWW inclusive

2) Need to use `RunIISummer20UL17` in PFNano for next production


### TODO:

1) Background estimation...


