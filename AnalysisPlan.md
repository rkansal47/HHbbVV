# Analysis Plan

- [Analysis Plan](#analysis-plan)
  - [TODOs](#todos)
    - [Trigger Efficiencies](#trigger-efficiencies)
    - [Incorporate remaining systematics](#incorporate-remaining-systematics)
      - [Weights](#weights)
      - [Values](#values)
      - [Datacard](#datacard)
    - [Full Run 2 and all samples](#full-run-2-and-all-samples)
      - [Update Processor](#update-processor)
    - [Scans](#scans)
    - [TTbar corrections](#ttbar-corrections)
    - [Statistical tests of fits](#statistical-tests-of-fits)
    - [Resonant X-\>HY sensitivity](#resonant-x-hy-sensitivity)
  - [Plan](#plan)
    - [Feb 13 - 17](#feb-13---17)
    - [Feb 20 - 24](#feb-20---24)
    - [Feb 27 - 3](#feb-27---3)
    - [In progress:](#in-progress)
  - [~Completed:](#completed)
    - [Preliminary 2017 cut-based signal and background yields estimate](#preliminary-2017-cut-based-signal-and-background-yields-estimate)
    - [Preliminary 2017 trigger scale factor measurements](#preliminary-2017-trigger-scale-factor-measurements)
    - [Processor for skimming nano files](#processor-for-skimming-nano-files)
    - [Triton/SONIC inference server](#tritonsonic-inference-server)
    - [BDT Training](#bdt-training)
    - [Tagger](#tagger)
    - [Fits, combine](#fits-combine)
    - [Samples generation](#samples-generation)


## TODOs

### Trigger Efficiencies

 - [ ] Measure for all years
 - [x] **Update selection**
 - [x] Check if binning in VV tagger is necessary (probably not since only btag is in the trigger)


### Incorporate remaining systematics

#### Weights
 - Measured
   - [ ] Pileup
   - [ ] Trigger SFs
     - [ ] Stat.
     - [ ] Correlated Syst.
   - [x] ParticleNet Xbb
 - Theory
   - [ ] pdf uncertainties
   - [ ] scale variation?
   - [ ] parton shower weights
   - [ ] W k factor??

#### Values
   - [ ] JES/R http://cds.cern.ch/record/2792322/files/DP2021_033.pdf

#### Datacard
   - [x] MC Stats
   - [ ] Lumi
   - [ ] JMS/R (Need to re-derive?) http://cds.cern.ch/record/2256875/files/JME-16-003-pas.pdf


### Full Run 2 and all samples

- [x] JetHT
- [x] QCD
- [x] TTbar
- [ ] ST: Figure out error in 2017)
- [x] W, Z+jets
- [x] Diboson
- [ ] HH4b (kL = 1): Re-run PFNano on preUL (no UL)
- [x] HHbbWW (all kL)
- [x] VBF HHbbWW (kL = 1, k2V = 1)
- [x] Hbb, HWW (ggF, VH, VBF, ttH)
  - [ ] Missing VBF, VH ttH Hbb
  - [ ] Missing VH HWW

#### Update Processor

- [ ] FatJet selections
- [ ] Regressed mass cut
- [ ] Add e, mu, b-tag jets


### Scans

 - kL
 - C2?


### TTbar corrections

 - Tagger efficiency
 - Recoil
 - JMS
 - Regressed mass
 - BDT
 - Check VBF?


### Statistical tests of fits

 - f-test
 - GoF
 - Impacts


### Resonant X->HY sensitivity

 - Run on 1-2 mass points
 - Check selection, fits, sensitivity


## Plan


### Feb 13 - 17

Raghav:
 - Trigger efficiencies
 - Full run 2, all samples
 - Add to skimmer:
   - num e, mu
   - e, mu 4 vectors 
   - ak4 b jets (medium btag)
 - Check vetoes on these

Cristina
 - Systematics


### Feb 20 - 24

 - Full run 2 fits
   - Goodness-of-fit, f-test, impacts 
 - Scans repository

### Feb 27 - 3

 - ttbar


### In progress:

 - Applying Lund plane scale factors to non-resonant signal
 - New fits, limits on HH with scale factors
 - AN: https://gitlab.cern.ch/tdr/notes/AN-21-126


## ~Completed:



### Preliminary 2017 cut-based signal and background yields estimate 
 - using a coarse-grained grid search on pT, msd, and tagger scores
 - 2017 lumi + data only
 - measured for two AK8 fat jets, two AK15 fat jets, and a hybrid (AK8 for bb candidate, AK15 for VV candidate)
 - background estimation from data sidebands
 - with AK8 and AK15 mass-decorrelated ParticleNet Hbb taggers + NOT mass-decorrelated AK8 ParticleNet H4q tagger

### Preliminary 2017 trigger scale factor measurements
 - Measured for AK8, AK15, and hybrid jets, single-jet 2D (mass, pT binned) efficiencies (applied assuming prob. of each fat jet passing trigger is independent) and 3D (jet 1 mass, jet 1 pt, jet 2 pt binned)  efficiencies ([processors](https://github.com/rkansal47/HHbbVV/blob/main/processors/JetHTTriggerEfficienciesProcessor.py))
 - Tentatively decided to use hybrid case which gave the highest preselection yield and should increase with a better HWW tagger

### Processor for skimming nano files
https://github.com/rkansal47/HHbbVV/blob/main/processors/bbVVSkimmer.py

 - Currently includes:
   - Signal gen-matching cuts
   - Pre-selection kinematic cuts
   - Inference via triton server running on SDSC
   - Lund plane scale factors for skimmer
   - Saving flat skimmed data to parquet files, and metadata (total events, cutflow) to pickles
 - TODO:
   - Add Txbb tagger pre-selection cut
   - Apply trigger SFs
   - Add remaining systematics

### Triton/SONIC inference server
https://github.com/rkansal47/sonic-models
https://gitlab.nrp-nautilus.io/raghsthebest/triton-server

Server for running inference with our new HWW tagger on samples. 

### BDT Training
https://github.com/rkansal47/HHbbVV/blob/main/src/HHbbVV/scripts/pickle_scripts/TrainBDT.py
     
### Tagger

HWW tagger + mass regression development ([Zichun's tagger repo](https://github.com/zichunhao/weaver) and [Ish's regression repo](https://github.com/ikaul00/weaver))

### Fits, combine


### Samples generation

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

