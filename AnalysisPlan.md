# Analysis Plan

- [Analysis Plan](#analysis-plan)
  - [TODOs](#todos)
    - [Incorporate remaining systematics](#incorporate-remaining-systematics)
      - [Skimmer](#skimmer)
      - [Shapes / Values](#shapes--values)
      - [Datacard](#datacard)
    - [Update processor](#update-processor)
    - [New LP SFs](#new-lp-sfs)
    - [Nonresonant](#nonresonant)
    - [Statistical tests of fits](#statistical-tests-of-fits)
    - [Resonant X-\>HY](#resonant-x-hy)
    - [Semi-leptonic ttbar selection skimmer](#semi-leptonic-ttbar-selection-skimmer)
    - [Resolved WW veto](#resolved-ww-veto)
    - [TTbar corrections](#ttbar-corrections)
  - [Plan](#plan)
    - [Feb 13 - 17](#feb-13---17)
    - [Feb 20 - 24](#feb-20---24)
    - [Feb 27 - Mar 3](#feb-27---mar-3)
    - [Mar](#mar)
    - [Apr](#apr)
    - [Apr 24 - 28](#apr-24---28)
    - [Jun 5 - 9](#jun-5---9)
  - [In progress:](#in-progress)
  - [~Completed:](#completed)
    - [Preliminary 2017 cut-based signal and background yields estimate](#preliminary-2017-cut-based-signal-and-background-yields-estimate)
    - [Trigger scale factor measurements](#trigger-scale-factor-measurements)
    - [Processor for skimming nano files](#processor-for-skimming-nano-files)
      - [Update 2/23](#update-223)
    - [Triton/SONIC inference server](#tritonsonic-inference-server)
    - [BDT Training](#bdt-training)
    - [Tagger](#tagger)
    - [Fits, combine](#fits-combine)
    - [Full Run 2 and all UL samples](#full-run-2-and-all-ul-samples)
    - [Lund plane scale factors](#lund-plane-scale-factors)
    - [Post-processing](#post-processing)
      - [Update post-processing](#update-post-processing)


## TODOs


### Incorporate remaining systematics

#### Skimmer
 - [x] Pileup
 - [x] JES/R http://cds.cern.ch/record/2792322/files/DP2021_033.pdf
   - [x] Need to update to latest
 - [x] JMS/R http://cds.cern.ch/record/2256875/files/JME-16-003-pas.pdf
   - [ ] Need UL mSD and regressed mass corrections
 - [ ] Top pt
 - Theory
   - [x] pdf uncertainties
   - [x] scale variation
   - [x] parton shower weights
   - [x] W k factor


#### Shapes / Values
 - [x] Pileup
 - [x] ParticleNet Xbb
   - [ ] Split up uncertainties
 - [x] JES/R
 - [x] JMS/R
 - [x] Trigger SFs
   - [x] Stat. Unc.
   - [ ] Correlated Syst.
 - [ ] Top pt
 - Theory
   - [x] pdf uncertainties
   - [x] parton shower weights
   - [ ] scale variation
   - [ ] W k factor


#### Datacard
 - [x] MC Stats
 - [x] Lumi
 - [x] Pileup
 - [x] Trigger SFs
   - [x] Stat.
   - [ ] Correlated Syst.
 - [x] ParticleNet Xbb
   - [ ] Separate uncertainties
 - [x] JES/R
 - [x] JMS/R
 - [ ] Top pT
 - Theory
   - [x] BR
   - [ ] pdf uncertainties
     - [x] rate
     - [ ] shape?
   - [x] QCD scale
   - [ ] alpha_s (for single Higgs)
   - [x] parton shower weights
   - [ ] scale variation?
   - [ ] W k factor??


### Update processor

- [ ] New e, mu, b-tag jets selections from VHbb
- [ ] Re-run with VV regressed mass for Dijet variables
- [ ] FatJet ID
- [ ] Look at ID scale factors
- [ ] New LP Method


### New LP SFs

 - [ ] Update LP method


### Nonresonant

- [x] BDT ROC
  - [x] Try multi-class BDT
  - [x] Try equalizing background weights
  - [x] Optimize hyperparams
  - [x] Trim features
  - [x] Check sculpting
- [x] Scan Txbb, BDT WPs
- [x] Run over all kL and k_2V
  - [x] Re-run VBF with gen selection
- [x] HH inference
- [ ] Theory uncertainties


### Statistical tests of fits

 - [x] GoF
 - [x] F-test
 - [ ] Impacts


### Resonant X->HY

 - [x] Control plots
 - [x] Preliminary signal region
 - [x] Validation region
 - [x] 2D fit, xUL
   - [x] GoF
   - [x] F-test
 - [x] Run over all signals
   - [ ] TRSM, NMSSM exclusions
 - [x] Scan Txbb, THWW working points


### Semi-leptonic ttbar selection skimmer

 - [ ] Update LP
 - [x] JECs for AK4 Jets
 - [x] DeepJet Btag
 - [x] B-tag scale factors
 - [x] muon ID SFs
 - [x] Save regressed mass
 - [x] Jet ID SFs (not required) https://twiki.cern.ch/twiki/bin/view/CMS/JetID13TeVUL
 - [ ] PFNano on SingleMuon to get regressed mass


### Resolved WW veto

 - [ ] Test veto on 2 AK8 W-tagged jets


### TTbar corrections

May not be necessary given low yield

- Tagger efficiency
- Recoil
- JMS
- Regressed mass
- BDT
- Check VBF?


## Plan


### Feb 13 - 17

Raghav:
 - [x] Trigger efficiencies
 - [x] Add to skimmer:
   - [x] num e, mu
   - [ ] e, mu 4 vectors 
   - [x] ak4 b jets (medium btag) - outside bb jet
   - [x] check control plots on num_x
 - [x] Check vetoes on these

Cristina
 - [x] Systematics


### Feb 20 - 24

Raghav:
 - [x] 2017 fits
   - [x] PR: Systematics into datacard

Cristina:
 - [x] update xsecs
 - [ ] Goodness-of-fit, f-test, impacts 


### Feb 27 - Mar 3

Raghav:
- [x] Full run 2, all samples


### Mar
- [x] Wrap up nonresonant for now with full run 2 limits and GoF
- [x] Develop resonant analysis strategy
- [x] Plots, selections for subset of signals
- [ ] Get 2D fit working


### Apr
- [x] Get 2D fit working
- [x] Upper limits for subset of signals
- [x] Upper limits for all signals!
- [ ] Complete v1 of AN
- [ ] Higher mass ParT training samples


### Apr 24 - 28
- [x] Complete v1 of AN
- [x] B2G Workshop talk
- [x] F-test
- [x] Start WP Scan


### Jun 5 - 9
- [ ] Update semi-leptonic ttbar processor
- [x] Twiki
- [x] Respond to comments
- [ ] Samples
  - [x] Nanogen
- [x] Skimming VBF
- [ ] Runnning over kL, k2V


## In progress:

 - AN: https://gitlab.cern.ch/tdr/notes/AN-21-126


## ~Completed:


### Preliminary 2017 cut-based signal and background yields estimate 
 - using a coarse-grained grid search on pT, msd, and tagger scores
 - 2017 lumi + data only
 - measured for two AK8 fat jets, two AK15 fat jets, and a hybrid (AK8 for bb candidate, AK15 for VV candidate)
 - background estimation from data sidebands
 - with AK8 and AK15 mass-decorrelated ParticleNet Hbb taggers + NOT mass-decorrelated AK8 ParticleNet H4q tagger

### Trigger scale factor measurements
 - Measured for AK8, AK15, and hybrid jets, single-jet 2D (mass, pT binned) efficiencies (applied assuming prob. of each fat jet passing trigger is independent) and 3D (jet 1 mass, jet 1 pt, jet 2 pt binned)  efficiencies ([processors](https://github.com/rkansal47/HHbbVV/blob/main/processors/JetHTTriggerEfficienciesProcessor.py))
 - Decided on AK8 only - ~same sensitivity, significantly easier practically

- [x] Measure for all years
- [x] **Update selection**
- [x] Check if binning in VV tagger is necessary (probably not since only btag is in the trigger)
- [x] Investigate high unc.

### Processor for skimming nano files
https://github.com/rkansal47/HHbbVV/blob/main/processors/bbVVSkimmer.py

 - Currently includes:
   - Signal gen-matching cuts
   - Pre-selection kinematic cuts
   - Inference via triton server running on SDSC
   - Lund plane scale factors for skimmer
   - Saving flat skimmed data to parquet files, and metadata (total events, cutflow) to pickles

#### Update 2/23

- [x] FatJet selections
- [x] Modify JECs code to save only variations of pT
- [x] Add JMS/R
- [x] Regressed mass cut
- [x] Add e, mu, b-tag jets
- [x] Add tagger vars
- [x] Dijet variables

### Triton/SONIC inference server
https://github.com/rkansal47/sonic-models
https://gitlab.nrp-nautilus.io/raghsthebest/triton-server

Server for running inference with our new HWW tagger on samples. 

### BDT Training
https://github.com/rkansal47/HHbbVV/blob/main/src/HHbbVV/scripts/pickle_scripts/TrainBDT.py
     
### Tagger

HWW tagger + mass regression development ([Zichun's tagger repo](https://github.com/zichunhao/weaver) and [Ish's regression repo](https://github.com/ikaul00/weaver))

### Fits, combine


### Full Run 2 and all UL samples

- [x] JetHT
- [x] QCD
- [x] TTbar
- [x] ST
- [x] W, Z+jets
- [x] Diboson
- [x] HHbbWW (all kL)
- Need xsecs for:
  https://docs.google.com/spreadsheets/d/1XQQsN4rl3xGDa35W516TwKyadccpfoT7M1mFGxZ4UjQ/edit#gid=1223976475
  - [x] VBF HHbbWW (kL = 1, k2V = 1)
  - [x] HH4b (kL = 1) (Pre-UL)
  - [x] HWW (ggF, VH, VBF, ttH)
  - [x] Hbb (ggF, VBF, VH, ttH)

### Lund plane scale factors

 - Implemented and validated for top jets in control region
 - Implemented and measured for nonresonant signal

### Post-processing

#### Update post-processing

- [x] Check control plots for all years
- [x] Re-train BDT for all years
- [x] Templates, systematics for all years
- [x] Update datacard with all years