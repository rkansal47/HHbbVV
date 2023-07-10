# HHbbVV

[![Codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/rkansal47/HHbbVV/main.svg)](https://results.pre-commit.ci/latest/github/rkansal47/HHbbVV/main)

<p align="left">
  <img width="300" src="https://raw.githubusercontent.com/rkansal47/HHbbVV/main/figure.png" />
</p>

Search for two boosted (high transverse momentum) Higgs bosons (H) decaying to two beauty quarks (b) and two vector bosons (V). The majority of the analysis uses a columnar framework to process input tree-based [NanoAOD](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD) files using the [coffea](https://coffeateam.github.io/coffea/) and [scikit-hep](https://scikit-hep.org) Python libraries.


- [HHbbVV](#hhbbvv)
  - [Instructions for running coffea processors](#instructions-for-running-coffea-processors)
    - [Coffea-Casa](#coffea-casa)
    - [Condor](#condor)
      - [TODO: instructions for lpcjobqueue (currently quite buggy)](#todo-instructions-for-lpcjobqueue-currently-quite-buggy)
  - [Processors](#processors)
    - [JetHTTriggerEfficiencies](#jethttriggerefficiencies)
    - [bbVVSkimmer](#bbvvskimmer)
    - [TaggerInputSkimmer](#taggerinputskimmer)
    - [TTScaleFactorsSkimmer](#ttscalefactorsskimmer)
  - [Condor Scripts](#condor-scripts)
    - [Check jobs](#check-jobs)
    - [Combine pickles](#combine-pickles)
  - [Post Processing](#post-processing)
    - [BDT Pre-Processing](#bdt-pre-processing)
    - [BDT Trainings](#bdt-trainings)
    - [Post-Processing](#post-processing-1)
      - [Making separate background and signal templates for scan (resonant)](#making-separate-background-and-signal-templates-for-scan-resonant)
    - [Create Datacard](#create-datacard)
    - [PlotFits](#plotfits)
  - [Combine](#combine)
    - [CMSSW + Combine Quickstart](#cmssw--combine-quickstart)
    - [Run fits and diagnostics locally](#run-fits-and-diagnostics-locally)
    - [Run fits on condor](#run-fits-on-condor)
  - [Misc](#misc)
    - [Command for copying directories to PRP in background](#command-for-copying-directories-to-prp-in-background)
    - [Get all running condor job names:](#get-all-running-condor-job-names)
    - [Crab data jobs recovery](#crab-data-jobs-recovery)


## Instructions for running coffea processors

General note: Coffea-casa is faster and more convenient, however still somewhat experimental so for large of inputs and/or processors which may require heavier cpu/memory usage (e.g. bbVVSkimmer) condor is recommended.

### [Coffea-Casa](https://coffea-casa.readthedocs.io/en/latest/cc_user.html)
1. after following instructions in the link ^ set up an account, open the coffea-casa GUI (https://cmsaf-jh.unl.edu) and create an image
2. open `src/runCoffeaCasa.ipynb`
3. import your desired processor, specify it in the `run_uproot_job` function, and specify your filelist
4. run the first three cells


### Condor

Manually splits up the files into condor jobs.

```bash
git clone https://github.com/rkansal47/HHbbVV/
cd HHbbVV
TAG=Aug18_skimmer
# will need python3 (recommended to set up via miniconda)
python src/condor/submit.py --processor skimmer --tag $TAG --files-per-job 20  
for i in condor/$TAG/*.jdl; do condor_submit $i; done
```

Alternatively, can be submitted from a yaml file:

```bash
python src/condor/submit_from_yaml.py --year 2017 --processor skimmer --tag $TAG --yaml src/condor/submit_configs/skimmer_inputs_07_24.yaml 
```

To test locally first (recommended), can do e.g.:

```bash
mkdir outfiles
python -W ignore src/run.py --starti 0 --endi 1 --year 2017 --processor skimmer --executor iterative --samples HWW --subsamples GluGluToHHTobbVV_node_cHHH1_pn4q
```

#### TODO: instructions for lpcjobqueue (currently quite buggy)

## Processors

### JetHTTriggerEfficiencies

Applies a muon pre-selection and accumulates 4D ([Txbb, Th4q, pT, mSD]) yields before and after our triggers.

To test locally:

```bash
python -W ignore src/run.py --year 2018 --processor trigger --sample SingleMu2017 --subsamples SingleMuon_Run2018B --starti 0 --endi 1
```

And to submit all:

```bash
nohup bash -c 'for i in 2016 2016APV 2017 2018; do python src/condor/submit.py --year $i --tag '"${TAG}"' --processor trigger --submit; done' &> tmp/submitout.txt &
```

### bbVVSkimmer

Applies pre-selection cuts, runs inference with our new HVV tagger, and saves unbinned branches as parquet files.

Parquet and pickle files will be saved in the eos directory of specified user at path `~/eos/bbVV/skimmer/<tag>/<sample_name>/<parquet or pickles>`. Pickles are in the format `{'nevents': int, 'cutflow': Dict[str, int]}`.

To test locally:

```bash
python -W ignore src/run.py --processor skimmer --year 2017 --samples HH --subsamples GluGluToHHTobbVV_node_cHHH1 --save-systematics --starti 0 --endi 1
```

Or on a specific file(s):

```bash
python -W ignore src/run.py --processor skimmer --year 2017 --files $FILE --files-name GluGluToHHTobbVV_node_cHHH1
```

Jobs

```bash
nohup python src/condor/submit_from_yaml.py --year 2017 --tag $TAG --processor skimmer --save-systematics --submit --yaml src/condor/submit_configs/skimmer_inputs_23_02_17.yaml &> tmp/submitout.txt &
```

All years:

```bash
nohup bash -c 'for year in 2016APV 2016 2017 2018; do python src/condor/submit_from_yaml.py --year $year --tag '"${TAG}"' --processor skimmer --save-systematics --submit --yaml src/condor/submit_configs/skimmer_inputs_23_02_17.yaml; done' &> tmp/submitout.txt &
```


To Submit (if not using the --submit flag)
```bash
nohup bash -c 'for i in condor/'"${TAG}"'/*.jdl; do condor_submit $i; done' &> tmp/submitout.txt &
```

Or just signal:

```bash
python src/condor/submit.py --year 2017 --tag $TAG --samples HH --subsamples GluGluToHHTobbVV_node_cHHH1 --processor skimmer --submit
```


### TaggerInputSkimmer

Applies a loose pre-selection cut, saves ntuples with training inputs.

To test locally:
```bash
python -W ignore src/run.py --year 2017 --starti 300 --endi 301 --samples HWWPrivate --subsamples jhu_HHbbWW --processor input --label AK15_H_VV
python -W ignore src/run.py --year 2017 --starti 300 --endi 301 --samples QCD --subsamples QCD_Pt_1000to1400 --processor input --label AK15_QCD --njets 1 --maxchunks 1
```

Jobs:
```bash
python src/condor/submit_from_yaml.py --year 2017 --tag $TAG --processor input --save-ak15 --yaml src/condor/submit_configs/training_inputs_07_21.yaml 
python src/condor/submit_from_yaml.py --year 2017 --tag $TAG --processor input --yaml src/condor/submit_configs/training_inputs_09_16.yaml --jet AK8
```
To submit add `--submit` flag.


### TTScaleFactorsSkimmer

Applies cuts for a semi-leptonic ttbar control region, as defined for the [JMAR W SF](https://indico.cern.ch/event/1101433/contributions/4775247/) and [CASE Lund Plane SF](https://indico.cern.ch/event/1208247/#10-lund-plane-reweighting-for) measurements to validate Lund plane scale factors.

Lund plane scale factors are calculated for top-matched jets in semi-leptonic ttbar events.

To test locally:
```bash
python -W ignore src/run.py --year 2018 --processor ttsfs --sample TTbar --subsamples TTToSemiLeptonic --starti 0 --endi 1
```

Jobs:
```bash
nohup python src/condor/submit_from_yaml.py --year 2018 --tag $TAG --processor ttsfs --submit --yaml src/condor/submit_configs/ttsfs_inputs_12_4.yaml &> submitout.txt &
```

Or to submit only the signal:

```bash
python src/condor/submit.py --year 2018 --tag $TAG --sample TTbar --subsamples TTToSemiLeptonic --processor ttsfs --submit
```


## Condor Scripts

### Check jobs

Check that all jobs completed by going through output files:

```bash
for year in 2016APV 2016 2017 2018; do python src/condor/check_jobs.py --tag $TAG --processor trigger (--submit) --year $year; done
```

nohup version:

(Do `condor_q | awk '{ print $9}' | grep -o '[^ ]*\.sh' > running_jobs.txt` first to get a list of jobs which are running.)

```bash
nohup bash -c 'for year in 2016APV 2016 2017 2018; do python src/condor/check_jobs.py --year $year --tag '"${TAG}"' --processor skimmer --submit --check-running; done' &> tmp/submitout.txt &
```

### Combine pickles

Combine all output pickles into one:

```bash
for year in 2016APV 2016 2017 2018; do python src/condor/combine_pickles.py --tag $TAG --processor trigger --r --year $year; done
```


## Post Processing

In `src/HHbbVV/postprocessing':


### BDT Pre-Processing

```bash
python BDTPreProcessing.py --data-dir "../../../../data/skimmer/Feb24/" --signal-data-dir "../../../../data/skimmer/Jun10/" --plot-dir "../../../plots/BDTPreProcessing/$TAG/" --year "2017" --bdt-data (--control-plots)
```


### BDT Trainings

```bash
python TrainBDT.py --model-dir testBDT --data-path "../../../../data/skimmer/Feb24/" --year "all" (--test)
```

Inference-only:
```bash
python TrainBDT.py  --data-path "../../../../data/skimmer/Feb24/bdt_data" --year "all" --inference-only --model-dir "../../../../data/skimmer/Feb24/23_05_12_multiclass_rem_feats_3"
```

### Post-Processing

```bash
python postprocessing.py --templates --year "2017" --template-dir "templates/$TAG/" --plot-dir "../../../plots/PostProcessing/$TAG/" --data-dir "../../../../data/skimmer/Feb24/" (--resonant --signal-data-dir "" --control-plots)
```

All years (non-resonant):
```bash
for year in 2016 2016APV 2017 2018; do python -u postprocessing.py --templates --year $year --template-dir "templates/Jun14" --data-dir "../../../../data/skimmer/Feb24" --signal-data-dir "../../../../data/skimmer/Jun10" --bdt-preds-dir "../../../../data/skimmer/Feb24/23_05_12_multiclass_rem_feats_3/inferences"; done
```

Scan (non-resonant):
```bash
for year in 2016 2016APV 2017 2018; do python -u postprocessing.py --templates --year $year --template-dir "templates/$TAG/" --data-dir "../../../../data/skimmer/Feb24/" --old-processor --nonres-txbb-wp "LP" "MP" "HP" --nonres-bdt-wp 0.995 0.998 0.999 --no-do-jshifts; done
```

#### Making separate background and signal templates for scan (resonant)

Background and data:

```bash
nohup bash -c 'for year in 2016APV 2016 2017 2018; do python -u postprocessing.py --templates --year $year --template-dir "/eos/uscms/store/user/rkansal/bbVV/templates/$TAG/" --data-dir "/eos/uscms/store/user/rkansal/bbVV/skimmer/Feb24" --resonant --sig-samples "" --res-txbb-wp LP MP HP --res-thww-wp 0.4 0.6 0.8 0.9 0.94 0.96 0.98 --no-do-jshifts --templates-name backgrounds --old-processor; done' &> outs/bgout.txt & 
```

Signal:

```bash
nohup bash -c 'for sample in NMSSM_XToYHTo2W2BTo4Q2B_MX-4000_MY-150 NMSSM_XToYHTo2W2BTo4Q2B_MX-3000_MY-250; do for year in 2016APV 2016 2017 2018; do python -u postprocessing.py --templates --year $year --template-dir "/eos/uscms/store/user/rkansal/bbVV/templates/$TAG/" --data-dir "/eos/uscms/store/user/rkansal/bbVV/skimmer/Feb24" --signal-data-dir "/eos/uscms/store/user/rkansal/bbVV/skimmer/Apr11" --resonant --sig-samples $sample --bg-keys "" --res-txbb-wp LP MP HP --res-thww-wp 0.4 0.6 0.8 0.9 0.94 0.96 0.98 --no-do-jshifts --templates-name $sample --no-data; done; done' &> outs/sigout.txt & 
```

### Create Datacard

Need `root==6.22.6`, and `square_coef` branch of https://github.com/rkansal47/rhalphalib installed (`pip install -e . --user` after checking out the branch). `CMSSW_11_2_0` recommended.

```bash
python3 postprocessing/CreateDatacard.py --templates-dir templates/$TAG --model-name $TAG (--resonant)
```

Or with separate templates for background and signal:

```bash
python3 -u postprocessing/CreateDatacard.py --templates-dir "/eos/uscms/store/user/rkansal/bbVV/templates/23Apr30Scan/txbb_HP_thww_0.96" \
--sig-separate --resonant --model-name $sample --sig-sample $sample
```

Scan (non-resonant):
```bash
for txbb_wp in "LP" "MP" "HP"; do for bdt_wp in 0.994 0.99 0.96 0.9 0.8 0.6 0.4; do python3 -u postprocessing/CreateDatacard.py --templates-dir templates/23May13NonresScan/txbb_${txbb_wp}_bdt_${bdt_wp} --model-name 23May13NonresScan/txbb_${txbb_wp}_bdt_${bdt_wp} --no-do-jshifts --nTF 0; done; done
```

Datacards with different orders of TFs for F-tests:

Use the `src/HHbbVV/combine/F_test_res.sh` script.


### PlotFits

```bash
python PlotFits.py --fit-file "cards/test_tied_stats/fitDiagnosticsBlindedBkgOnly.root" --plots-dir "../../../plots/PostFit/09_02/"
```

## Combine

### CMSSW + Combine Quickstart
```bash
cmsrel CMSSW_11_2_0
cd CMSSW_11_2_0/src
cmsenv
git clone -b py3 https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
git clone -b v2.0.0 https://github.com/cms-analysis/CombineHarvester.git CombineHarvester
scramv1 b clean; scramv1 b
```

I also add this to my .bashrc for convenience:

```
export PATH="$PATH:/uscms_data/d1/rkansal/HHbbVV/src/HHbbVV/combine"

csubmit() {
    local file=$1; shift;
    python "/uscms_data/d1/rkansal/HHbbVV/src/HHbbVV/combine/submit/submit_${file}.py" "$@"
}
```

### Run fits and diagnostics locally

All via the below script, with a bunch of options (see script):

```bash
run_blinded.sh --workspace --bfit --limits
```

### Run fits on condor

Can run over all the resonant signals (default) or scan working points for a subset of signals (`--scan`)

```bash
csubmit cards --test --scan --resonant --templates-dir 23Apr30Scan
```

Generate toys and fits for F-tests (after making cards and b-only fits)

```bash
csubmit f_test --tag 23May2 --cards-tag 23May2 --low1 0 --low2 0
```

Bias tests:

```bash
for bias in 0 0.15 0.3
do
  csubmit bias --seed 42 --num-jobs 10 --toys-per-job 10 --bias $bias --submit
done
```


## Misc

### Command for copying directories to PRP in background 

([krsync](https://serverfault.com/a/887402))
(you may need to install `rsync` in the PRP pod first if it's been restarted)

```bash
cd ~/eos/bbVV/input/${TAG}_${JETS}/2017
mkdir ../copy_logs
for i in *; do echo $i && sleep 3 && (nohup sh -c "krsync -av --progress --stats $i/root/ $HWWTAGGERDEP_POD:/hwwtaggervol/training/$FOLDER/$i" &> ../copy_logs/$i.txt &) done
```


### Get all running condor job names:

```bash
condor_q | awk '{ print $9}' | grep -o '[^ ]*\.sh'
```


### Crab data jobs recovery

Kill each task:

```bash
dataset=SingleMuon

for i in {B..F}; do crab kill -d crab/pfnano_v2_3/crab_pfnano_v2_3_2016_${dataset}_Run2016${i}*HIPM; done
for i in {F..H}; do crab kill -d crab/pfnano_v2_3/crab_pfnano_v2_3_2016_${dataset}_Run2016$i; done
for i in {B..F}; do crab kill -d crab/pfnano_v2_3/crab_pfnano_v2_3_2017_${dataset}_Run2017$i; done
for i in {A..D}; do crab kill -d crab/pfnano_v2_3/crab_pfnano_v2_3_2018_${dataset}_Run2018$i; done
```

Get a crab report for each task:

```bash
for i in {B..F}; do crab report -d crab/pfnano_v2_3/crab_pfnano_v2_3_2016_${dataset}_Run2016${i}*HIPM; done
for i in {F..H}; do crab report -d crab/pfnano_v2_3/crab_pfnano_v2_3_2016_${dataset}_Run2016$i; done
for i in {B..F}; do crab report -d crab/pfnano_v2_3/crab_pfnano_v2_3_2017_${dataset}_Run2017$i; done
for i in {A..D}; do crab report -d crab/pfnano_v2_3/crab_pfnano_v2_3_2018_${dataset}_Run2018$i; done
```

Combine the lumis to process:

```bash
mkdir -p recovery/$dataset/
# shopt extglob
# jq -s 'reduce .[] as $item ({}; . * $item)' crab/pfnano_v2_3/crab_pfnano_v2_3_2016_${dataset}_*HIPM/results/notFinishedLumis.json > recovery/$dataset/2016APVNotFinishedLumis.json
for year in 2016 2017 2018; do jq -s 'reduce .[] as $item ({}; . * $item)' crab/pfnano_v2_3/crab_pfnano_v2_3_${year}_${dataset}_*/results/notFinishedLumis.json > recovery/$dataset/${year}NotFinishedLumis.json; done
```

Finally, add these as a lumimask for the recovery task.