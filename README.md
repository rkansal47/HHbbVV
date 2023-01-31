# HHbbVV

[![Codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/rkansal47/HHbbVV/main.svg)](https://results.pre-commit.ci/latest/github/rkansal47/HHbbVV/main)

<p align="left">
  <img width="300" src="https://raw.githubusercontent.com/rkansal47/HHbbVV/main/figure.png" />
</p>

Search for two boosted (high transverse momentum) Higgs bosons (H) decaying to two beauty quarks (b) and two vector bosons (V). The majority of the analysis uses a columnar framework to process input tree-based [NanoAOD](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD) files using the [coffea](https://coffeateam.github.io/coffea/) and [scikit-hep](https://scikit-hep.org) Python libraries.

  - [Instructions for running coffea processors](#instructions-for-running-coffea-processors)
    * [Coffea-Casa](#coffea-casa)
    * [Condor](#condor)
  - [Processors](#processors)
    * [bbVVSkimmer](#bbvvskimmer)
    * [TaggerInputSkimmer](#taggerinputskimmer)
    * [TTScaleFactorsSkimmer](#ttscalefactorsskimmer)
  - [Misc](#misc)


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
python src/condor/submit_from_yaml.py --year 2017 --processor skimmer --tag $TAG --files-per-job 20 --yaml src/condor/submit_configs/skimmer_inputs_07_24.yaml 
```

To test locally first (recommended), can do e.g.:

```bash
mkdir outfiles
python -W ignore src/run.py --starti 0 --endi 1 --year 2017 --processor skimmer --executor iterative --samples HWW --subsamples GluGluToHHTobbVV_node_cHHH1_pn4q
```

#### TODO: instructions for lpcjobqueue (currently quite buggy)

## Processors

### bbVVSkimmer

Applies pre-selection cuts, runs inference with our new HVV tagger, and saves unbinned branches as parquet files.

Parquet and pickle files will be saved in the eos directory of specified user at path `~/eos/bbVV/skimmer/<tag>/<sample_name>/<parquet or pickles>`. Pickles are in the format `{'nevents': int, 'cutflow': Dict[str, int]}`.

To test locally:

`python -W ignore src/run.py --processor skimmer --year 2017 --samples HH --subsamples GluGluToHHTobbVV_node_cHHH1 --starti 0 --endi 1`

Jobs
```bash
nohup python src/condor/submit_from_yaml.py --year 2017 --tag $TAG --processor skimmer --submit --yaml src/condor/submit_configs/skimmer_inputs_07_24.yaml &> tmp/submitout.txt &

# Submit (if not using the --submit flag)
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

Local testing:

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



## Misc

Command for copying directories to PRP in background ([krsync](https://serverfault.com/a/887402))
(you may need to install `rsync` in the PRP pod first if it's been restarted)
```bash
cd ~/eos/bbVV/input/${TAG}_${JETS}/2017
mkdir ../copy_logs
for i in *; do echo $i && sleep 3 && (nohup sh -c "krsync -av --progress --stats $i/root/ $HWWTAGGERDEP_POD:/hwwtaggervol/training/$FOLDER/$i" &> ../copy_logs/$i.txt &) done```
