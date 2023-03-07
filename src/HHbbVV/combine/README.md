# Combine Scripts

- [Combine Scripts](#combine-scripts)
  - [CMSSW + Combine Quickstart](#cmssw--combine-quickstart)
  - [Run basic fits and diagnostics](#run-basic-fits-and-diagnostics)


## CMSSW + Combine Quickstart
```bash
cmsrel CMSSW_11_2_0
cd CMSSW_11_2_0/src
cmsenv
git clone -b py3 https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit
scramv1 b clean; scramv1 b

cd $CMSSW_BASE/src/
git clone -b 113x https://github.com/cms-analysis/CombineHarvester.git CombineHarvester
scram b
```

## Run basic fits and diagnostics

```bash
/uscms/home/rkansal/nobackup/HHbbVV/src/HHbbVV/combine/run_blinded.sh "./"
```

## Get data and toy test statistics for simple GoF

```bash
/uscms/home/rkansal/nobackup/HHbbVV/src/HHbbVV/combine/gof.sh "./"
```