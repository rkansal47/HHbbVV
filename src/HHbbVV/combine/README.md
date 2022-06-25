# combine scripts

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
