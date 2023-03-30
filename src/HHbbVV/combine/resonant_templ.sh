#!/bin/bash                                                                                                                   
echo "Starting job on " `date` #Date/time of start of job                                                                       
echo "Running on: `uname -a`" #Condor job is running on this node                                                               
echo "System software: `cat /etc/redhat-release`" #Operating System on that node                                             

# CMSSW                                                                                                                        
source /cvmfs/cms.cern.ch/cmsset_default.sh
scramv1 project CMSSW CMSSW_11_2_0 # cmsrel is an alias not on the workers                                                        
ls -alrth
cd CMSSW_11_2_0/src/
eval `scramv1 runtime -sh` # cmsenv is an alias not on the workers                                                          
echo $CMSSW_BASE "is the CMSSW we created on the local worker node"

# Combine                                                                                                                   
git clone -b py3 https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
# git clone -b v2.0.0 https://github.com/cms-analysis/CombineHarvester.git CombineHarvester
scramv1 b clean; scramv1 b

dataset=data_obs
cards_dir="./"
ws=${cards_dir}/combined
wsm=${ws}_withmasks
wsm_snapshot=higgsCombineSnapshot.MultiDimFit.mH125

# python3 postprocessing/CreateDatacard.py --templates-dir postprocessing/templates/$template --model-name $tag --resonant --year $year --no-bblite
# cd ${cards_dir}

# one channel per bin
ccargs=""
for bin in {0..18}
do
    for channel in fail failBlinded pass passBlinded;
    do
        ccargs+="mXbin${bin}${channel}=${cards_dir}/mXbin${bin}${channel}.txt "
        maskunblindedargs+=""
    done
done

# mask args
maskunblindedargs=""
maskblindedargs=""
for bin in {0..18}
do
    for channel in fail pass;
    do
        maskunblindedargs+="mask_mXbin${bin}${channel}=1,mask_mXbin${bin}${channel}Blinded=0,"
        maskblindedargs+="mask_mXbin${bin}${channel}=0,mask_mXbin${bin}${channel}Blinded=1,"
    done
done

# combineCards.py $ccargs > $ws.txt

echo "text2workspace"
text2workspace.py -D $dataset $ws.txt --channel-masks -o $wsm.root 2>&1 | tee outs/text2workspace.txt
xrdcp -f combined_withmasks.root root://cmseos.fnal.gov//store/user/rkansal/
xrdcp -f outs/text2workspace.txt root://cmseos.fnal.gov//store/user/rkansal/