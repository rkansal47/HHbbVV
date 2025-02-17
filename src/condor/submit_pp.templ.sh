#!/bin/bash
# shellcheck disable=SC2154,SC2086,SC2034,SC1036,SC1088,SC1083

# make sure this is installed
# python3 -m pip install correctionlib==2.0.0rc6
# pip install --upgrade numpy==1.21.5

# make dir for output
mkdir outfiles

for t2_prefix in ${t2_prefixes}
do
    xrdfs $${t2_prefix} mkdir -p -mrwxr-xr-x "/${outdir}"
done


# try 3 times in case of network errors
(
    r=3
    # shallow clone of single branch (keep repo size as small as possible)
    while ! git clone --single-branch --branch $branch --depth=1 https://github.com/$gituser/HHbbVV/
    do
        ((--r)) || exit
        sleep 60
    done
)
cd HHbbVV || exit

commithash=$$(git rev-parse HEAD)
echo "https://github.com/rkansal47/HHbbVV/commit/$${commithash}" > commithash.txt

#move output to t2s
for t2_prefix in ${t2_prefixes}
do
    xrdcp -f commithash.txt $${t2_prefix}/${outdir}/commithash_${jobnum}.txt
done


pip3 install -e .
cd src/HHbbVV/postprocessing/ || exit

# make dir for output
mkdir condor_templates

# run code
for sample in $samples
do
    for year in 2016 2016APV 2017 2018
    do
        mkdir -p samples/$${year}
        xrdcp -r root://cmseos.fnal.gov//store/user/rkansal/bbVV/skimmer/25Feb6XHY/$${year}/$${sample} samples/$${year}
    done

    ls -lh samples
    ls -lh samples/*

    for year in 2016 2016APV 2017 2018
    do
        python -u postprocessing.py --templates --resonant --no-data --bg-keys "" --year "$${year}" \
        --sig-samples "$${sample}" --template-dir "condor_templates/$${sample}" \
        --data-dir "samples"

        rm -rf samples/$${year}/$${sample}
    done

    #move output to t2s
    for t2_prefix in ${t2_prefixes}
    do
        xrdcp -fr condor_templates/$${sample} $${t2_prefix}/${outdir}/
    done

    rm -rf condor_templates/$${sample}
done
