#!/bin/bash
# shellcheck disable=SC2154,SC2086

# make sure this is installed
# python3 -m pip install correctionlib==2.0.0rc6
# pip install --upgrade numpy==1.21.5

# make dir for output
mkdir outfiles

# shallow clone of single branch (keep repo size as small as possible)
git clone --single-branch --branch $branch --depth=1 https://github.com/rkansal47/HHbbVV/
cd HHbbVV || exit

commithash=$(git rev-parse HEAD)
echo "https://github.com/rkansal47/HHbbVV/commit/${commithash}" > commithash.txt
xrdcp -f commithash.txt $eosoutgithash

pip install -e .

# run code
# pip install --user onnxruntime
python -u -W ignore $script --year $year --starti $starti --endi $endi --samples $sample --subsamples $subsample --processor $processor --maxchunks $maxchunks --chunksize $chunksize --label $label --njets $njets ${save_ak15} ${save_systematics} ${inference} ${save_all} ${lp_sfs}

#move output to eos
xrdcp -f outfiles/* $eosoutpkl
xrdcp -f ./*.parquet $eosoutparquet
xrdcp -f ./*.root $eosoutroot

rm ./*.parquet
rm ./*.root
rm commithash.txt
