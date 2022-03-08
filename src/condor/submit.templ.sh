#!/bin/bash

# make sure this is installed
# python3 -m pip install correctionlib==2.0.0rc6
# pip install --upgrade numpy==1.21.5

# make dir for output
mkdir outfiles

# run code
# pip install --user onnxruntime
python -u -W ignore $script --year $year --starti $starti --endi $endi --samples $sample --processor $processor

#move output to eos
xrdcp -f outfiles/* $eosoutpkl
xrdcp -f *.parquet $eosoutparquet
