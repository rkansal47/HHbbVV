#!/bin/bash

# make sure this is installed
# python3 -m pip install correctionlib==2.0.0rc6

# make dir for output
mkdir outfiles

# run code
python SCRIPTNAME --year YEAR --starti STARTNUM --endi ENDNUM --samples SAMPLE --processor PROCESSOR --condor

#move output to eos
xrdcp -f outfiles/* EOSOUT
