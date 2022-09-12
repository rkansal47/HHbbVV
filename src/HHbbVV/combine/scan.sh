#!/bin/bash

cutstrs=($(ls -d */))

for cutstr in ${cutstrs[@]}
do
  echo $cutstr
  cd $cutstr

  /uscms/home/rkansal/nobackup/HHbbVV/src/HHbbVV/combine/run_blinded.sh "./"
  
  cd ..
done
