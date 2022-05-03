import os
from os import listdir
import sys
import numpy as np

tag = sys.argv[1]
eosdir = f"/eos/uscms/store/user/rkansal/bbVV/skimmer/{tag}/2017/"

samples = listdir(eosdir)
jdls = [jdl for jdl in listdir(f"condor/{tag}/") if jdl.endswith(".jdl")]

jdl_dict = {sample: np.sort([int(jdl[:-4].split('_')[-1]) for jdl in jdls if sample in jdl])[-1] for sample in samples} 

for sample in samples:
    outs = [int(out.split('.')[0].split('_')[-1]) for out in listdir(f"{eosdir}/{sample}/parquet")]
    print(sample)
    print(outs)

    for i in range(jdl_dict[sample]):
        if i not in outs:
            print(i)
            os.system(f"condor_submit condor/{tag}/2017_{sample}_{i}.jdl")

