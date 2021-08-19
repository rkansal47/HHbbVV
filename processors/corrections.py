import os
import numpy as np
import gzip
import pickle
import importlib.resources

with importlib.resources.path("HHbbVV.data", "corrections.pkl.gz") as path:
    with gzip.open(path) as fin:
        compiled = pickle.load(fin)

def add_pileup_weight(weights, nPU, year='2017', dataset=None):
    weights.add(
        'pileup_weight',
        compiled[f'{year}_pileupweight'](nPU),
        compiled[f'{year}_pileupweight_puUp'](nPU),
        compiled[f'{year}_pileupweight_puDown'](nPU),
    )
