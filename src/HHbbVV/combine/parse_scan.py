"""
Scrapes through the scan folders to find the significances.

Author: Raghav Kansal
"""

from os import scandir
from os.path import exists
import pandas as pd

scan_dir = '../cards/04_07_scan/'

cut_dirs = [f.path for f in scandir(scan_dir) if f.is_dir()]

sigs = []

for cut_dir in cut_dirs:
    sig_file = f"{cut_dir}/significance.txt"
    lim_file = f"{cut_dir}/asymptoticlimits.txt"

    if not exists(sig_file):
        print(f"{cut_dir} Doesn't exist")
        continue

    split = cut_dir.split('_')
    bdt_cut = split[-3]
    bb_cut = split[-1]

    with open(sig_file, 'r') as f:
        try:
            sig = float(f.readlines()[-2].split(' ')[-1])
        except:
            print(f"{cut_dir} Error parsing significance")
            continue

    with open(lim_file, 'r') as f:
        try:
            lim = float(f.readlines()[-5].split(' ')[-1])
        except:
            print(f"{cut_dir} Error parsing limits")
            continue

    sigs.append([bdt_cut, bb_cut, sig, lim])

print(len(sigs))

sig_table = pd.DataFrame(sigs, columns=['BDT Cut', 'Txbb Cut', 'Sign.', 'Limit'])
sig_table.sort_values(['BDT Cut', 'Txbb Cut'], inplace=True)
sig_table.to_csv(f"{scan_dir}/signs.csv", index=False)