"""
Filter skimmed data further and save in a new directory
"""

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from os import listdir
import os

eos_dir = "/eos/uscms/store/user/rkansal/bbVV/skimmer/Jul27"
new_dir = "/eos/uscms/store/user/rkansal/bbVV/skimmer/Jul27_TxbbCut"

year = 2017

# Both Jet's Msds > 50 & at least one jet with Txbb > 0.8
filters = [
    [
        ("('ak8FatJetParticleNetMD_Txbb', '0')", ">=", 0.8),
    ],
    [
        ("('ak8FatJetParticleNetMD_Txbb', '1')", ">=", 0.8),
    ],
]

for sample in listdir(f"{eos_dir}/{year}/"):
    if sample == "GluGluToHHTobbVV_node_cHHH1_pn4q":
        continue

    print(sample)

    _ = os.system(f"mkdir -p {new_dir}/{year}/{sample}/parquet")
    # _ = os.system(f"cp -r {eos_dir}/{year}/{sample}/pickles {new_dir}/{year}/{sample}/pickles")

    table = pa.Table.from_pandas(
        pd.read_parquet(f"{eos_dir}/{year}/{sample}/parquet", filters=filters)
    )

    print("Writing parquet")

    pq.write_table(table, f"{new_dir}/{year}/{sample}/parquet/all.parquet")
