from dask.distributed import Client

client = Client("tls://localhost:8786")
print(client)

# loading full analysis dataset
from os import listdir

# TODO: replace with UL sample once we have it
with open("data/2017_preUL_nano/HHToBBVVToBBQQQQ_cHHH1.txt", "r") as file:
    filelist = [
        f[:-1].replace("/eos/uscms/", "root://xcache//") for f in file.readlines()
    ]  # need to use xcache redirector at Nebraksa coffea-casa

fileset = {"2017_HHToBBVVToBBQQQQ_cHHH1": filelist}

ignore_samples = [
    "GluGluHToTauTau_M125_TuneCP5_13TeV-powheg-pythia8",
    "GluGluHToWWToLNuQQ_M125_TuneCP5_PSweight_13TeV-powheg2-jhugen727-pythia8",
    "ST_tW_antitop_5f_DS_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
    "ST_tW_top_5f_DS_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
]

for sample in listdir("data/2017_UL_nano/"):
    if sample[-4:] == ".txt" and sample[:-4] not in ignore_samples:
        with open(f"data/2017_UL_nano/{sample}", "r") as file:
            replace_string = "/hadoop/cms/" if "JetHT" in sample else "/eos/uscms/"
            filelist = [
                f[:-1].replace(replace_string, "root://xcache//") for f in file.readlines()
            ]  # need to use xcache redirector at Nebraksa coffea-casa
        fileset["2017_" + sample[:-4].split("_TuneCP5")[0]] = filelist

testing_fileset = {key: filelist[:2] for key, value in fileset.items()}

# loading cross sections
import json

with open("data/xsecs.json") as f:
    xsecs = json.load(f)

for key, value in xsecs.items():
    if type(value) == str:
        xsecs[key] = eval(value)


# check that we have xsecs for all samples
for key in fileset.keys():
    if "JetHT" not in key:
        dname = key.split("2017_")[1]
        if dname not in xsecs:
            import warnings

            warnings.warn(f"cross section not found for {dname}")


# run processor
from coffea import processor
from coffea.nanoevents import NanoAODSchema
import time

import processors

# need to upload processors to all nodes
import shutil

shutil.make_archive("processors", "zip", base_dir="processors")
client.upload_file("processors.zip")


tic = time.time()

exe_args = {
    "client": client,
    "savemetrics": True,
    "schema": NanoAODSchema,
    "align_clusters": True,
    "skipbadfiles": True,
}

print("Waiting for at least one worker...")
client.wait_for_workers(1)

print("Found at least one worker...")

out, metrics = processor.run_uproot_job(
    fileset,
    treename="Events",
    processor_instance=processors.bbVVSkimmer(xsecs),  # replace with your processor
    executor=processor.dask_executor,
    executor_args=exe_args,
    #     maxchunks=10
)

elapsed = time.time() - tic

print(f"Metrics: {metrics}")
print(f"Finished in {elapsed:.1f}s")


# finally, save output
import pickle
from os.path import exists

out_file = "outPickles/out_skimmed.pickle"  # make sure to change!!

# if not exists(out_file):
with open(out_file, "wb") as filehandler:
    pickle.dump(out, filehandler)
