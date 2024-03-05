from __future__ import annotations

import json
import subprocess

eosbase = "root://cmseos.fnal.gov/"
eosdirs = {
    "pfnanov1": "/store/user/lpchbb/cmantill/v2_2/2017v1/",
}

sampledict = {
    "HWW": {},
    "TTbar": {},
}


def eos_rec_search(startdir, suffix, dirs):
    dirlook = (
        subprocess.check_output(f"eos {eosbase} ls {startdir}", shell=True)
        .decode("utf-8")
        .split("\n")[:-1]
    )
    donedirs = [[] for d in dirlook]
    for di, d in enumerate(dirlook):
        if d.endswith(suffix):
            donedirs[di].append(startdir + "/" + d)
        elif d == "log":
            continue
        else:
            # print(f"Searching {d}")
            donedirs[di] = donedirs[di] + eos_rec_search(
                startdir + "/" + d, suffix, dirs + donedirs[di]
            )
    donedir = [d for da in donedirs for d in da]
    return dirs + donedir


for tag, eospfnano in eosdirs.items():
    try:
        for sample in sampledict.keys():
            datasets = (
                subprocess.check_output(f"eos {eosbase} ls {eospfnano}/{sample}/", shell=True)
                .decode("utf-8")
                .split("\n")[:-1]
            )
            for dataset in datasets:
                curdir = f"{eospfnano}/{sample}/{dataset}"
                dirlog = eos_rec_search(curdir, ".root", [])
                if dataset not in sampledict[sample].keys():
                    sampledict[sample][dataset] = dirlog
                else:
                    print(f"repeated {sample}/{dataset} in {user}")
                    sampledict[sample][dataset] = sampledict[sample][dataset] + dirlog
                    # print(user,sample,dataset,len(dirlog))
    except:
        pass
with open(f"{tag}_2017.json", "w") as outfile:
    json.dump(sampledict, outfile, indent=4, sort_keys=True)
