import subprocess
import json


def get_children(parent):
    # print(f"DEBUG : Call to get_children({parent})")
    command = f"eos root://cmseos.fnal.gov ls -F {parent}"
    # print(command)
    result = subprocess.getoutput(command)  # , stdout=subprocess.PIPE)
    # print(result)
    return result.split("\n")


def get_subfolders(parent):
    subfolders = []
    for x in get_children(parent):
        if len(x) == 0:
            continue
        if x[-1] == "/":
            subfolders.append(x)
    return subfolders


folders_to_index = [
    "/store/user/lpcpfnano/jekrupa/v2_2/2016/JetHT2016",
    "/store/user/lpcpfnano/jekrupa/v2_2/2017/JetHT2017",
    "/store/user/lpcpfnano/jekrupa/v2_2/2018/JetHT2018",
    "/store/user/lpcpfnano/jekrupa/v2_2/2016APV/TTbar",
    "/store/user/lpcpfnano/jekrupa/v2_2/2016/TTbar",
    "/store/user/lpcpfnano/jekrupa/v2_2/2017/TTbar",
    "/store/user/lpcpfnano/emoreno/v2_2/2018/TTbar",
    "/store/user/lpcpfnano/drankin/v2_2/2016/TTbar",
    "/store/user/lpcpfnano/drankin/v2_2/2016APV/TTbar",
    "/store/user/lpcpfnano/drankin/v2_2/2017/TTbar",
    "/store/user/lpcpfnano/drankin/v2_2/2018/TTbar",
    "/store/user/lpcpfnano/yihan/v2_2/2016/TTbar",
    "/store/user/lpcpfnano/yihan/v2_2/2016APV/TTbar",
    "/store/user/lpcpfnano/yihan/v2_2/2017/TTbar",
    "/store/user/lpcpfnano/yihan/v2_2/2018/TTbar",
    "/store/user/lpcpfnano/yihan/v2_2/2016/QCD",
    "/store/user/lpcpfnano/yihan/v2_2/2016APV/QCD",
    "/store/user/lpcpfnano/jekrupa/v2_2/2017/QCD",
    "/store/user/lpchbb/cmantill/v2_2/2017v1/QCDHerwig",
    "/store/user/lpcpfnano/jekrupa/v2_2/2018/QCD",
    "/store/user/lpcpfnano/jekrupa/v2_2/2016/WJetsToQQ",
    "/store/user/lpcpfnano/jekrupa/v2_2/2016APV/WJetsToQQ",
    "/store/user/lpcpfnano/jekrupa/v2_2/2017/WJetsToQQ",
    "/store/user/lpcpfnano/jekrupa/v2_2/2018/WJetsToQQ",
    "/store/user/lpcpfnano/jekrupa/v2_2/2016/ZJetsToQQ",
    "/store/user/lpcpfnano/jekrupa/v2_2/2016APV/ZJetsToQQ",
    "/store/user/lpcpfnano/jekrupa/v2_2/2017/ZJetsToQQ",
    "/store/user/lpcpfnano/jekrupa/v2_2/2018/ZJetsToQQ",
    "/store/user/lpcpfnano/pharris/v2_2/2016/SingleTop",
    "/store/user/lpcpfnano/pharris/v2_2/2016APV/SingleTop",
    "/store/user/lpcpfnano/pharris/v2_2/2017/SingleTop",
    "/store/user/lpcpfnano/pharris/v2_2/2018/SingleTop",
    "/store/user/lpcpfnano/drankin/v2_2/2016/SingleTop/",
    "/store/user/lpcpfnano/drankin/v2_2/2016APV/SingleTop/",
    "/store/user/lpcpfnano/drankin/v2_2/2017/SingleTop/",
    "/store/user/lpcpfnano/drankin/v2_2/2018/SingleTop/",
    "/store/user/lpcpfnano/drankin/v2_2/2016/WJetsToLNu",
    "/store/user/lpcpfnano/drankin/v2_2/2016APV/WJetsToLNu",
    "/store/user/lpcpfnano/drankin/v2_2/2017/WJetsToLNu",
    "/store/user/lpcpfnano/drankin/v2_2/2018/WJetsToLNu",
    "/store/user/lpcpfnano/cmantill/v2_2/2016/DYJetsToLL",
    "/store/user/lpcpfnano/cmantill/v2_2/2016APV/DYJetsToLL",
    "/store/user/lpcpfnano/cmantill/v2_2/2017/DYJetsToLL",
    "/store/user/lpcpfnano/cmantill/v2_2/2018/DYJetsToLL",
    "/store/user/lpcpfnano/jdickins/v2_2/2016/Diboson/",
    "/store/user/lpcpfnano/jdickins/v2_2/2017/Diboson/",
    "/store/user/lpcpfnano/jdickins/v2_2/2018/Diboson/",
    "/store/user/lpcpfnano/jdickins/v2_2/2016/ttHToBB/"
    "/store/user/lpcpfnano/jdickins/v2_2/2016APV/ttHToBB/",
    "/store/user/lpcpfnano/jdickins/v2_2/2017/ttHToBB/",
    "/store/user/lpcpfnano/jdickins/v2_2/2018/ttHToBB/",
    "/store/user/lpcpfnano/jdickins/v2_2/2016/VBFHToBB/",
    "/store/user/lpcpfnano/jdickins/v2_2/2016APV/VBFHToBB/",
    "/store/user/lpcpfnano/jdickins/v2_2/2017/VBFHToBB/",
    "/store/user/lpcpfnano/jdickins/v2_2/2018/VBFHToBB/",
    "/store/user/lpcpfnano/jdickins/v2_2/2016/GluGluHToBB/",
    "/store/user/lpcpfnano/jdickins/v2_2/2016APV/GluGluHToBB/",
    "/store/user/lpcpfnano/jdickins/v2_2/2017/GluGluHToBB/",
    "/store/user/lpcpfnano/jdickins/v2_2/2018/GluGluHToBB/",
    "/store/user/lpcpfnano/jdickins/v2_2/2016/WHToBB/",
    "/store/user/lpcpfnano/jdickins/v2_2/2016APV/WHToBB/",
    "/store/user/lpcpfnano/jdickins/v2_2/2017/WHToBB/",
    "/store/user/lpcpfnano/jdickins/v2_2/2018/WHToBB/",
    "/store/user/lpcpfnano/jdickins/v2_2/2016/ZHToBB/",
    "/store/user/lpcpfnano/jdickins/v2_2/2016APV/ZHToBB/",
    "/store/user/lpcpfnano/jdickins/v2_2/2017/ZHToBB/",
    "/store/user/lpcpfnano/jdickins/v2_2/2018/ZHToBB/",
    "/store/user/lpcpfnano/cmantill/v2_2/2016/HWW",
    "/store/user/lpcpfnano/cmantill/v2_2/2016APV/HWW",
    "/store/user/lpcpfnano/cmantill/v2_2/2017/HWW",
    "/store/user/lpcpfnano/cmantill/v2_2/2017/HWWPrivate",
    "/store/user/lpcpfnano/cmantill/v2_2/2018/HWW",
    "/store/user/lpcpfnano/cmantill/v2_3/2016/HWW",
    "/store/user/lpcpfnano/cmantill/v2_3/2016APV/HWW",
    "/store/user/lpcpfnano/cmantill/v2_3/2017/HWW",
    "/store/user/lpcpfnano/cmantill/v2_3/2018/HWW",
]

index_APV = {}

# Data path:
# .......................f1........................|...f2.....|..........f3.......|.....f4......|.f5.|....
# /store/user/lpcpfnano/dryu/v2_2/2017/SingleMu2017/SingleMuon/SingleMuon_Run2017C/211102_162942/0000/*root
#
# MC path:
# .......................f1........................|.......................f2..............................|..........f3.........|.....f4......|.f5.|....
# /store/user/lpcpfnano/jekrupa/v2_2/2017/WJetsToQQ/WJetsToQQ_HT-800toInf_TuneCP5_13TeV-madgraphMLM-pythia8/WJetsToQQ_HT-800toInf/211108_171840/0000/*root

for pyear in ["2016", "2016APV", "2017", "2018"]:
    # for pyear in ["2017"]:
    print(pyear)
    index = {}
    for f1 in folders_to_index:
        f1 = f1.rstrip("/")
        # print(f1)
        year = f1.split("/")[-2]
        sample_short = f1.split("/")[-1]
        if year == "2017v1":
            year = "2017"
        if year != pyear:
            continue

        sample_short = f1.split("/")[-1]
        print(f" {sample_short}")

        if not year in index:
            index[year] = {}
        if not sample_short in index[year]:
            index[year][sample_short] = {}

        f1_subfolders = get_subfolders(f"{f1}")
        for f2 in f1_subfolders:
            # print(f"\t/{f2}")
            subsample_long = f2.replace("/", "")  # This should be the actual dataset name
            f2_subfolders = get_subfolders(f"{f1}/{f2}")
            if len(f2_subfolders) == 0:
                root_files = [
                    f"{f1}/{f2}/{x}".replace("//", "/")
                    for x in get_children((f"{f1}/{f2}"))
                    if x[-5:] == ".root"
                ]
                if not subsample_long in index[year][sample_short]:
                    index[year][sample_short][subsample_long] = []
                index[year][sample_short][subsample_long].extend(root_files)

            for f3 in f2_subfolders:
                # print(f"\t\t/{f3}")
                subsample_short = f3.replace("/", "")
                if "ext1" in subsample_short:
                    print("   Ext1")

                subsample_short = subsample_short.replace("_ext1", "")
                print(f"  {subsample_short}")

                if not subsample_short in index[year][sample_short]:
                    index[year][sample_short][subsample_short] = []
                f3_subfolders = get_subfolders(f"{f1}/{f2}/{f3}")
                if len(f3_subfolders) >= 2:
                    print(f"WARNING : Found multiple timestamps for {f1}/{f2}/{f3}")

                for f4 in f3_subfolders:  # Timestamp
                    f4_subfolders = get_subfolders(f"{f1}/{f2}/{f3}/{f4}")

                    for f5 in f4_subfolders:  # 0000, 0001, ...
                        f5_children = get_children((f"{f1}/{f2}/{f3}/{f4}/{f5}"))
                        root_files = [
                            f"{f1}/{f2}/{f3}/{f4}/{f5}/{x}".replace("//", "/")
                            for x in f5_children
                            if x[-5:] == ".root"
                        ]
                        if year == "2016" and "HIPM" in subsample_short:
                            if not sample_short in index_APV:
                                index_APV[sample_short] = {}
                            if not subsample_short in index_APV[sample_short]:
                                index_APV[sample_short][subsample_short] = []
                                index_APV[sample_short][subsample_short].extend(root_files)
                        else:
                            if not subsample_short in index[year][sample_short]:
                                index[year][sample_short][subsample_short] = []
                            index[year][sample_short][subsample_short].extend(root_files)

    if pyear == "2016APV":
        for sample_short in index_APV.keys():
            for subsample_short in index_APV[sample_short].keys():
                if not sample_short in index[pyear]:
                    index[pyear][sample_short] = {}
                if not subsample_short in index[pyear][sample_short]:
                    index[pyear][sample_short][subsample_short] = []
                index[pyear][sample_short][subsample_short] = index_APV[sample_short][
                    subsample_short
                ]

    with open(f"pfnanoindex_{pyear}.json", "w") as f:
        json.dump(index, f, sort_keys=True, indent=2)
