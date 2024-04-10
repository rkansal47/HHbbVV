# from distributed.diagnostics.plugin import WorkerPlugin
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from string import Template

from colorama import Fore, Style


def add_bool_arg(parser, name, help, default=False, no_name=None):
    """Add a boolean command line argument for argparse"""
    varname = "_".join(name.split("-"))  # change hyphens to underscores
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=varname, action="store_true", help=help)
    if no_name is None:
        no_name = "no-" + name
        no_help = "don't " + help
    else:
        no_help = help
    group.add_argument("--" + no_name, dest=varname, action="store_false", help=no_help)
    parser.set_defaults(**{varname: default})


def write_template(templ_file: str, out_file: Path, templ_args: dict, safe: bool = False):
    """Write to ``out_file`` based on template from ``templ_file`` using ``templ_args``"""

    with Path(templ_file).open() as f:
        templ = Template(f.read())

    with Path(out_file).open("w") as f:
        if not safe:
            f.write(templ.substitute(templ_args))
        else:
            f.write(templ.safe_substitute(templ_args))


def print_red(s):
    return print(f"{Fore.RED}{s}{Style.RESET_ALL}")


def add_mixins(nanoevents):
    # for running on condor
    nanoevents.PFNanoAODSchema.nested_index_items["FatJetAK15_pFCandsIdxG"] = (
        "FatJetAK15_nConstituents",
        "JetPFCandsAK15",
    )
    nanoevents.PFNanoAODSchema.mixins["FatJetAK15"] = "FatJet"
    nanoevents.PFNanoAODSchema.mixins["FatJetAK15SubJet"] = "FatJet"
    nanoevents.PFNanoAODSchema.mixins["SubJet"] = "FatJet"
    nanoevents.PFNanoAODSchema.mixins["PFCands"] = "PFCand"
    nanoevents.PFNanoAODSchema.mixins["SV"] = "PFCand"


def check_branch(git_branch: str, allow_diff_local_repo: bool = False):
    """Check that specified git branch exists in the repo, and local repo is up-to-date"""
    assert not bool(
        os.system(
            f'git ls-remote --exit-code --heads "https://github.com/rkansal47/HHbbVV" "{git_branch}"'
        )
    ), f"Branch {git_branch} does not exist"

    print(f"Using branch {git_branch}")

    # check if there are uncommitted changes
    uncommited_files = int(subprocess.getoutput("git status -s | wc -l"))

    if uncommited_files:
        print_red("There are local changes that have not been committed!")
        os.system("git status -s")
        if allow_diff_local_repo:
            print_red("Proceeding anyway...")
        else:
            print_red("Exiting! Use the --allow-diff-local-repo option to override this.")
            sys.exit(1)

    # check that the local repo's latest commit matches that on github
    remote_hash = subprocess.getoutput(f"git show origin/{git_branch} | head -n 1").split(" ")[1]
    local_hash = subprocess.getoutput("git rev-parse HEAD")

    if remote_hash != local_hash:
        print_red("Latest local and github commits do not match! Did you push your changes?")
        print(f"Local commit hash: {local_hash}")
        print(f"Remote commit hash: {remote_hash}")
        if allow_diff_local_repo:
            print_red("Proceeding anyway...")
        else:
            print_red("Exiting! Use the --allow-diff-local-repo option to override this.")
            sys.exit(1)


# for Dask executor
# class NanoeventsSchemaPlugin(WorkerPlugin):
#     def __init__(self):
#         pass

#     def setup(self, worker):
#         from coffea import nanoevents

#         add_mixins(nanoevents)


def get_fileset(
    processor: str,
    year: int,
    samples: list,
    subsamples: list,
    starti: int = 0,
    endi: int = -1,
    get_num_files: bool = False,
    coffea_casa: str = False,
):
    if processor.startswith("trigger"):
        samples = [f"SingleMu{year[:4]}"]

    redirector = "root://cmseos.fnal.gov//" if not coffea_casa else "root://xcache//"

    with Path(f"data/pfnanoindex_{year}.json").open() as f:
        full_fileset_pfnano = json.load(f)

    fileset = {}

    for sample in samples:
        sample_set = full_fileset_pfnano[year][sample]

        set_subsamples = list(sample_set.keys())

        # check if any subsamples for this sample have been specified
        get_subsamples = set(set_subsamples).intersection(subsamples)

        # if so keep only that subset
        if len(get_subsamples):
            sample_set = {subsample: sample_set[subsample] for subsample in get_subsamples}

        if get_num_files:
            # return only the number of files per subsample (for splitting up jobs)
            fileset[sample] = {}
            for subsample, fnames in sample_set.items():
                fileset[sample][subsample] = len(fnames)

        else:
            # return all files per subsample
            sample_fileset = {}

            for subsample, fnames in sample_set.items():
                run_fnames = fnames[starti:] if endi < 0 else fnames[starti:endi]
                sample_fileset[f"{year}_{subsample}"] = [redirector + fname for fname in run_fnames]

            fileset = {**fileset, **sample_fileset}

    return fileset


def get_xsecs():
    with Path("data/xsecs.json").open() as f:
        xsecs = json.load(f)

    for key, value in xsecs.items():
        if isinstance(value, str):
            xsecs[key] = eval(value)

    return xsecs


def get_processor(
    processor: str,
    save_ak15: bool = None,
    label: str = None,
    njets: int = None,
    save_systematics: bool = None,
    inference: bool = None,
    save_all: bool = None,
    lp_sfs: bool = None,
):
    # define processor
    if processor == "trigger":
        from HHbbVV.processors import JetHTTriggerEfficienciesProcessor

        return JetHTTriggerEfficienciesProcessor()
    elif processor == "trigger4d":
        from HHbbVV.processors import JetHT4DTriggerEfficienciesProcessor

        return JetHT4DTriggerEfficienciesProcessor()
    elif processor == "skimmer":
        from HHbbVV.processors import bbVVSkimmer

        return bbVVSkimmer(
            xsecs=get_xsecs(),
            save_ak15=save_ak15,
            save_systematics=save_systematics,
            inference=inference,
            save_all=save_all,
            lp_sfs=lp_sfs,
        )
    elif processor == "input":
        from HHbbVV.processors import TaggerInputSkimmer

        return TaggerInputSkimmer(label, njets)
    elif processor == "ttsfs":
        from HHbbVV.processors import TTScaleFactorsSkimmer

        return TTScaleFactorsSkimmer(xsecs=get_xsecs(), inference=inference)
    elif processor == "xhy":
        from HHbbVV.processors import XHYProcessor

        return XHYProcessor()


def parse_common_args(parser):
    parser.add_argument(
        "--processor",
        required=True,
        help="Which processor to run",
        type=str,
        choices=["trigger", "trigger4d", "skimmer", "input", "ttsfs", "xhy"],
    )

    parser.add_argument(
        "--year", help="year", type=str, required=True, choices=["2016APV", "2016", "2017", "2018"]
    )

    parser.add_argument(
        "--samples",
        default=[],
        help="which samples to run",  # , default will be all samples",
        nargs="*",
    )
    parser.add_argument(
        "--subsamples",
        default=[],
        help="which subsamples, by default will be all in the specified sample(s)",
        nargs="*",
    )

    parser.add_argument("--tag", default="Test", help="process tag", type=str)

    parser.add_argument("--maxchunks", default=0, help="max chunks", type=int)
    parser.add_argument("--chunksize", default=10000, help="chunk size", type=int)
    parser.add_argument("--label", default="AK15_H_VV", help="label", type=str)
    parser.add_argument("--njets", default=2, help="njets", type=int)

    # REMEMBER TO PROPAGATE THESE TO SUBMIT TEMPLATE!!
    # processor args
    add_bool_arg(parser, "inference", default=True, help="run inference for ak8 jets")
    # bbVVSkimmer-only args
    add_bool_arg(parser, "save-ak15", default=False, help="run inference for and save ak15 jets")
    add_bool_arg(parser, "save-systematics", default=True, help="save systematic variations")
    add_bool_arg(parser, "save-all", default=False, help="save all branches")
    add_bool_arg(parser, "lp-sfs", default=True, help="run LP SFs for signals")
