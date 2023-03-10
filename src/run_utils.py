import warnings

# from distributed.diagnostics.plugin import WorkerPlugin
import json


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
    if processor == "trigger":
        samples = [f"SingleMu{year[:4]}"]

    # redirector = "root://cmsxrootd.fnal.gov//" if not coffea_casa else "root://xcache//"
    redirector = "root://cmseos.fnal.gov//" if not coffea_casa else "root://xcache//"

    with open(f"data/pfnanoindex_{year}.json", "r") as f:
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
                fnames = fnames[starti:] if endi < 0 else fnames[starti:endi]
                sample_fileset[f"{year}_{subsample}"] = [redirector + fname for fname in fnames]

            fileset = {**fileset, **sample_fileset}

    return fileset


def get_xsecs():
    with open("data/xsecs.json") as f:
        xsecs = json.load(f)

    for key, value in xsecs.items():
        if type(value) == str:
            xsecs[key] = eval(value)

    return xsecs


def get_processor(
    processor: str,
    save_ak15: bool = None,
    label: str = None,
    njets: int = None,
    save_systematics: bool = None,
    inference: bool = None,
):
    # define processor
    if processor == "trigger":
        from HHbbVV.processors import JetHTTriggerEfficienciesProcessor

        return JetHTTriggerEfficienciesProcessor()
    elif processor == "skimmer":
        from HHbbVV.processors import bbVVSkimmer

        return bbVVSkimmer(
            xsecs=get_xsecs(),
            save_ak15=save_ak15,
            save_systematics=save_systematics,
            inference=inference,
        )
    elif processor == "input":
        from HHbbVV.processors import TaggerInputSkimmer

        return TaggerInputSkimmer(label, njets)
    elif processor == "ttsfs":
        from HHbbVV.processors import TTScaleFactorsSkimmer

        return TTScaleFactorsSkimmer(xsecs=get_xsecs())
    elif processor == "xhy":
        from HHbbVV.processors import XHYProcessor

        return XHYProcessor()


def parse_common_args(parser):
    parser.add_argument(
        "--processor",
        default="trigger",
        help="Trigger processor",
        type=str,
        choices=["trigger", "skimmer", "input", "ttsfs", "xhy"],
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

    parser.add_argument("--maxchunks", default=0, help="max chunks", type=int)
    parser.add_argument("--chunksize", default=10000, help="chunk size", type=int)
    parser.add_argument("--label", default="AK15_H_VV", help="label", type=str)
    parser.add_argument("--njets", default=2, help="njets", type=int)

    add_bool_arg(parser, "save-ak15", default=False, help="run inference for and save ak15 jets")
    add_bool_arg(parser, "save-systematics", default=False, help="save systematic variations")
    add_bool_arg(parser, "inference", default=True, help="run inference for ak8 jets")
