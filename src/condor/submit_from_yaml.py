"""
Runs the submit script but with samples specified in a yaml file.

Author(s): Raghav Kansal, Cristina Mantilla Suarez
"""

import os
import argparse

import submit
import yaml


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", help="year", type=str, required=True)
    parser.add_argument("--tag", default="Test", help="process tag", type=str)
    parser.add_argument("--jet", default="AK8", help="jet", type=str)
    parser.add_argument(
        "--submit", dest="submit", action="store_true", help="submit jobs when created"
    )
    parser.add_argument(
        "--site",
        default="lpc",
        help="computing cluster we're running this on",
        type=str,
        choices=["lpc", "ucsd"],
    )
    parser.add_argument(
        "--processor",
        default="trigger",
        help="which processor",
        type=str,
        choices=["trigger", "skimmer", "input", "ttsfs"],
    )
    parser.add_argument("--yaml", default="", help="yaml file", type=str)
    add_bool_arg(parser, "save-ak15", default=False, help="run inference for and save ak15 jets")
    add_bool_arg(parser, "save-systematics", default=False, help="save systematic variations")
    add_bool_arg(parser, "inference", default=True, help="run inference for ak8 jets")

    args = parser.parse_args()

    with open(args.yaml, "r") as file:
        samples_to_submit = yaml.safe_load(file)

    args.script = "run.py"
    args.outdir = "outfiles"
    args.test = False
    tag = args.tag
    for key, tdict in samples_to_submit.items():
        for sample, sdict in tdict.items():
            if sample in ["JetHT", "SingleMu"]:
                sample += args.year[:4]

            args.samples = [sample]
            args.subsamples = sdict.get("subsamples", [])
            args.files_per_job = sdict["files_per_job"]
            args.njets = sdict.get("njets", 2)
            args.maxchunks = sdict.get("maxchunks", 0)
            args.chunksize = sdict.get("chunksize", 1000)

            args.tag = tag

            args.label = args.jet + sdict["label"] if "label" in sdict.keys() else "None"

            if key == "Validation":
                args.tag = f"{args.tag}_Validation"

            print(args)
            submit.main(args)
