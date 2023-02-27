from collections import OrderedDict
from typing import Dict
import utils
import plotting
import postprocessing
import numpy as np
import warnings
import pandas as pd
from pandas.errors import SettingWithCopyWarning
from hh_vars import samples, sig_key, data_key, jec_shifts, jmsr_shifts, jec_vars, jmsr_vars
import os, sys

# ignore these because they don't seem to apply
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from copy import deepcopy


BDT_samples = list(samples.keys())


BDT_data_vars = [
    "MET_pt",
    "DijetEta",
    "DijetPt",
    "DijetMass",
    "bbFatJetPt",
    "VVFatJetEta",
    "VVFatJetPt",
    "VVFatJetParticleNetMass",
    "VVFatJetParTMD_THWW4q",
    "VVFatJetParTMD_probQCD",
    "VVFatJetParTMD_probHWW3q",
    "VVFatJetParTMD_probHWW4q",
    "VVFatJetParTMD_probT",
    "bbFatJetParticleNetMD_Txbb",
    "bbFatJetPtOverDijetPt",
    "VVFatJetPtOverDijetPt",
    "VVFatJetPtOverbbFatJetPt",
    "finalWeight",
]


def main(args):
    if not (args.control_plots or args.bdt_data):
        print("No option selected")
        sys.exit()

    # make plot, template dirs if needed
    _make_dirs(args)

    # save cutflow as pandas table
    cutflow = pd.DataFrame(index=list(samples.keys()))
    systematics = {}

    events_dict = utils.load_samples(
        args.data_dir, samples, args.year, postprocessing.filters if args.filters else None
    )
    utils.add_to_cutflow(events_dict, "BDTPreselection", "weight", cutflow)

    # print weighted sample yields
    print("")
    for sample in events_dict:
        tot_weight = np.sum(events_dict[sample]["weight"].values)
        print(f"Pre-selection {sample} yield: {tot_weight:.2f}")

    postprocessing.apply_weights(events_dict, args.year, cutflow)
    bb_masks = postprocessing.bb_VV_assignment(events_dict)
    _ = postprocessing.postprocess_lpsfs(events_dict[sig_key], save_all=False)
    print("\nCutflow:\n", cutflow)

    control_plot_vars = postprocessing.control_plot_vars
    del control_plot_vars["BDTScore"]

    if args.control_plots:
        print("\nMaking control plots")
        postprocessing.control_plots(
            events_dict,
            bb_masks,
            control_plot_vars,
            f"{args.plot_dir}/{args.year}",
            args.year,
        )
        print("Made and saved control plots")

    if args.bdt_data:
        print("\nSaving BDT Data")
        save_bdt_data(
            events_dict, bb_masks, f"{args.data_dir}/bdt_data/{args.year}_bdt_data.parquet"
        )
        print("Saved BDT Data")


def _make_dirs(args):
    if args.plot_dir != "" and args.control_plots:
        os.system(f"mkdir -p {args.plot_dir}/{args.year}/")

    if args.bdt_data:
        os.system(f"mkdir -p {args.data_dir}/bdt_data/")


def save_bdt_data(
    events_dict: Dict[str, pd.DataFrame],
    bb_masks: Dict[str, pd.DataFrame],
    out_file: str,
):
    import pyarrow.parquet as pq
    import pyarrow as pa

    jec_jmsr_vars = []

    for var in BDT_data_vars:
        if var in jec_vars:
            for jshift in jec_shifts:
                jec_jmsr_vars.append(f"{var}_{jshift}")

        if var in jmsr_vars:
            for jshift in jmsr_shifts:
                jec_jmsr_vars.append(f"{var}_{jshift}")

    bdt_events_dict = []

    for sample in BDT_samples:
        save_vars = BDT_data_vars + jec_jmsr_vars if sample != "Data" else BDT_data_vars
        events = pd.DataFrame(
            {var: utils.get_feat(events_dict[sample], var, bb_masks[sample]) for var in save_vars}
        )
        events["Dataset"] = sample
        bdt_events_dict.append(events)

    bdt_events = pd.concat(bdt_events_dict, axis=0)

    table = pa.Table.from_pandas(bdt_events)
    pq.write_table(table, out_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        default="../../../../data/skimmer/Feb24/",
        help="path to skimmed parquet",
        type=str,
    )

    parser.add_argument(
        "--year",
        default="2017",
        choices=["2016", "2016APV", "2017", "2018"],
        type=str,
    )

    parser.add_argument(
        "--plot-dir",
        help="If making control or template plots, path to directory to save them in",
        default="",
        type=str,
    )

    utils.add_bool_arg(parser, "control-plots", "make control plots", default=False)
    utils.add_bool_arg(parser, "bdt-data", "save bdt training data", default=False)
    utils.add_bool_arg(
        parser, "filters", "use pre-selection filters when loading samples", default=True
    )

    args = parser.parse_args()
    main(args)
