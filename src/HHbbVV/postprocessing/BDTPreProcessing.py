from collections import OrderedDict
from typing import Dict
import utils
import plotting
import postprocessing
import numpy as np
import warnings
import pandas as pd
from pandas.errors import SettingWithCopyWarning
from hh_vars import (
    samples,
    res_samples,
    nonres_samples,
    nonres_sig_keys,
    res_sig_keys,
    data_key,
    jec_shifts,
    jmsr_shifts,
    jec_vars,
    jmsr_vars,
)
import os, sys

# ignore these because they don't seem to apply
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from copy import deepcopy


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
    # make plot, template dirs if needed
    _make_dirs(args)

    BDT_sample_order = nonres_sig_keys + ["QCD", "TT", "ST", "V+Jets", "Diboson", "Data"]

    sig_keys, sig_samples, bg_keys, bg_samples = postprocessing._process_samples(
        args, BDT_sample_order
    )

    # save cutflow as pandas table
    all_samples = sig_keys + bg_keys
    cutflow = pd.DataFrame(index=all_samples)

    filters = postprocessing.new_filters
    events_dict = postprocessing._load_samples(args, bg_samples, sig_samples, cutflow, filters)
    postprocessing.apply_weights(events_dict, args.year, cutflow)
    bb_masks = postprocessing.bb_VV_assignment(events_dict)
    if args.control_plots:
        cutflow.to_csv(f"{args.plot_dir}/{args.year}/cutflow.csv")

    control_plot_vars = postprocessing.control_plot_vars
    del control_plot_vars["BDTScore"]

    if args.control_plots:
        # print("\nMaking control plots")
        postprocessing.control_plots(
            events_dict,
            bb_masks,
            control_plot_vars,
            f"{args.plot_dir}/{args.year}",
            args.year,
        )
        # print("Made and saved control plots")

    if args.bdt_data:
        # print("\nSaving BDT Data")
        data_dir = args.data_dir if args.data_dir else args.signal_data_dir
        save_bdt_data(events_dict, bb_masks, f"{data_dir}/bdt_data/{args.year}_bdt_data.parquet")
        # print("Saved BDT Data")


def _make_dirs(args):
    if args.plot_dir != "" and args.control_plots:
        os.system(f"mkdir -p {args.plot_dir}/{args.year}/")

    if args.bdt_data:
        if args.data_dir:
            os.system(f"mkdir -p {args.data_dir}/bdt_data/")
        else:
            if args.signal_data_dir:
                os.system(f"mkdir -p {args.signal_data_dir}/bdt_data/")


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
    bdt_sample_order = []
    for key in events_dict.keys():
        save_vars = BDT_data_vars + jec_jmsr_vars if key != "Data" else BDT_data_vars
        events = pd.DataFrame(
            {var: utils.get_feat(events_dict[key], var, bb_masks[key]) for var in save_vars}
        )
        events["Dataset"] = key
        bdt_events_dict.append(events)
        bdt_sample_order.append(key)

    bdt_events = pd.concat(bdt_events_dict, axis=0)
    table = pa.Table.from_pandas(bdt_events)
    pq.write_table(table, out_file)

    sample_order_dict = OrderedDict(
        [(sample, len(bdt_events_dict[sample])) for sample in bdt_sample_order]
    )

    with open(out_file.replace("bdt_data.parquet", "sample_order.txt"), "w") as f:
        f.write(str(sample_order_dict))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        default=None,
        help="path to skimmed parquet",
        type=str,
    )
    parser.add_argument(
        "--signal-data-dir",
        default=None,
        help="path to skimmed signal parquets, if different from other data",
        type=str,
    )
    parser.add_argument(
        "--year",
        default="2017",
        choices=["2016", "2016APV", "2017", "2018"],
        type=str,
    )
    parser.add_argument(
        "--sig-samples",
        help="specify signal samples. By default, will use the samples defined in `hh_vars`.",
        nargs="*",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--bg-keys",
        help="specify background samples",
        nargs="*",
        default=["QCD", "TT", "ST", "V+Jets", "Diboson"],
        type=str,
    )
    parser.add_argument(
        "--plot-dir",
        help="If making control or template plots, path to directory to save them in",
        default="",
        type=str,
    )

    utils.add_bool_arg(parser, "data", "include data", default=True)
    utils.add_bool_arg(
        parser, "read-sig-samples", "read signal samples from directory", default=False
    )
    utils.add_bool_arg(parser, "control-plots", "make control plots", default=False)
    utils.add_bool_arg(parser, "resonant", "for resonant or nonresonant", default=False)
    utils.add_bool_arg(parser, "bdt-data", "save bdt training data", default=False)

    args = parser.parse_args()
    main(args)
