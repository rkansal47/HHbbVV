from __future__ import annotations

import argparse
import warnings
from collections import OrderedDict
from copy import copy
from pathlib import Path

import pandas as pd
import postprocessing
import TrainBDT
import utils
from pandas.errors import SettingWithCopyWarning

from HHbbVV import run_utils
from HHbbVV.hh_vars import (
    BDT_sample_order,
    jec_shifts,
    jec_vars,
    jmsr_shifts,
    jmsr_vars,
)

# ignore these because they don't seem to apply
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


BDT_data_vars = [
    "MET_pt",
    # "DijetEta",  # no improvement on BDT
    "DijetPt",
    "DijetMass",
    "bbFatJetPt",
    # "VVFatJetEta",  # no improvement on BDT
    "VVFatJetPt",
    "VVFatJetParticleNetMass",
    "VVFatJetParTMD_THWWvsT",
    "VVFatJetParTMD_probQCD",
    "VVFatJetParTMD_probHWW3q",
    "VVFatJetParTMD_probHWW4q",
    "VVFatJetParTMD_probT",
    "bbFatJetParticleNetMass",  # just for checking training vs testing dists
    "bbFatJetParticleNetMD_Txbb",
    # "bbFatJetPtOverDijetPt",  # no improvement on BDT
    "VVFatJetPtOverDijetPt",
    "VVFatJetPtOverbbFatJetPt",
    "DijetdEta",
    "DijetdPhi",
    "vbf_Mass_jj",
    "vbf_dEta_jj",
    "finalWeight",
]


def main(args):
    # make plot, template dirs if needed
    sig_keys, sig_samples, bg_keys, bg_samples = postprocessing._process_samples(
        args, BDT_sample_order
    )

    # save cutflow as pandas table
    all_samples = sig_keys + bg_keys
    cutflow = pd.DataFrame(index=all_samples)

    events_dict = postprocessing._load_samples(
        args, bg_samples, sig_samples, cutflow, variations=False
    )
    postprocessing.qcd_sf(events_dict, cutflow)
    bb_masks = postprocessing.bb_VV_assignment(events_dict)
    postprocessing.derive_variables(
        events_dict, bb_masks, nonres_vars=True, vbf_vars=False, do_jshifts=True
    )

    bdt_data_dir = (
        (args.data_dir / "bdt_data")
        if args.data_dir
        else (Path(args.signal_data_dirs[0]) / "bdt_data")
    )
    bdt_data_dir.mkdir(exist_ok=True)

    for key in copy(BDT_sample_order):
        if key not in all_samples:
            BDT_sample_order.remove(key)

    bdt_events_dict = get_bdt_data(events_dict, bb_masks, BDT_sample_order)

    if args.save_data:
        save_bdt_data(
            bdt_events_dict, BDT_sample_order, bdt_data_dir / f"{args.year}_bdt_data.parquet"
        )

    if args.inference:
        run_inference(args.year, bdt_events_dict, args.bdt_preds_dir, args.do_jshifts)


def get_bdt_data(
    events_dict: dict[str, pd.DataFrame],
    bb_masks: dict[str, pd.DataFrame],
    BDT_sample_order: str,
):
    jec_jmsr_vars = []

    for var in BDT_data_vars:
        if var in jec_vars:
            for jshift in jec_shifts:
                jec_jmsr_vars.append(f"{var}_{jshift}")

        if var in jmsr_vars:
            for jshift in jmsr_shifts:
                jec_jmsr_vars.append(f"{var}_{jshift}")

    bdt_events_dict = []
    for key in BDT_sample_order:
        save_vars = BDT_data_vars + jec_jmsr_vars if key != "Data" else BDT_data_vars
        events = pd.DataFrame(
            {var: utils.get_feat(events_dict[key], var, bb_masks[key]) for var in save_vars}
        )
        events["Dataset"] = key
        bdt_events_dict.append(events)

    return bdt_events_dict


def save_bdt_data(bdt_events_dict: list[pd.DataFrame], BDT_sample_order: list[str], out_file: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    print("Saving BDT data to", out_file)

    bdt_events = pd.concat(bdt_events_dict, axis=0)
    table = pa.Table.from_pandas(bdt_events)
    pq.write_table(table, out_file)

    sample_order_dict = OrderedDict(
        [(sample, len(bdt_events_dict[i])) for i, sample in enumerate(BDT_sample_order)]
    )

    with Path(str(out_file).replace("bdt_data.parquet", "sample_order.txt")).open("w") as f:
        f.write(str(sample_order_dict))


def run_inference(
    year: str, bdt_events_dict: list[pd.DataFrame], bdt_preds_dir: str, do_jshifts: bool
):
    import xgboost as xgb

    model = xgb.XGBClassifier()
    model.load_model(args.bdt_model)

    bdt_events = pd.concat(bdt_events_dict, axis=0)

    TrainBDT.do_inference_year(
        model,
        bdt_preds_dir,
        year,
        bdt_events,
        TrainBDT.AllTaggerBDTVars,
        jec_jmsr_shifts=do_jshifts,
        multiclass=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    run_utils.add_bool_arg(parser, "save-data", default=True, help="save preprocessed data")
    run_utils.add_bool_arg(parser, "inference", default=False, help="run inference on data")
    parser.add_argument(
        "--bdt-model",
        default="src/HHbbVV/postprocessing/bdt_models/24_04_05_k2v0_training_eqsig_vbf_vars_rm_deta.model",
        help="path to BDT model, if running inference",
        type=str,
    )
    args = postprocessing.parse_args(parser)
    if args.data_dir:
        args.data_dir = Path(args.data_dir)

    main(args)
