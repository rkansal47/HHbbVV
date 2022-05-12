import os

import numpy as np
import pandas as pd
from hist import Hist

import utils
import plotting

from typing import Dict, List, Tuple

from sample_labels import sig_key, data_key, qcd_key, bg_keys
from utils import CUT_MAX_VAL


# Both Jet's Msds > 50 & at least one jet with Txbb > 0.8
filters = [
    [
        ("('ak8FatJetMsd', '0')", ">=", "50"),
        ("('ak8FatJetMsd', '1')", ">=", "50"),
        ("('ak8FatJetParticleNetMD_Txbb', '0')", ">=", "0.8"),
    ],
    [
        ("('ak8FatJetMsd', '0')", ">=", "50"),
        ("('ak8FatJetMsd', '1')", ">=", "50"),
        ("('ak8FatJetParticleNetMD_Txbb', '1')", ">=", "0.8"),
    ],
]

# var: (bins, label)
control_plot_vars = {
    "MET_pt": ([50, 0, 250], r"$p^{miss}_T$ (GeV)"),
    "DijetEta": ([50, -8, 8], r"$\eta^{jj}$"),
    "DijetPt": ([50, 0, 750], r"$p_T^{jj}$ (GeV)"),
    "DijetMass": ([50, 0, 2500], r"$m^{jj}$ (GeV)"),
    "bbFatJetEta": ([50, -3, 3], r"$\eta^{bb}$"),
    "bbFatJetPt": ([50, 200, 1000], r"$p^{bb}_T$ (GeV)"),
    "bbFatJetMsd": ([50, 20, 250], r"$m^{bb}$ (GeV)"),
    "bbFatJetParticleNetMD_Txbb": ([50, 0, 1], r"$p^{bb}_{Txbb}$"),
    "VVFatJetEta": ([50, -3, 3], r"$\eta^{VV}$"),
    "VVFatJetPt": ([50, 200, 1000], r"$p^{VV}_T$ (GeV)"),
    "VVFatJetMsd": ([50, 20, 500], r"$m^{VV}$ (GeV)"),
    "VVFatJetParticleNet_Th4q": ([50, 0, 1], r"$p^{VV}_{Th4q}$"),
    "VVFatJetParticleNetHWWMD_THWW4q": ([50, 0, 1], r"$p^{VV}_{THVV4q}$"),
    "bbFatJetPtOverDijetPt": ([50, 0, 40], r"$p^{bb}_T / p_T^{jj}$"),
    "VVFatJetPtOverDijetPt": ([50, 0, 40], r"$p^{VV}_T / p_T^{jj}$"),
    "VVFatJetPtOverbbFatJetPt": ([50, 0.4, 2.5], r"$p^{VV}_T / p^{bb}_T$"),
    "BDTScore": ([50, 0, 1], r"BDT Score"),
}

selection_regions = {
    "pass": {
        "BDTScore": [0.9602, CUT_MAX_VAL],
        "bbFatJetParticleNetMD_Txbb": [0.98, CUT_MAX_VAL],
    },
    "fail": {
        "bbFatJetParticleNetMD_Txbb": [0.8, 0.98],
    },
}

mass_signal_window = [100, 150]


def main(args):
    from sample_labels import samples, bdt_sample_order

    if args.plot_dir:
        os.system(f"mkdir -p {args.plot_dir}")

    overall_cutflow = pd.DataFrame(index=list(samples.keys()))
    events_dict = load_samples(args.data_dir, samples, args.year, overall_cutflow, filters)
    apply_weights(events_dict, args.year, overall_cutflow)
    bb_masks = bb_VV_assignment(events_dict)
    derive_variables(events_dict, bb_masks)

    if args.bdt_preds != "":
        load_bdt_preds(events_dict, args.bdt_preds, bdt_sample_order)

    if args.control_plots:
        control_plots(events_dict, bb_masks, control_plot_vars, args.plot_dir)

    if args.templates:
        templates = get_templates(events_dict, bb_masks, selection_regions, args.plot_dir)
        save_templates(templates)


def load_samples(
    data_dir: str, samples: List[str], year: str, cutflow: pd.DataFrame = None, filters: List = None
) -> Dict[str, pd.DataFrame]:
    """
    Loads events with an optional filter.
    Reweights samples by nevents.

    Args:
        data_dir (str): path to data directory.
        samples (List[str]): list of samples to load.
        year (str): year.
        cutflow (pd.DataFrame): Optional cutflow dataframe.
        filters (List): Optional filters when loading data.

    Returns:
        Dict[str, pd.DataFrame]: ``events_dict`` dictionary of events dataframe for each sample.

    """

    from os import listdir

    full_samples_list = listdir(f"{data_dir}/{year}")
    events_dict = {}

    for label, selector in samples.items():
        print(label)
        events_dict[label] = []
        for sample in full_samples_list:
            if not sample.startswith(selector):
                continue

            print(sample)

            events = pd.read_parquet(f"{data_dir}/{year}/{sample}/parquet", filters=filters)
            pickles_path = f"{data_dir}/{year}/{sample}/pickles"

            if label != data_key:
                if label == sig_key:
                    n_events = utils.get_cutflow(pickles_path, year, sample)["has_4q"]
                else:
                    n_events = utils.get_nevents(pickles_path, year, sample)

                events["weight"] /= n_events

            events_dict[label].append(events)

        events_dict[label] = pd.concat(events_dict[label])

    if cutflow is not None:
        utils.add_to_cutflow(events_dict, "BDTPreselection", "weight", cutflow)

    return events_dict


def apply_weights(
    events_dict: Dict[str, pd.DataFrame],
    year: str,
    cutflow: pd.DataFrame,
    weight_key: str = "finalWeight",
):
    """
    Applies (1) 2D trigger scale factors, (2) QCD scale facotr.

    Args:
        cutflow (pd.DataFrame): cutflow to which to add yields after scale factors.
        weight_key (str): column in which to store scaled weights in. Defaults to "finalWeight".

    """
    from coffea.lookup_tools.dense_lookup import dense_lookup
    import pickle

    with open(
        f"../corrections/trigEffs/AK8JetHTTriggerEfficiency_{year}.hist", "rb"
    ) as filehandler:
        ak8TrigEffs = pickle.load(filehandler)

    ak8TrigEffsLookup = dense_lookup(
        np.nan_to_num(ak8TrigEffs.view(flow=False), 0), np.squeeze(ak8TrigEffs.axes.edges)
    )

    for sample in events_dict:
        print(sample)
        events = events_dict[sample]
        if sample == data_key:
            events[weight_key] = events["weight"]
        else:
            fj_trigeffs = ak8TrigEffsLookup(
                events["ak8FatJetPt"].values, events["ak8FatJetMsd"].values
            )
            # combined eff = 1 - (1 - fj1_eff) * (1 - fj2_eff)
            combined_trigEffs = 1 - np.prod(1 - fj_trigeffs, axis=1, keepdims=True)
            events[weight_key] = events["weight"] * combined_trigEffs

    utils.add_to_cutflow(events_dict, "TriggerEffs", weight_key, cutflow)

    # calculate QCD scale factor
    trig_yields = cutflow["TriggerEffs"]
    non_qcd_bgs_yield = np.sum(
        [
            trig_yields[sample]
            for sample in events_dict
            if sample not in {sig_key, qcd_key, data_key}
        ]
    )
    QCD_SCALE_FACTOR = (trig_yields[data_key] - non_qcd_bgs_yield) / trig_yields[qcd_key]
    events_dict[qcd_key][weight_key] *= QCD_SCALE_FACTOR

    print(f"{QCD_SCALE_FACTOR = }")

    utils.add_to_cutflow(events_dict, "QCD SF", weight_key, cutflow)


def bb_VV_assignment(events_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Creates a dataframe of masks for extracting the bb or VV candidates.
    bb candidate is chosen based on higher Txbb score.

    Returns:
        Dict[str, pd.DataFrame]: ``bb_masks`` dict of boolean masks for each sample,
          of shape ``[num_events, 2]``.

    """
    bb_masks = {}

    for sample, events in events_dict.items():
        txbb = events["ak8FatJetParticleNetMD_Txbb"]
        bb_mask = txbb[0] >= txbb[1]
        bb_masks[sample] = pd.concat((bb_mask, ~bb_mask), axis=1)

    return bb_masks


def derive_variables(events_dict: Dict[str, pd.DataFrame], bb_masks: Dict[str, pd.DataFrame]):
    """Derives more dijet kinematic variables for control plots."""
    for sample, events in events_dict.items():
        print(sample)
        bb_mask = bb_masks[sample]

        fatjet_vectors = utils.make_vector(events, "ak8FatJet")
        Dijet = fatjet_vectors[:, 0] + fatjet_vectors[:, 1]

        events["DijetPt"] = Dijet.pt
        events["DijetMass"] = Dijet.M
        events["DijetEta"] = Dijet.eta

        events["bbFatJetPtOverDijetPt"] = (
            utils.get_feat(events, "bbFatJetPt", bb_mask) / events["DijetPt"]
        )
        events["VVFatJetPtOverDijetPt"] = (
            utils.get_feat(events, "VVFatJetPt", bb_mask) / events["DijetPt"]
        )
        events["VVFatJetPtOverbbFatJetPt"] = utils.get_feat(
            events, "VVFatJetPt", bb_mask
        ) / utils.get_feat(events, "bbFatJetPt", bb_mask)


def load_bdt_preds(
    events_dict: Dict[str, pd.DataFrame], bdt_preds: str, bdt_sample_order: List[str]
):
    """
    Loads the BDT scores for each event and saves in the dataframe in the "BDTScore" column.

    Args:
        bdt_preds (str): Path to the bdt_preds .npy file.
        bdt_sample_order (List[str]): Order of samples in the predictions file.

    """
    bdt_preds = np.load(bdt_preds)

    i = 0
    for sample in bdt_sample_order:
        events = events_dict[sample]
        num_events = len(events)
        events["BDTScore"] = bdt_preds[i : i + num_events]
        i += num_events


def control_plots(
    events_dict: Dict[str, pd.DataFrame],
    bb_masks: Dict[str, pd.DataFrame],
    control_plot_vars: Dict[str, Tuple],
    plot_dir: str,
    weight_key: str = "finalWeight",
):
    """
    Makes and plots histograms of each variable in ``control_plot_vars``.

    Args:
        control_plot_vars (Dict[str, Tuple]): Dictionary of variables to plot, formatted as
          {var1: ([num bins, min, max], label), var2...}.

    """

    from PyPDF2 import PdfFileMerger

    sig_scale = np.sum(events_dict[data_key][weight_key]) / np.sum(events_dict[sig_key][weight_key])

    hists = {}

    for var, (bins, label) in control_plot_vars.items():
        if var not in events_dict[sig_key]:
            print(f"Control Plots: {var} not found in events, skipping)")
            continue

        if var not in hists:
            print(var)
            hists[var] = utils.singleVarHist(
                events_dict, var, bins, label, bb_masks, weight_key=weight_key
            )

    merger_control_plots = PdfFileMerger()

    for var, var_hist in hists.items():
        name = f"{plot_dir}/{var}.pdf"
        plotting.ratioHistPlot(
            var_hist,
            bg_keys,
            sig_key,
            name=name,
            sig_scale=sig_scale,
        )
        merger_control_plots.append(name)

    merger_control_plots.write(f"{plot_dir}/ControlPlots.pdf")
    merger_control_plots.close()


def get_templates(
    events_dict: Dict[str, pd.DataFrame],
    bb_masks: Dict[str, pd.DataFrame],
    selection_regions: Dict[str, Dict],
    plot_dir: str,
    weight_key: str = "finalWeight",
    prev_cutflow: pd.DataFrame = None,
) -> Dict[str, Hist]:
    """
    (1) Makes histograms for each region in the ``selection_regions`` dictionary,
    (2) Saves a plot of each,
    (3) And for the Pass region calculates the signal and data-driven bg estimate.

    Args:
        selection_region (Dict[str, Dict]): Dictionary of cuts for each region
          formatted as {region1: {cutvar1: [min, max], ...}, ...}.

    Returns:
        Dict[str, Hist]: dictionary of templates, saved as hist.Hist objects.

    """

    selections, cutflows, templates = {}, {}, {}

    for label, region in selection_regions:
        pass_region = label == "pass"

        sel, cf = utils.make_selection(region, events_dict, bb_masks, prev_cutflow=prev_cutflow)
        cf.to_csv(f"{plot_dir}/{label}_region_cutflow.csv")

        template = utils.singleVarHist(
            events_dict,
            "bbFatJetMsd",
            [8, 50, 250],
            r"$m^{bb}$ (GeV)",
            bb_masks,
            selection=sel,
            blind_region=mass_signal_window if pass_region else None,
        )

        sig_scale = utils.getSignalPlotScaleFactor(events_dict, selection=sel)

        plotting.ratioHistPlot(
            template,
            bg_keys,
            name=f"{plot_dir}/{label}_region_bb_mass.pdf",
            sig_scale=sig_scale / 2,
        )

        if pass_region:
            pass_sig_yield, pass_bg_yield = utils.getSigSidebandBGYields(
                "bbFatJetMsd",
                mass_signal_window,
                events_dict,
                bb_masks,
                selection=sel,
            )

            print(
                f"""Pass region signal yield: {pass_sig_yield}
                background yield from data in sidebands: {pass_bg_yield}"""
            )

        selections[label] = sel
        cutflows[label] = cf
        templates[label] = template

    return templates


def save_templates(templates):
    """TODO: Check best way to save templates for use with combine"""
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        default="../../../../data/skimmer/Apr28/",
        help="path to skimmed parquet",
        type=str,
    )

    parser.add_argument(
        "--year",
        default="year",
        choices=["2016", "2016APV", "2017", "2018"],
        type=str,
    )

    utils.add_bool_arg(
        parser, "save-bdt-data", "save parquet files for bdt training", default=False
    )

    parser.add_argument(
        "--bdt-preds",
        help="path to bdt predictions, if empty, don't load",
        default="",
        type=str,
    )

    parser.add_argument(
        "--plot-dir",
        help="If making control plots, path to directory to save them in",
        default="",
        type=str,
    )

    utils.add_bool_arg(parser, "control-plots", "make control plots", default=True)
    utils.add_bool_arg(parser, "templates", "save m_bb templates using bdt cut", default=False)

    args = parser.parse_args()
    main(args)
