"""
Takes the skimmed parquet files (output of bbVVSkimmer + YieldsAnalysis.py) and trains a BDT using xgboost.

Author(s): Raghav Kansal
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import hist
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
import pandas as pd
import plotting
import utils
import xgboost as xgb
from hist import Hist
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from HHbbVV.hh_vars import (
    data_key,
    jec_shifts,
    jec_vars,
    jmsr_shifts,
    jmsr_vars,
    nonres_samples,
    years,
)
from HHbbVV.run_utils import add_bool_arg

try:
    from pandas.errors import SettingWithCopyWarning
except:
    from pandas.core.common import SettingWithCopyWarning

# ignore these because they don't seem to apply
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

plt.style.use(hep.style.CMS)
hep.style.use("CMS")
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))
plt.rcParams.update({"font.size": 28})


weight_key = "finalWeight"


# only vars used for training, ordered by importance
AllTaggerBDTVars = [
    # "VVFatJetParTMD_THWW4q",
    "VVFatJetParTMD_probHWW3q",
    "VVFatJetParTMD_probQCD",
    "VVFatJetParTMD_probHWW4q",
    "VVFatJetParticleNetMass",
    "DijetMass",
    "VVFatJetParTMD_probT",
    "VVFatJetPtOverDijetPt",
    "DijetPt",
    "bbFatJetPt",
    "VVFatJetPt",
    "VVFatJetPtOverbbFatJetPt",
    "MET_pt",
]


SingleTaggerBDTVars = [
    "VVFatJetParTMD_THWWvsT",
    "VVFatJetParticleNetMass",
    "DijetMass",
    "VVFatJetPtOverDijetPt",
    "DijetPt",
    "bbFatJetPt",
    "VVFatJetPt",
    "VVFatJetPtOverbbFatJetPt",
    "MET_pt",
]


# ignore bins
var_label_map = {
    "MET_pt": ([50, 0, 250], r"$p^{miss}_T$ (GeV)"),
    "DijetEta": ([50, -8, 8], r"$\eta^{jj}$"),
    "DijetPt": ([50, 0, 750], r"$p_T^{jj}$ (GeV)"),
    "DijetMass": ([50, 500, 3000], r"$m^{jj}$ (GeV)"),
    "bbFatJetEta": ([50, -2.4, 2.4], r"$\eta^{bb}$"),
    "bbFatJetPt": ([50, 300, 1300], r"$p^{bb}_T$ (GeV)"),
    "VVFatJetEta": ([50, -2.4, 2.4], r"$\eta^{VV}$"),
    "VVFatJetPt": ([50, 300, 1300], r"$p^{VV}_T$ (GeV)"),
    "VVFatJetParticleNetMass": ([50, 0, 300], r"$m^{VV}_{reg}$ (GeV)"),
    # "VVFatJetMsd": ([50, 0, 300], r"$m^{VV}_{msd}$ (GeV)"),
    "VVFatJetParTMD_THWWvsT": ([50, 0, 1], r"ParT $T_{HWW}$"),
    "VVFatJetParTMD_probT": ([50, 0, 1], r"ParT $Prob(Top)^{VV}$"),
    "VVFatJetParTMD_probQCD": ([50, 0, 1], r"ParT $Prob(QCD)^{VV}$"),
    "VVFatJetParTMD_probHWW3q": ([50, 0, 1], r"ParT $Prob(HWW3q)^{VV}$"),
    "VVFatJetParTMD_probHWW4q": ([50, 0, 1], r"ParT $Prob(HWW4q)^{VV}$"),
    "bbFatJetPtOverDijetPt": ([50, 0, 40], r"$p^{bb}_T / p_T^{jj}$"),
    "VVFatJetPtOverDijetPt": ([50, 0, 40], r"$p^{VV}_T / p_T^{jj}$"),
    "VVFatJetPtOverbbFatJetPt": ([50, 0.4, 2.0], r"$p^{VV}_T / p^{bb}_T$"),
}


def get_X(
    data_dict: dict[str, pd.DataFrame],
    bdtVars: list[str],
    jec_shift: str = None,
    jmsr_shift: str = None,
):
    """
    Gets variables for BDT for all samples in ``data``.
    Optionally gets shifted variables (in which returns only MC samples).
    """
    X = []

    if jec_shift is None and jmsr_shift is None:
        for _year, data in data_dict.items():
            X.append(data.filter(items=bdtVars))

        return pd.concat(X, axis=0)

    mc_vars = deepcopy(bdtVars)

    if jec_shift is not None:
        for i, var in enumerate(mc_vars):
            if var in jec_vars:
                mc_vars[i] = f"{var}_{jec_shift}"

    if jmsr_shift is not None:
        for i, var in enumerate(mc_vars):
            if var in jmsr_vars:
                mc_vars[i] = f"{var}_{jmsr_shift}"

    for _year, data in data_dict.items():
        X.append(data.filter(items=mc_vars)[data["Dataset"] != data_key])

    return pd.concat(X, axis=0), mc_vars


def get_Y(
    data_dict: dict[str, pd.DataFrame],
    sig_key: list[str] = None,
    multiclass: bool = False,
    label_encoder: LabelEncoder = None,
):
    Y = []
    for _year, data in data_dict.items():
        if multiclass:
            Y.append(pd.DataFrame(label_encoder.transform(data["Dataset"])))
        else:
            Y.append((data["Dataset"] == sig_key).astype(int))

    return pd.concat(Y, axis=0)


def add_preds(data_dict: dict[str, pd.DataFrame], preds: np.ndarray, sig_keys: list[str]):
    """Adds BDT predictions to ``data_dict``."""
    count = 0
    for _year, data in data_dict.items():
        if len(sig_keys) == 1:
            data["BDTScore"] = preds[count : count + len(data)]
        else:
            for i, sig_key in enumerate(sig_keys):
                data[f"BDTScore{sig_key}"] = preds[count : count + len(data), i]
        count += len(data)

    return data_dict


def get_weights(data_dict: dict[str, pd.DataFrame], abs_weights: bool = True):
    weights = []
    for _year, data in data_dict.items():
        weights.append(np.abs(data[weight_key]) if abs_weights else data[weight_key])

    return pd.concat(weights, axis=0)


def remove_neg_weights(data: pd.DataFrame):
    return data[data[weight_key] > 0]


def equalize_weights(
    data: pd.DataFrame,
    sig_keys: list[str],
    bg_keys: list[str],
    equalize_sig_bg: bool = True,
    equalize_per_process: bool = False,
    equalize_sig_total: bool = True,  # TODO: add arg
):
    """
    If `equalize_sig_bg`: scales signal such that total signal = total background
    If `equalize_per_process`: scales each background process separately to be equal as well
        If `equalize_sig_bg` is False: signal scaled to match the individual bg process yields
        instead of total.
    if `equalize_sig_total`: all signals combined equal total background, otherwise, each signal matched total background
    """

    if equalize_per_process:
        if len(sig_keys) > 1:
            raise NotImplementedError("Equalize per process implemented yet for multiple sig keys")

        sig_key = sig_keys[0]
        sig_total = np.sum(data[data["Dataset"] == sig_keys[0]][weight_key])

        qcd_total = np.sum(data[data["Dataset"] == "QCD"][weight_key])
        for bg_key in bg_keys:
            if bg_key != "QCD":
                total = np.sum(data[data["Dataset"] == bg_key][weight_key])
                data[weight_key].loc[data["Dataset"] == bg_key] *= qcd_total / total

        if not equalize_sig_bg:
            data[weight_key].loc[data["Dataset"] == sig_key] *= qcd_total / sig_total

    if equalize_sig_bg:
        bg_total = np.sum(
            data[np.all([data["Dataset"] != sig_key for sig_key in sig_keys], axis=0)][weight_key]
        )

        sig_factor = 1.0 / len(sig_keys) if equalize_sig_total else 1.0
        for sig_key in sig_keys:
            sig_total = np.sum(data[data["Dataset"] == sig_key][weight_key])
            data[weight_key].loc[data["Dataset"] == sig_key] *= bg_total / sig_total * sig_factor


def load_data(data_path: str, year: str, all_years: bool):
    if not all_years:
        return OrderedDict([(year, pd.read_parquet(f"{data_path}/{year}_bdt_data.parquet"))])
    else:
        return OrderedDict(
            [(year, pd.read_parquet(f"{data_path}/{year}_bdt_data.parquet")) for year in years]
        )


def main(args):
    bg_keys = ["QCD", "TT", "Z+Jets"]
    if args.wjets_training:
        bg_keys += ["W+Jets"]

    sig_keys = args.sig_keys
    if len(sig_keys) > 1:
        assert args.multiclass, "need --multiclass training for multiple sig keys"
    for sig_key in sig_keys:
        assert sig_key in nonres_samples, f"{sig_key} is not a valid signal key"

    training_keys = sig_keys + bg_keys

    # for multiclass classification, encoding each process separately
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(training_keys)  # need this to maintain training keys order

    bdtVars = AllTaggerBDTVars if args.all_tagger_vars else SingleTaggerBDTVars

    early_stopping_callback = xgb.callback.EarlyStopping(
        rounds=args.early_stopping_rounds, min_delta=args.early_stopping_min_delta
    )

    classifier_params = {
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "verbosity": 2,
        "n_jobs": 4,
        "reg_lambda": 1.0,
        "callbacks": [early_stopping_callback],
    }

    if args.rem_feats:
        bdtVars = bdtVars[: -args.rem_feats]

    print("BDT features:\n", bdtVars)

    data_dict = load_data(args.data_path, args.year, args.all_years)

    for year, data in data_dict.items():
        for key in training_keys:
            print(
                f"{year} {key} Yield: "
                f'{np.sum(data[data["Dataset"] == key][weight_key])}, '
                "Number of Events: "
                f'{len(data[data["Dataset"] == key])}, '
            )

        bg_select = np.sum(
            [data["Dataset"] == key for key in bg_keys],
            axis=0,
        ).astype(bool)

        print(
            f"Total BG Yield: {np.sum(data[bg_select][weight_key])}, "
            f"Number of Events: {len(data[bg_select])}"
        )

    if not args.inference_only:
        training_data_dict = OrderedDict(
            [
                (
                    year,
                    data[
                        # select only signals and `bg_keys` backgrounds for training - rest are only inferenced
                        np.sum(
                            [data["Dataset"] == key for key in training_keys],
                            axis=0,
                        ).astype(bool)
                    ],
                )
                for year, data in data_dict.items()
            ]
        )

        training_samples = np.unique(next(iter(training_data_dict.values()))["Dataset"])
        print("Training samples:", training_samples)

        if args.test:
            # get a sample of different processes
            data_dict = OrderedDict(
                [
                    (
                        year,
                        pd.concat(
                            (data[:50], data[1000000:1000050], data[2000000:2000050], data[-50:]),
                            axis=0,
                        ),
                    )
                    for year, data in data_dict.items()
                ]
            )

            # 50 events from each training process
            training_data_dict = OrderedDict(
                [
                    (
                        year,
                        pd.concat(
                            [data[data["Dataset"] == key][:50] for key in training_samples],
                            axis=0,
                        ),
                    )
                    for year, data in training_data_dict.items()
                ]
            )

        if args.equalize_weights or args.equalize_weights_per_process:
            for year, data in training_data_dict.items():
                for key in training_keys:
                    print(
                        f"Pre-equalization {year} {key} total: "
                        f'{np.sum(data[data["Dataset"] == key][weight_key])}'
                    )

                equalize_weights(
                    data,
                    sig_keys,
                    bg_keys,
                    args.equalize_weights,
                    args.equalize_weights_per_process,
                    args.equalize_sig_total,
                )

                for key in training_keys:
                    print(
                        f"Post-equalization {year} {key} total: "
                        f'{np.sum(data[data["Dataset"] == key][weight_key])}'
                    )

                print("")

        if len(training_samples) > 0:
            train, test = OrderedDict(), OrderedDict()
            for year, data in training_data_dict.items():
                train[year], test[year] = train_test_split(
                    remove_neg_weights(data) if not args.absolute_weights else data,
                    test_size=args.test_size,
                    random_state=args.seed,
                )

    if args.evaluate_only or args.inference_only:
        model = xgb.XGBClassifier()
        model.load_model(args.model_dir / "trained_bdt.model")
    else:
        args.model_dir.mkdir(exist_ok=True, parents=True)
        model = train_model(
            get_X(train, bdtVars),
            get_X(test, bdtVars),
            get_Y(train, sig_keys[0], args.multiclass, label_encoder),
            get_Y(test, sig_keys[0], args.multiclass, label_encoder),
            get_weights(train, args.absolute_weights),
            get_weights(test, args.absolute_weights),
            bdtVars,
            args.model_dir,
            use_sample_weights=args.use_sample_weights,
            **classifier_params,
        )

    if not args.inference_only:
        evaluate_model(
            model,
            args.model_dir,
            train,
            test,
            args.test_size,
            sig_keys,
            training_keys,
            args.equalize_weights,
            bdtVars,
            multiclass=args.multiclass,
        )

    if not args.evaluate_only:
        do_inference(model, args.model_dir, data_dict, bdtVars, multiclass=args.multiclass)


def plot_losses(trained_model: xgb.XGBClassifier, model_dir: Path):
    evals_result = trained_model.evals_result()

    with (model_dir / "evals_result.txt").open("w") as f:
        f.write(str(evals_result))

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(["Train", "Test"]):
        plt.plot(evals_result[f"validation_{i}"]["mlogloss"], label=label, linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(model_dir / "losses.pdf", bbox_inches="tight")
    plt.close()


def train_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    weights_train: np.ndarray,
    weights_test: np.ndarray,
    bdtVars: list[str],
    model_dir: Path,
    use_sample_weights: bool = False,
    **classifier_params,
):
    """Trains BDT. ``classifier_params`` are hyperparameters for the classifier"""
    print("Training model")
    model = xgb.XGBClassifier(**classifier_params)
    print("Training features: ", list(X_train.columns))
    assert set(bdtVars) == set(X_train.columns), "Missing Training Vars!"
    trained_model = model.fit(
        X_train,
        y_train,
        sample_weight=weights_train if use_sample_weights else None,
        # xgboost uses the last set for early stopping
        # https://xgboost.readthedocs.io/en/stable/python/python_intro.html#early-stopping
        eval_set=[(X_train, y_train), (X_test, y_test)],
        sample_weight_eval_set=[weights_train, weights_test] if use_sample_weights else None,
        verbose=True,
    )
    trained_model.save_model(model_dir / "trained_bdt.model")
    plot_losses(trained_model, model_dir)
    return model


def _txbb_thresholds(test: dict[str, pd.DataFrame], txbb_threshold: float):
    t = []
    for _year, data in test.items():
        t.append(data["bbFatJetParticleNetMD_Txbb"] < txbb_threshold)

    return pd.concat(t, axis=0)


def _get_bdt_scores(preds, sig_keys, multiclass):
    """Helper function to calculate which BDT outputs to use"""
    if not multiclass:
        return preds[:, 1:]
    else:
        if len(sig_keys) == 1:
            return preds[:, :1]
        else:
            # Relevant score is signal score / (signal score + all background scores)
            bg_tot = np.sum(preds[:, len(sig_keys) :], axis=1, keepdims=True)
            return preds[:, : len(sig_keys)] / (preds[:, : len(sig_keys)] + bg_tot)


def evaluate_model(
    model: xgb.XGBClassifier,
    model_dir: str,
    train: dict[str, pd.DataFrame],
    test: dict[str, pd.DataFrame],
    test_size: float,
    sig_keys: list[str],
    training_keys: list[str],
    equalize_sig_bg: bool,
    bdtVars: list[str],
    txbb_threshold: float = 0.98,
    multiclass: bool = False,
):
    """
    1) Saves feature importance
    2) Makes ROC curves for training and testing data
    3) Combined ROC Curve
    4) Plots BDT score shape
    """
    print("Evaluating model")

    var_labels = [var_label_map[var][1] for var in bdtVars]

    # sorting by importance
    feature_importances = np.stack((var_labels, model.feature_importances_)).T[
        np.argsort(model.feature_importances_)[::-1]
    ]

    feature_importance_df = pd.DataFrame.from_dict({"Importance": feature_importances[:, 1]})
    feature_importance_df.index = feature_importances[:, 0]
    feature_importance_df.to_csv(f"{model_dir}/feature_importances.csv")
    feature_importance_df.to_markdown(f"{model_dir}/feature_importances.md")

    print(feature_importance_df)

    # make and save ROCs for training and testing data
    rocs = OrderedDict()

    ttlabelmap = {"train": "Train", "test": "Test"}
    for data, label in [(train, "train"), (test, "test")]:
        save_model_dir = model_dir / f"rocs_{label}"
        save_model_dir.mkdir(exist_ok=True, parents=True)

        weights_test = get_weights(data)

        preds = model.predict_proba(get_X(data, bdtVars))
        print("pre preds", preds[:10])
        preds = _get_bdt_scores(preds, sig_keys, multiclass)
        print("post preds", preds[:10])
        add_preds(data, preds, sig_keys)

        sig_effs = [0.15, 0.2]

        rocs[label] = {}

        for i, sig_key in enumerate(sig_keys):
            print(sig_key)
            Y = get_Y(data, sig_key, multiclass=False)
            fpr, tpr, thresholds = roc_curve(Y, preds[:, i], sample_weight=weights_test)

            rocs[label][sig_key] = {
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
                "label": f"{ttlabelmap[label]} {plotting.sample_label_map[sig_key]}",
            }

            for sig_eff in sig_effs:
                thresh = thresholds[np.searchsorted(tpr, sig_eff)]
                print(f"Threshold at {sig_eff} sig_eff: {thresh:0.4f}")

        with (save_model_dir / "roc_dict.pkl").open("wb") as f:
            pickle.dump(rocs[label], f)

        plotting.multiROCCurveGrey(
            {label: rocs[label]}, sig_effs=sig_effs, plot_dir=save_model_dir, name="roc"
        )
        plotting.multiROCCurve({label: rocs[label]}, plot_dir=save_model_dir, name="roc_thresholds")

        if txbb_threshold > 0 and len(sig_keys) == 1:
            preds_txbb_thresholded = preds.copy()
            preds_txbb_thresholded[_txbb_thresholds(data, txbb_threshold)] = 0

            fpr_txbb_threshold, tpr_txbb_threshold, thresholds_txbb_threshold = roc_curve(
                Y, preds_txbb_thresholded, sample_weight=weights_test
            )

            plotting.rocCurve(
                fpr_txbb_threshold,
                tpr_txbb_threshold,
                # auc(fpr_txbb_threshold, tpr_txbb_threshold),
                sig_eff_lines=sig_effs,
                title=f"Including Txbb > {txbb_threshold} Cut",
                plot_dir=save_model_dir,
                name="bdtroc_txbb_cut",
            )

            np.savetxt(f"{save_model_dir}/fpr_txbb_threshold.txt", fpr_txbb_threshold)
            np.savetxt(f"{save_model_dir}/tpr_txbb_threshold.txt", tpr_txbb_threshold)
            np.savetxt(f"{save_model_dir}/thresholds_txbb_threshold.txt", thresholds_txbb_threshold)

            for sig_eff in sig_effs:
                thresh = thresholds_txbb_threshold[np.searchsorted(tpr_txbb_threshold, sig_eff)]
                print(f"Incl Txbb Threshold at {sig_eff} sig_eff: {thresh:0.4f}")

    # combined ROC curve with thresholds
    plotting.multiROCCurveGrey(
        rocs, sig_effs=[0.05, 0.1, 0.15, 0.2], plot_dir=model_dir, name="roc"
    )
    plotting.multiROCCurve(rocs, plot_dir=model_dir, name="roc_combined_thresholds")
    plotting.multiROCCurve(rocs, thresholds=[], plot_dir=model_dir, name="roc_combined")

    # BDT score shapes
    bins = [[20, 0, 1], [20, 0.4, 1], [20, 0.8, 1], [20, 0.9, 1], [20, 0.98, 1]]

    if len(sig_keys) == 1:
        scores = [("BDTScore", "BDT Score")]
    else:
        scores = [
            (f"BDTScore{sig_key}", f"BDT Score {plotting.sample_label_map[sig_key]}")
            for sig_key in sig_keys
        ]

    plot_vars = [utils.ShapeVar(*score, tbins) for score in scores for tbins in bins]

    save_model_dir = model_dir / "hists"
    save_model_dir.mkdir(exist_ok=True, parents=True)

    for year in train:
        for shape_var in plot_vars:
            h = Hist(
                hist.axis.StrCategory(["Train", "Test"], name="Data"),
                hist.axis.StrCategory(training_keys, name="Sample"),
                shape_var.axis,
                storage="weight",
            )

            for dataset, label in [(train, "Train"), (test, "Test")]:
                # Normalize the two distributions
                data_sf = (0.5 / test_size) if label == "Test" else (0.5 / (1 - test_size))
                for key in training_keys:
                    # scale signals back down to normal by ~equalizing scale factor
                    sf = data_sf / 1e6 if (key in sig_keys and equalize_sig_bg) else data_sf
                    data = dataset[year][dataset[year]["Dataset"] == key]
                    fill_data = {shape_var.var: data[shape_var.var]}
                    h.fill(Data=label, Sample=key, **fill_data, weight=data[weight_key] * sf)

            plotting.ratioTestTrain(
                h,
                training_keys,
                shape_var,
                year,
                save_model_dir,
                name=f"{year}_{shape_var.var}_{shape_var.bins[1]}",
            )

            plotting.ratioTestTrain(
                h,
                [key for key in training_keys if key != "QCD"],
                shape_var,
                year,
                save_model_dir,
                name=f"{year}_{shape_var.var}_{shape_var.bins[1]}_noqcd",
            )

    # temporarily save train and test data as pickles to iterate on plots
    with (model_dir / "train.pkl").open("wb") as f:
        pickle.dump(train, f)

    with (model_dir / "test.pkl").open("wb") as f:
        pickle.dump(test, f)


def do_inference(
    model: xgb.XGBClassifier,
    model_dir: str,
    data_dict: dict[str, pd.DataFrame],
    bdtVars: list[str],
    jec_jmsr_shifts: bool = True,
    multiclass: bool = False,
):
    """ """
    import time

    (model_dir / "inferences").mkdir(exist_ok=True, parents=True)

    for year, data in data_dict.items():
        year_data_dict = {year: data}
        (model_dir / "inferences" / year).mkdir(exist_ok=True, parents=True)

        sample_order = list(pd.unique(data["Dataset"]))
        value_counts = data["Dataset"].value_counts()
        sample_order_dict = OrderedDict([(sample, value_counts[sample]) for sample in sample_order])

        with (model_dir / f"inferences/{year}/sample_order.txt").open("w") as f:
            f.write(str(sample_order_dict))

        print("Running inference")
        X = get_X(year_data_dict, bdtVars)
        model.get_booster().feature_names = bdtVars

        start = time.time()
        preds = model.predict_proba(X)
        print(f"Finished in {time.time() - start:.2f}s")
        # preds = preds[:, :-1] if multiclass else preds[:, 1]  # save n-1 probs to save space
        preds = preds if multiclass else preds[:, 1]
        np.save(f"{model_dir}/inferences/{year}/preds.npy", preds)

        if jec_jmsr_shifts:
            for jshift in jec_shifts:
                print("Running inference for", jshift)
                X, mcvars = get_X(year_data_dict, bdtVars, jec_shift=jshift)
                # have to change model's feature names since we're passing in a dataframe
                model.get_booster().feature_names = mcvars
                preds = model.predict_proba(X)
                preds = preds[:, :-1] if multiclass else preds[:, 1]
                np.save(f"{model_dir}/inferences/{year}/preds_{jshift}.npy", preds)

            for jshift in jmsr_shifts:
                print("Running inference for", jshift)
                X, mcvars = get_X(year_data_dict, bdtVars, jmsr_shift=jshift)
                # have to change model's feature names since we're passing in a dataframe
                model.get_booster().feature_names = mcvars
                preds = model.predict_proba(X)
                preds = preds[:, :-1] if multiclass else preds[:, 1]
                np.save(f"{model_dir}/inferences/{year}/preds_{jshift}.npy", preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path",
        default="/hhbbvvvol/data/",
        help="path to training parquet",
        type=str,
    )
    parser.add_argument(
        "--model-dir",
        default="testBDT",
        help="directory in which to save model and evaluation output",
        type=str,
    )
    parser.add_argument(
        "--year",
        choices=["2016", "2016APV", "2017", "2018", "all"],
        type=str,
        required=True,
    )
    add_bool_arg(parser, "load-data", "Load pre-processed data if done already", default=True)
    add_bool_arg(parser, "save-data", "Save pre-processed data if loading the data", default=True)

    parser.add_argument(
        "--num-events",
        default=0,
        help="Num events per sample to train on - if 0 train on all",
        type=int,
    )

    parser.add_argument(
        "--sig-keys", default=["HHbbVV"], help="which signals to train on", type=str, nargs="+"
    )
    add_bool_arg(parser, "wjets-training", "Include W+Jets in training", default=False)

    """
    Varying between 0.01 - 1 showed no significant difference
    https://hhbbvv.nrp-nautilus.io/bdt/24_03_07_new_samples_lr_0.01/
    https://hhbbvv.nrp-nautilus.io/bdt/24_03_07_new_samples_nestimators_10000/
    https://hhbbvv.nrp-nautilus.io/bdt/24_03_07_new_samples_lr_1/
    """
    parser.add_argument("--learning-rate", default=0.1, help="learning rate", type=float)
    """
    hyperparam optimizations show max depth 5 is optimal:
    https://hhbbvv.nrp-nautilus.io/bdt/24_03_07_new_samples_nestimators_10000/
    https://hhbbvv.nrp-nautilus.io/bdt/24_03_07_new_samples_max_depth_4/
    https://hhbbvv.nrp-nautilus.io/bdt/24_03_07_new_samples_max_depth_5/
    https://hhbbvv.nrp-nautilus.io/bdt/24_03_07_new_samples_max_depth_6/
    unclear if gain from 4 is enough to justify increasing complexity
    """
    parser.add_argument("--max-depth", default=5, help="max depth of each tree", type=int)
    """
    hyperparam optimizations show min child weight has ~no effect
    https://hhbbvv.nrp-nautilus.io/bdt/23_05_10_multiclass_max_depth_3_min_child_1_n_1000/
    https://hhbbvv.nrp-nautilus.io/bdt/23_05_10_multiclass_max_depth_3_min_child_5_n_1000/
    """
    parser.add_argument(
        "--min-child-weight",
        default=1,
        help="minimum weight required to keep splitting (higher is more conservative)",
        type=float,
    )

    # This just needs to be higher than the # rounds needed for early-stopping to kick in
    parser.add_argument(
        "--n-estimators", default=10000, help="max number of trees to keep adding", type=int
    )

    parser.add_argument("--rem-feats", default=0, help="remove N lowest importance feats", type=int)

    """
    Slightly worse to use a single tagger score
    https://hhbbvv.nrp-nautilus.io/bdt/24_03_07_new_samples_single_tagger_var
    """
    add_bool_arg(
        parser, "all-tagger-vars", "Use all tagger outputs vs. single THWWvsT score", default=True
    )
    add_bool_arg(parser, "multiclass", "Classify each background separately", default=True)

    add_bool_arg(parser, "use-sample-weights", "Use properly scaled event weights", default=True)
    add_bool_arg(
        parser,
        "absolute-weights",
        "Use absolute weights if using sample weights (if false, will remove negative weights)",
        default=True,
    )
    add_bool_arg(
        parser, "equalize-weights", "Equalise total signal and background weights", default=True
    )
    add_bool_arg(
        parser,
        "equalize-weights-per-process",
        "Equalise each backgrounds' weights too",
        default=False,
    )
    add_bool_arg(
        parser,
        "equalize-sig-total",
        "Total signal = total bg, rather than each signal's total equals the total background (only matters for multiple signals)",
        default=False,
    )

    parser.add_argument(
        "--early-stopping-rounds", default=5, help="early stopping rounds", type=int
    )
    """
    Increasing this consistently decreased performance
    e.g. https://hhbbvv.nrp-nautilus.io/bdt/24_03_07_new_samples_min_delta_0.0001/
    """
    parser.add_argument(
        "--early-stopping-min-delta",
        default=0.0,
        help="min abs improvement needed for early stopping",
        type=float,
    )
    parser.add_argument("--test-size", default=0.3, help="testing/training split", type=float)
    parser.add_argument("--seed", default=4, help="seed for testing/training split", type=int)

    add_bool_arg(parser, "evaluate-only", "Only evaluation, no training", default=False)
    add_bool_arg(parser, "inference-only", "Only inference, no training", default=False)
    add_bool_arg(parser, "test", "Testing BDT Training - run on a small sample", default=False)

    args = parser.parse_args()

    if args.equalize_weights or args.equalize_weights_per_process:
        args.use_sample_weights = True  # sample weights are used before equalizing

    args.all_years = args.year == "all"
    args.model_dir = Path(args.model_dir)

    main(args)
