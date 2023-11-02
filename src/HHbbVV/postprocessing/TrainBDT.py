"""
Takes the skimmed parquet files (output of bbVVSkimmer + YieldsAnalysis.py) and trains a BDT using xgboost.

Author(s): Raghav Kansal
"""

import argparse
from collections import OrderedDict
import os
import pickle
from typing import Dict
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib.ticker as mticker

plt.style.use(hep.style.CMS)
hep.style.use("CMS")
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))
plt.rcParams.update({"font.size": 28})

# from pandas.errors import SettingWithCopyWarning

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

import utils
import plotting

from hh_vars import years, data_key, jec_shifts, jmsr_shifts, jec_vars, jmsr_vars

from copy import deepcopy


# ignore these because they don't seem to apply
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


weight_key = "finalWeight"
sig_key = "HHbbVV"
bg_keys = ["QCD", "TT", "V+Jets"]
training_keys = [sig_key] + bg_keys

# if doing multiclass classification, encode each process separately
label_encoder = LabelEncoder()
label_encoder.fit(training_keys)


# only vars used for training, ordered by importance
bdtVars = [
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
    "bbFatJetPtOverDijetPt",
    "VVFatJetEta",
    "DijetEta",
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
    "VVFatJetParTMD_probT": ([50, 0, 1], r"ParT $Prob(Top)^{VV}$"),
    "VVFatJetParTMD_probQCD": ([50, 0, 1], r"ParT $Prob(QCD)^{VV}$"),
    "VVFatJetParTMD_probHWW3q": ([50, 0, 1], r"ParT $Prob(HWW3q)^{VV}$"),
    "VVFatJetParTMD_probHWW4q": ([50, 0, 1], r"ParT $Prob(HWW4q)^{VV}$"),
    "bbFatJetPtOverDijetPt": ([50, 0, 40], r"$p^{bb}_T / p_T^{jj}$"),
    "VVFatJetPtOverDijetPt": ([50, 0, 40], r"$p^{VV}_T / p_T^{jj}$"),
    "VVFatJetPtOverbbFatJetPt": ([50, 0.4, 2.0], r"$p^{VV}_T / p^{bb}_T$"),
}


def get_X(data_dict: Dict[str, pd.DataFrame], jec_shift: str = None, jmsr_shift: str = None):
    """
    Gets variables for BDT for all samples in ``data``.
    Optionally gets shifted variables (in which returns only MC samples).
    """
    X = []

    if jec_shift is None and jmsr_shift is None:
        for year, data in data_dict.items():
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

    for year, data in data_dict.items():
        X.append(data.filter(items=mc_vars)[data["Dataset"] != data_key])

    return pd.concat(X, axis=0), mc_vars


def get_Y(data_dict: Dict[str, pd.DataFrame], multiclass: bool = False):
    Y = []
    for year, data in data_dict.items():
        if multiclass:
            Y.append(pd.DataFrame(label_encoder.transform(data["Dataset"])))
        else:
            Y.append((data["Dataset"] == sig_key).astype(int))

    return pd.concat(Y, axis=0)


def get_weights(data_dict: Dict[str, pd.DataFrame], abs_weights: bool = True):
    weights = []
    for year, data in data_dict.items():
        weights.append(np.abs(data[weight_key]) if abs_weights else data[weight_key])

    return pd.concat(weights, axis=0)


def remove_neg_weights(data: pd.DataFrame):
    return data[data[weight_key] > 0]


def equalize_weights(
    data: pd.DataFrame, equalize_sig_bg: bool = True, equalize_per_process: bool = False
):
    """
    If `equalize_sig_bg`: scales signal such that total signal = total background
    If `equalize_per_process`: scales each background process separately to be equal as well
        If `equalize_sig_bg` is False: signal scaled to match the individual bg process yields
        instead of total.
    """
    sig_total = np.sum(data[data["Dataset"] == sig_key][weight_key])

    if equalize_per_process:
        qcd_total = np.sum(data[data["Dataset"] == "QCD"][weight_key])
        for bg_key in bg_keys:
            if bg_key != "QCD":
                total = np.sum(data[data["Dataset"] == bg_key][weight_key])
                data[weight_key].loc[data["Dataset"] == bg_key] *= qcd_total / total

        if not equalize_sig_bg:
            data[weight_key].loc[data["Dataset"] == sig_key] *= qcd_total / sig_total

    if equalize_sig_bg:
        bg_total = np.sum(data[data["Dataset"] != sig_key][weight_key])
        data[weight_key].loc[data["Dataset"] == sig_key] *= bg_total / sig_total


def load_data(data_path: str, year: str, all_years: bool):
    if not all_years:
        return OrderedDict([(year, pd.read_parquet(f"{data_path}/{year}_bdt_data.parquet"))])
    else:
        return OrderedDict(
            [(year, pd.read_parquet(f"{data_path}/{year}_bdt_data.parquet")) for year in years]
        )


def main(args):
    global bdtVars

    early_stopping_callback = xgb.callback.EarlyStopping(
        rounds=args.early_stopping_rounds, min_delta=args.early_stopping_min_delta
    )

    classifier_params = {
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "learning_rate": 0.1,
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
                (
                    f"{year} {key} Yield: "
                    f'{np.sum(data[data["Dataset"] == key][weight_key])}, '
                    "Number of Events: "
                    f'{len(data[data["Dataset"] == key])}, '
                )
            )

        bg_select = np.sum(
            [data["Dataset"] == key for key in bg_keys],
            axis=0,
        ).astype(bool)

        print(
            f"Total BG Yield: {np.sum(data[bg_select][weight_key])}, Number of Events: {len(data[bg_select])}"
        )

    if not args.inference_only:
        training_data_dict = {
            year: data[
                # select only signal and `bg_keys` backgrounds for training - rest are only inferenced
                np.sum(
                    [data["Dataset"] == key for key in training_keys],
                    axis=0,
                ).astype(bool)
            ]
            for year, data in data_dict.items()
        }

        training_samples = np.unique(list(training_data_dict.values())[0]["Dataset"])
        print("Training samples:", training_samples)

        if args.test:
            # get a sample of different processes
            data_dict = {
                year: pd.concat(
                    (data[:50], data[1000000:1000050], data[2000000:2000050], data[-50:]), axis=0
                )
                for year, data in data_dict.items()
            }
            # 100 signal, 100 bg events
            training_data_dict = {
                year: pd.concat(
                    (
                        data[:150],
                        data[
                            np.sum(data["Dataset"] == sig_key) : np.sum(data["Dataset"] == sig_key)
                            + 50
                        ],
                        data[
                            np.sum(data["Dataset"] == "V+Jets")
                            - 50 : np.sum(data["Dataset"] == "V+Jets")
                        ],
                        data[-50:],
                    ),
                    axis=0,
                )
                for year, data in training_data_dict.items()
            }

        if args.equalize_weights or args.equalize_weights_per_process:
            for year, data in training_data_dict.items():
                for key in training_keys:
                    print(
                        (
                            f"Pre-equalization {year} {key} total: "
                            f'{np.sum(data[data["Dataset"] == key][weight_key])}'
                        )
                    )

                equalize_weights(data, args.equalize_weights, args.equalize_weights_per_process)

                for key in training_keys:
                    print(
                        (
                            f"Post-equalization {year} {key} total: "
                            f'{np.sum(data[data["Dataset"] == key][weight_key])}'
                        )
                    )

                print("")

        if len(training_samples) > 0:
            train, test = {}, {}
            for year, data in training_data_dict.items():
                train[year], test[year] = train_test_split(
                    remove_neg_weights(data) if not args.absolute_weights else data,
                    test_size=args.test_size,
                    random_state=args.seed,
                )

    if args.evaluate_only or args.inference_only:
        model = xgb.XGBClassifier()
        model.load_model(f"{args.model_dir}/trained_bdt.model")
    else:
        os.system(f"mkdir -p {args.model_dir}")
        model = train_model(
            get_X(train),
            get_X(test),
            get_Y(train, args.multiclass),
            get_Y(test, args.multiclass),
            get_weights(train, args.absolute_weights),
            get_weights(test, args.absolute_weights),
            args.model_dir,
            use_sample_weights=args.use_sample_weights,
            **classifier_params,
        )

    if not args.inference_only:
        evaluate_model(model, args.model_dir, train, test, multiclass=args.multiclass)

    if not args.evaluate_only:
        do_inference(model, args.model_dir, data_dict, multiclass=args.multiclass)


def _plot_losses(trained_model: xgb.XGBClassifier, model_dir: str):
    evals_result = trained_model.evals_result()

    with open(f"{model_dir}/evals_result.txt", "w") as f:
        f.write(str(evals_result))

    fig = plt.figure(figsize=(10, 8))
    for i, label in enumerate(["Train", "Test"]):
        plt.plot(evals_result[f"validation_{i}"]["mlogloss"], label=label, linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{model_dir}/losses.pdf", bbox_inches="tight")
    plt.close()


def train_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    weights_train: np.ndarray,
    weights_test: np.ndarray,
    model_dir: str,
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
    trained_model.save_model(f"{model_dir}/trained_bdt.model")
    _plot_losses(trained_model, model_dir)
    return model


def _txbb_thresholds(test: Dict[str, pd.DataFrame], txbb_threshold: float):
    t = []
    for year, data in test.items():
        t.append(data["bbFatJetParticleNetMD_Txbb"] < txbb_threshold)

    return pd.concat(t, axis=0)


def evaluate_model(
    model: xgb.XGBClassifier,
    model_dir: str,
    train: Dict[str, pd.DataFrame],
    test: Dict[str, pd.DataFrame],
    txbb_threshold: float = 0.98,
    multiclass: bool = False,
):
    """ """
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
    rocs = {}

    for data, label in [(train, "train"), (test, "test")]:
        save_model_dir = f"{model_dir}/rocs_{label}/"
        os.makedirs(save_model_dir, exist_ok=True)

        Y = get_Y(data)
        weights_test = get_weights(data)

        preds = model.predict_proba(get_X(data))
        preds = preds[:, 0] if multiclass else preds[:, 1]

        sig_effs = [0.15, 0.2]

        fpr, tpr, thresholds = roc_curve(Y, preds, sample_weight=weights_test)

        rocs[label] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
        }

        with open(f"{save_model_dir}/roc_dict.pkl", "wb") as f:
            pickle.dump(rocs[label], f)

        plotting.rocCurve(
            fpr,
            tpr,
            # auc(fpr, tpr),
            sig_eff_lines=sig_effs,
            title=None,
            plotdir=save_model_dir,
            name="bdtroc",
        )

        for sig_eff in sig_effs:
            thresh = thresholds[np.searchsorted(tpr, sig_eff)]
            print(f"Threshold at {sig_eff} sig_eff: {thresh:0.4f}")

        if txbb_threshold > 0:
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
                plotdir=save_model_dir,
                name="bdtroc_txbb_cut",
            )

            np.savetxt(f"{save_model_dir}/fpr_txbb_threshold.txt", fpr_txbb_threshold)
            np.savetxt(f"{save_model_dir}/tpr_txbb_threshold.txt", tpr_txbb_threshold)
            np.savetxt(f"{save_model_dir}/thresholds_txbb_threshold.txt", thresholds_txbb_threshold)

            for sig_eff in sig_effs:
                thresh = thresholds_txbb_threshold[np.searchsorted(tpr_txbb_threshold, sig_eff)]
                print(f"Incl Txbb Threshold at {sig_eff} sig_eff: {thresh:0.4f}")

    # combined ROC curve with thresholds
    xlim = [0, 1]
    ylim = [1e-6, 1]
    sig_effs = [0.15, 0.2]
    line_style = {"colors": "lightgrey", "linestyles": "dashed"}

    plt.figure(figsize=(12, 12))
    for inf, label in [("train", "Training Data"), ("test", "Testing Data")]:
        roc = rocs[inf]

        plt.plot(
            roc["tpr"],
            roc["fpr"],
            label=label,
            linewidth=2,
        )

        for sig_eff in sig_effs:
            y = roc["fpr"][np.searchsorted(roc["tpr"], sig_eff)]
            plt.hlines(y=y, xmin=0, xmax=sig_eff, **line_style)
            plt.vlines(x=sig_eff, ymin=0, ymax=y, **line_style)

    hep.cms.label(data=False, rlabel="")
    plt.yscale("log")
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background efficiency")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend(loc="upper left")
    plt.savefig(f"{model_dir}/roc.pdf", bbox_inches="tight")
    plt.close()


def do_inference(
    model: xgb.XGBClassifier,
    model_dir: str,
    data_dict: Dict[str, pd.DataFrame],
    jec_jmsr_shifts: bool = True,
    multiclass: bool = False,
):
    """ """
    import time

    os.system(f"mkdir -p {model_dir}/inferences/")

    for year, data in data_dict.items():
        year_data_dict = {year: data}
        os.system(f"mkdir -p {model_dir}/inferences/{year}")

        sample_order = list(pd.unique(data["Dataset"]))
        value_counts = data["Dataset"].value_counts()
        sample_order_dict = OrderedDict([(sample, value_counts[sample]) for sample in sample_order])

        with open(f"{model_dir}/inferences/{year}/sample_order.txt", "w") as f:
            f.write(str(sample_order_dict))

        print("Running inference")
        X = get_X(year_data_dict)
        model.get_booster().feature_names = bdtVars

        start = time.time()
        preds = model.predict_proba(X)
        print(f"Finished in {time.time() - start:.2f}s")
        preds = preds[:, :-1] if multiclass else preds[:, 1]  # save n-1 probs to save space
        np.save(f"{model_dir}/inferences/{year}/preds.npy", preds)

        if jec_jmsr_shifts:
            for jshift in jec_shifts:
                print("Running inference for", jshift)
                X, mcvars = get_X(year_data_dict, jec_shift=jshift)
                # have to change model's feature names since we're passing in a dataframe
                model.get_booster().feature_names = mcvars
                preds = model.predict_proba(X)
                preds = preds[:, :-1] if multiclass else preds[:, 1]
                np.save(f"{model_dir}/inferences/{year}/preds_{jshift}.npy", preds)

            for jshift in jmsr_shifts:
                print("Running inference for", jshift)
                X, mcvars = get_X(year_data_dict, jmsr_shift=jshift)
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
    utils.add_bool_arg(parser, "load-data", "Load pre-processed data if done already", default=True)
    utils.add_bool_arg(
        parser, "save-data", "Save pre-processed data if loading the data", default=True
    )

    parser.add_argument(
        "--num-events",
        default=0,
        help="Num events per sample to train on - if 0 train on all",
        type=int,
    )

    parser.add_argument("--max-depth", default=6, help="xgboost param", type=int)
    parser.add_argument("--min-child-weight", default=1, help="xgboost param", type=int)
    parser.add_argument("--n-estimators", default=1000, help="xgboost param", type=int)

    parser.add_argument("--rem-feats", default=3, help="remove N lowest importance feats", type=int)

    utils.add_bool_arg(parser, "multiclass", "Classify each background separtely", default=True)

    utils.add_bool_arg(
        parser, "use-sample-weights", "Use properly scaled event weights", default=True
    )
    utils.add_bool_arg(
        parser,
        "absolute-weights",
        "Use absolute weights if using sample weights (if false, will remove negative weights)",
        default=True,
    )
    utils.add_bool_arg(
        parser, "equalize-weights", "Equalise total signal and background weights", default=True
    )
    utils.add_bool_arg(
        parser,
        "equalize-weights-per-process",
        "Equalise each backgrounds' weights too",
        default=False,
    )

    parser.add_argument(
        "--early-stopping-rounds", default=5, help="early stopping rounds", type=int
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        default=0.0,
        help="min abs improvement needed for early stopping",
        type=float,
    )
    parser.add_argument("--test-size", default=0.3, help="testing/training split", type=float)
    parser.add_argument("--seed", default=4, help="seed for testing/training split", type=int)

    utils.add_bool_arg(parser, "evaluate-only", "Only evaluation, no training", default=False)
    utils.add_bool_arg(parser, "inference-only", "Only inference, no training", default=False)
    utils.add_bool_arg(
        parser, "test", "Testing BDT Training - run on a small sample", default=False
    )

    args = parser.parse_args()

    if args.equalize_weights or args.equalize_weights_per_process:
        args.use_sample_weights = True  # sample weights are used before equalizing

    args.all_years = True if args.year == "all" else False

    main(args)
