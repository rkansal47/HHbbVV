"""
Takes the skimmed parquet files (output of bbVVSkimmer + YieldsAnalysis.py) and trains a BDT using xgboost.

Author(s): Raghav Kansal
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import xgboost as xgb

import utils
import plotting

from hh_vars import sig_key, data_key, jec_shifts, jmsr_shifts, jec_vars, jmsr_vars

from copy import deepcopy

weight_key = "finalWeight"

# only vars used for training
# TODO: Change VV msd to regressed mass?
bdtVars = [
    "MET_pt",
    "DijetEta",
    "DijetPt",
    "DijetMass",
    "bbFatJetPt",
    "VVFatJetEta",
    "VVFatJetPt",
    "VVFatJetMsd",
    # "VVFatJetParTMD_THWW4q",
    "VVFatJetParTMD_probQCD",
    "VVFatJetParTMD_probT",
    "VVFatJetParTMD_probHWW3q",
    "VVFatJetParTMD_probHWW4q",
    "bbFatJetPtOverDijetPt",
    "VVFatJetPtOverDijetPt",
    "VVFatJetPtOverbbFatJetPt",
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
    "VVFatJetMsd": ([50, 0, 300], r"$m^{VV}_{msd}$ (GeV)"),
    "VVFatJetParTMD_probT": ([50, 0, 1], r"ParT $Prob(Top)^{VV}$"),
    "VVFatJetParTMD_probQCD": ([50, 0, 1], r"ParT $Prob(QCD)^{VV}$"),
    "VVFatJetParTMD_probHWW3q": ([50, 0, 1], r"ParT $Prob(HWW3q)^{VV}$"),
    "VVFatJetParTMD_probHWW4q": ([50, 0, 1], r"ParT $Prob(HWW4q)^{VV}$"),
    "bbFatJetPtOverDijetPt": ([50, 0, 40], r"$p^{bb}_T / p_T^{jj}$"),
    "VVFatJetPtOverDijetPt": ([50, 0, 40], r"$p^{VV}_T / p_T^{jj}$"),
    "VVFatJetPtOverbbFatJetPt": ([50, 0.4, 2.0], r"$p^{VV}_T / p^{bb}_T$"),
}


def get_X(data: pd.DataFrame, jec_shift: str = None, jmsr_shift: str = None):
    """
    Gets variables for BDT for all samples in ``data``.
    Optionally gets shifted variables (in which returns only MC samples).
    """
    if jec_shift is None and jmsr_shift is None:
        return data.filter(items=bdtVars)

    mc_vars = deepcopy(bdtVars)

    if jec_shift is not None:
        for i, var in enumerate(mc_vars):
            if var in jec_vars:
                mc_vars[i] = f"{var}_{jec_shift}"

        # print(data.filter(items=[f"bbFatJetPt_{jec_shift}", f"VVFatJetPt_{jec_shift}"])[data["Dataset"] != "Data"].iloc[0])
        # print(data.filter(items=[f"bbFatJetPt_{jec_shift}", f"VVFatJetPt_{jec_shift}"])[data["Dataset"] != "Data"].iloc[13])
        # print(data.filter(items=[f"bbFatJetPt_{jec_shift}", f"VVFatJetPt_{jec_shift}"])[data["Dataset"] != "Data"].iloc[21])

    if jmsr_shift is not None:
        for i, var in enumerate(mc_vars):
            if var in jmsr_vars:
                mc_vars[i] = f"{var}_{jmsr_shift}"

    return data.filter(items=mc_vars)[data["Dataset"] != "Data"], mc_vars


def get_Y(data: pd.DataFrame):
    return (data["Dataset"] == sig_key).astype(int)


def get_weights(data: pd.DataFrame, abs_weights: bool = True):
    return np.abs(data[weight_key]) if abs_weights else data[weight_key]


def remove_neg_weights(data: pd.DataFrame):
    return data[data[weight_key] > 0]


def equalize_weights(data: pd.DataFrame):
    """Scales signal such that total signal = total background"""
    sig_total = np.sum(data[data["Dataset"] == sig_key][weight_key])
    bg_total = np.sum(data[data["Dataset"] != sig_key][weight_key])

    print(
        ("Pre-equalization sig total: " f'{np.sum(data[data["Dataset"] == sig_key][weight_key])}')
    )

    data[weight_key].loc[data["Dataset"] == sig_key] *= bg_total / sig_total

    print(
        ("Post-equalization sig total: " f'{np.sum(data[data["Dataset"] == sig_key][weight_key])}')
    )


def main(args):
    classifier_params = {
        "max_depth": 3,
        "learning_rate": 0.1,
        "n_estimators": 400,
        "verbosity": 2,
        "n_jobs": 4,
        "reg_lambda": 1.0,
    }

    data = pd.read_parquet(args.data_path)
    training_data = data[
        (data["Dataset"] == sig_key)
        | (data["Dataset"] == "QCD")
        | (data["Dataset"] == "TT")
        | (data["Dataset"] == "V+Jets")
    ]

    print("Training samples: ", np.unique(training_data["Dataset"]))

    if args.test:
        data = pd.concat(
            (data[:50], data[1000000:1000050], data[2000000:2000050], data[-50:]), axis=0
        )
        # 100 signal, 100 bg events
        training_data = pd.concat((training_data[:100], training_data[-100:]), axis=0)

    if args.equalize_weights:
        equalize_weights(training_data)

    train, test = train_test_split(
        remove_neg_weights(training_data) if not args.absolute_weights else training_data,
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
            get_Y(train),
            get_Y(test),
            get_weights(train, args.absolute_weights),
            get_weights(test, args.absolute_weights),
            args.model_dir,
            use_sample_weights=args.use_sample_weights,
            early_stopping_rounds=args.early_stopping_rounds,
            **classifier_params,
        )

    if not args.inference_only:
        evaluate_model(model, args.model_dir, test)

    if not args.evaluate_only:
        do_inference(model, args.model_dir, data, "2017")


def train_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    weights_train: np.ndarray,
    weights_test: np.ndarray,
    model_dir: str,
    use_sample_weights: bool = False,
    early_stopping_rounds: int = 5,
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
        early_stopping_rounds=5,
        eval_set=[(X_test, y_test)],
        sample_weight_eval_set=[weights_test] if use_sample_weights else None,
    )
    trained_model.save_model(f"{model_dir}/trained_bdt.model")
    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    model_dir: str,
    test: pd.DataFrame,
    txbb_threshold: float = 0.98,
):
    """ """
    print("Evaluating model")

    Y_test = get_Y(test)
    weights_test = get_weights(test)

    preds = model.predict_proba(get_X(test))

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

    sig_effs = [0.15, 0.2]

    fpr, tpr, thresholds = roc_curve(Y_test, preds[:, 1], sample_weight=weights_test)
    plotting.rocCurve(
        fpr,
        tpr,
        auc(fpr, tpr),
        sig_eff_lines=sig_effs,
        title="ROC Curve",
        plotdir=model_dir,
        name="bdtroccurve",
    )

    np.savetxt(f"{model_dir}/fpr.txt", fpr)
    np.savetxt(f"{model_dir}/tpr.txt", tpr)
    np.savetxt(f"{model_dir}/thresholds.txt", thresholds)

    for sig_eff in sig_effs:
        thresh = thresholds[np.searchsorted(tpr, sig_eff)]
        print(f"Threshold at {sig_eff} sig_eff: {thresh:0.4f}")

    if txbb_threshold > 0:
        preds_txbb_threshold = preds[:, 1].copy()
        preds_txbb_threshold[test["bbFatJetParticleNetMD_Txbb"] < txbb_threshold] = 0

        fpr_txbb_threshold, tpr_txbb_threshold, thresholds_txbb_threshold = roc_curve(
            Y_test, preds_txbb_threshold, sample_weight=weights_test
        )

        plotting.rocCurve(
            fpr_txbb_threshold,
            tpr_txbb_threshold,
            auc(fpr_txbb_threshold, tpr_txbb_threshold),
            sig_eff_lines=sig_effs,
            title=f"ROC Curve Including Txbb > {txbb_threshold} Cut",
            plotdir=model_dir,
            name="bdtroccurve_txbb_cut",
        )

        np.savetxt(f"{model_dir}/fpr_txbb_threshold.txt", fpr_txbb_threshold)
        np.savetxt(f"{model_dir}/tpr_txbb_threshold.txt", tpr_txbb_threshold)
        np.savetxt(f"{model_dir}/thresholds_txbb_threshold.txt", thresholds_txbb_threshold)

        for sig_eff in sig_effs:
            thresh = thresholds_txbb_threshold[np.searchsorted(tpr_txbb_threshold, sig_eff)]
            print(f"Incl Txbb Threshold at {sig_eff} sig_eff: {thresh:0.4f}")


def do_inference(
    model: xgb.XGBClassifier,
    model_dir: str,
    data: pd.DataFrame,
    year: str,
    jec_jmsr_shifts: bool = True,
):
    """ """
    os.system(f"mkdir -p {model_dir}/inferences/")

    import time

    start = time.time()
    print("Running inference")
    X = get_X(data)
    preds = model.predict_proba(X)[:, 1]
    print(f"Finished in {time.time() - start:.2f}s")
    np.save(f"{model_dir}/inferences/{year}_preds.npy", preds)

    if jec_jmsr_shifts:
        for jshift in jec_shifts:
            print("Running inference for", jshift)
            X, mcvars = get_X(data, jec_shift=jshift)
            # have to change model's feature names since we're passing in a dataframe
            model.get_booster().feature_names = mcvars
            preds = model.predict_proba(X)[:, 1]
            np.save(f"{model_dir}/inferences/{year}_preds_{jshift}.npy", preds)

        for jshift in jmsr_shifts:
            print("Running inference for", jshift)
            X, mcvars = get_X(data, jmsr_shift=jshift)
            # have to change model's feature names since we're passing in a dataframe
            model.get_booster().feature_names = mcvars
            preds = model.predict_proba(X)[:, 1]
            np.save(f"{model_dir}/inferences/{year}_preds_{jshift}.npy", preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path",
        default="/hhbbvvvol/data/2017_bdt_data.parquet",
        help="path to training parquet",
        type=str,
    )
    parser.add_argument(
        "--model-dir",
        default="./",
        help="directory in which to save model and evaluation output",
        type=str,
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
    utils.add_bool_arg(
        parser, "preselection", "Apply preselection on events before training", default=True
    )

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
        parser, "equalize-weights", "Equalise signal and background weights", default=True
    )

    parser.add_argument(
        "--early-stopping-rounds", default=5, help="early stopping rounds", type=int
    )
    parser.add_argument("--test-size", default=0.3, help="testing/training split", type=float)
    parser.add_argument("--seed", default=4, help="seed for testing/training split", type=int)

    utils.add_bool_arg(parser, "evaluate-only", "Only evaluation, no training", default=False)
    utils.add_bool_arg(parser, "inference-only", "Only inference, no training", default=False)
    utils.add_bool_arg(
        parser, "test", "Testing BDT Training - run on a small sample", default=False
    )

    args = parser.parse_args()

    if args.equalize_weights:
        args.use_sample_weights = True  # sample weights are used before equalizing

    main(args)
