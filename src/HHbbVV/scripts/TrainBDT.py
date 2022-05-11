"""
Takes the skimmed parquet files (output of bbVVSkimmer + YieldsAnalysis.py) and trains a BDT using xgboost.

Author(s): Raghav Kansal
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import xgboost as xgb

import utils
import plotting

from sample_labels import sig_key


weight_key = "finalWeight"

bdtVars = [
    "MET_pt",
    "DijetEta",
    "DijetPt",
    "DijetMass",
    "bbFatJetPt",
    "VVFatJetEta",
    "VVFatJetPt",
    "VVFatJetMsd",
    # "VVFatJetParticleNetHWWMD_THWW4q",
    "VVFatJetParticleNetHWWMD_probQCD",
    "VVFatJetParticleNetHWWMD_probHWW3q",
    "VVFatJetParticleNetHWWMD_probHWW4q",
    "bbFatJetPtOverDijetPt",
    "VVFatJetPtOverDijetPt",
    "VVFatJetPtOverbbFatJetPt",
]


def get_X(data: pd.DataFrame):
    return data.filter(items=bdtVars)


def get_Y(data: pd.DataFrame):
    return (data["Dataset"] == sig_key).astype(int)


def get_weights(data: pd.DataFrame, equalize_weights: bool = False):
    return data[weight_key]


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
    print("Setup")

    classifier_params = {
        "max_depth": 3,
        "learning_rate": 0.1,
        "n_estimators": 400,
        "verbosity": 2,
        "n_jobs": 4,
        "reg_lambda": 1.0,
    }

    training_data = pd.read_parquet(args.data_path)

    if args.test:
        training_data = pd.concat((training_data[:100], training_data[-100:]), axis=0)

    if args.equalize_weights:
        equalize_weights(training_data)

    train, test = train_test_split(training_data, test_size=args.test_size, random_state=args.seed)

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
            get_weights(train),
            get_weights(test),
            args.model_dir,
            use_sample_weights=args.use_sample_weights,
            early_stopping_rounds=args.early_stopping_rounds,
            **classifier_params,
        )

    if not args.inference_only:
        evaluate_model(model, args.model_dir, test)

    if not args.evaluate_only:
        do_inference(model, args.model_dir, training_data)


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

    # sorting by importance
    feature_importance = np.stack((bdtVars, model.feature_importances_)).T[
        np.argsort(model.feature_importances_)[::-1]
    ]

    np.save(f"{model_dir}/feature_importance.np", feature_importance)

    print("Feature importance")
    for feature, imp in feature_importance:
        print(f"{feature}: {imp}")

    fpr, tpr, thresholds = roc_curve(Y_test, preds[:, 1], sample_weight=weights_test)
    plotting.rocCurve(fpr, tpr, title="ROC Curve", plotdir=model_dir, name="bdtroccurve")

    np.savetxt(f"{model_dir}/fpr.txt", fpr)
    np.savetxt(f"{model_dir}/tpr.txt", tpr)
    np.savetxt(f"{model_dir}/thresholds.txt", thresholds)

    if txbb_threshold > 0:
        preds_txbb_threshold = preds[:, 1].copy()
        preds_txbb_threshold[test["bbFatJetParticleNetMD_Txbb"] < txbb_threshold] = 0

        fpr_txbb_threshold, tpr_txbb_threshold, thresholds_txbb_threshold = roc_curve(
            Y_test, preds_txbb_threshold, sample_weight=weights_test
        )

        fpr_txbb_threshold[np.argmin(np.abs(tpr_txbb_threshold - 0.15))]
        thresholds_txbb_threshold[np.argmin(np.abs(tpr_txbb_threshold - 0.15))]

        plotting.rocCurve(
            fpr_txbb_threshold,
            tpr_txbb_threshold,
            title=f"ROC Curve Including Txbb {txbb_threshold} Cut",
            plotdir=model_dir,
            name="bdtroccurve_txbb_cut",
        )

        np.savetxt(f"{model_dir}/fpr_txbb_threshold.txt", fpr_txbb_threshold)
        np.savetxt(f"{model_dir}/tpr_txbb_threshold.txt", tpr_txbb_threshold)
        np.savetxt(f"{model_dir}/thresholds_txbb_threshold.txt", thresholds_txbb_threshold)


def do_inference(
    model: xgb.XGBClassifier,
    model_dir: str,
    data: pd.DataFrame,
):
    """ """
    print("Running inference")
    preds = model.predict_proba(get_X(data))
    np.save(f"{model_dir}/preds.npy", preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path",
        default="/hhbbvvvol/data/2017/bdt_training_data.parquet",
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
        parser, "use-sample-weights", "Use properly scaled event weights", default=False
    )
    utils.add_bool_arg(
        parser, "equalize-weights", "Equalise signal and background weights", default=False
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
