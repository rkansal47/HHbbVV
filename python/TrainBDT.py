"""
Takes the skimmed pickles (output of bbVVSkimmer) and trains a BDT using xgboost.

Author(s): Raghav Kansal
"""

import argparse
import os
from os.path import exists

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import xgboost as xgb

import utils
import plotting
# import matplotlib.pyplot as plt


# import importlib
# importlib.reload(utils)
# importlib.reload(plotting)


# backgrounds listed first and plotted in order
keys = ['V', 'Top', 'QCD', 'HHbbVV4q']
labels = ['VV/V+jets', 'ST/TT', 'QCD', 'HHbbVV4q']
num_bg = 3  # up to this label for bg
sig = 'HHbbVV4q'


bdtVars = [
    'MET_pt',

    'DijetEta',
    'DijetPt',
    'DijetMass',

    'bbFatJetPt',

    'VVFatJetEta',
    'VVFatJetPt',
    'VVFatJetMsd',
    'VVFatJetParticleNet_Th4q',

    'bbFatJetPtOverDijetPt',
    'VVFatJetPtOverDijetPt',
    'VVFatJetPtOverbbFatJetPt',
]


def main(args):
    data_path = f'{args.data_dir}/{"all_" if args.num_events <= 0 else args.num_events}_events_{"" if args.preselection else "no_"}preselection_test_size_0{args.test_size * 10:.0f}_seed_{args.seed}'

    classifier_params = {
        'max_depth': 3,
        'learning_rate': 0.1,
        'n_estimators': 400,
        'verbosity': 2,
        'n_jobs': 4,
        'reg_lambda': 1.0
    }

    if args.load_data and exists(data_path):
        X_train, X_test, X_Txbb_train, X_Txbb_test, y_train, y_test, weights_train, weights_test = load_training_data(data_path)
    else:
        os.system(f'mkdir -p {data_path}')
        events = load_events(num_events=args.num_events, preselection=args.preselection)
        X_train, X_test, X_Txbb_train, X_Txbb_test, y_train, y_test, weights_train, weights_test = preprocess_events(events, bdtVars, test_size=args.test_size, seed=args.seed, save=True, save_dir=data_path)

    if not args.evaluate_only:
        os.system(f'mkdir -p {data_path}')
        model = train_model(X_train, X_test, y_train, y_test, args.model_dir, **classifier_params)
    else:
        model = xgb.XGBClassifier()
        model.load_model(f'{args.model_dir}/trained_bdt.model')

    evaluate_model(model, args.model_dir, X_test, y_test, weights_test)


def load_events(pickles_path: str, num_events: int = 0, preselection: bool = True):
    """
    Loads events from pickles.
    If `num_events` > 0, only returns `num_events` entries for each sample.
    If `preselection` is True, applies a preselection as defined below
    """
    import pickle

    events = {}
    for key in keys:
        # if key != sig: continue
        print(key)
        with open(f'{pickles_path}{key}.pkl', 'rb') as file:
            events[key] = pickle.load(file)['skimmed_events']

    if num_events > 0:
        for key in keys:
            for var in events[key].keys():
                events[key][var] = events[key][var][:num_events]

    if preselection:
        for key in keys:
            cut = (events[key]['bbFatJetParticleNetMD_Txbb'] > 0.8) * (events[key]['bbFatJetMsd'] > 50) * (events[key]['bbFatJetMsd'] > 50)
            for var in events[key].keys():
                events[key][var] = events[key][var][cut]

    # Just for checking
    for key in keys:
        print(f"{key} events: {np.sum(events[key]['finalWeight']):.2f}")

    return events


# bdt_presel_cut = {}
#
# for key in keys:
#     bdt_presel_cut[key] = (events[key]['bbFatJetParticleNetMD_Txbb'] > 0.8) * (events[key]['bbFatJetMsd'] > 50) * (events[key]['bbFatJetMsd'] > 50)
#
# # Just for checking
# print("Post bdt preselection")
# for key in keys:
#     print(f"{key} events: {np.sum(events[key]['finalWeight'][bdt_presel_cut[key]]):.2f}")
# # For checking again 4b AN
# bbbb_presel_cut = {}
#
# for key in keys:
#     bbbb_presel_cut[key] = (events[key]['ak8FatJetPt'][:, 0] > 300) * (events[key]['ak8FatJetPt'][:, 1] > 300) * (events[key]['ak8FatJetMsd'][:, 0] > 50) * (events[key]['ak8FatJetMsd'][:, 1] > 50) * (events[key]['ak8FatJetParticleNetMD_Txbb'][:, 0] > 0.8)
#
# print("Post 4b preselection")
# for key in keys:
#     print(f"{key} events: {np.sum(events[key]['finalWeight'][bbbb_presel_cut[key]]):.2f}")
# Features to use for BDT


def preprocess_events(events: dict, bdtVars: list, test_size: float = 0.3, seed: int = 4, save: bool = True, save_dir: str = ""):
    """ Preprocess events for training """
    # NUM_EVENTS = 10000
    TEST_SIZE = 0.3
    SEED = 4

    # X = np.concatenate([np.concatenate([events[key][var][bdt_presel_cut[key]][:, np.newaxis] for var in bdtVars], axis=1) for key in keys], axis=0)
    # X_Txbb = np.concatenate([events[key]['bbFatJetParticleNetMD_Txbb'][bdt_presel_cut[key]] for key in keys])  # for the final ROC curve
    # Y = np.concatenate((np.concatenate([np.zeros_like(events[key]['weight'][bdt_presel_cut[key]][:]) for key in keys[:num_bg]]),
    #                     np.ones_like(events[sig]['weight'][bdt_presel_cut[key]][:])))
    # weights = np.concatenate([events[key]['finalWeight'][bdt_presel_cut[key]][:] for key in keys])

    X = np.concatenate([np.concatenate([events[key][var][:, np.newaxis] for var in bdtVars], axis=1) for key in keys], axis=0)
    X_Txbb = np.concatenate([events[key]['bbFatJetParticleNetMD_Txbb'] for key in keys])  # for the final ROC curve
    Y = np.concatenate((np.concatenate([np.zeros_like(events[key]['weight'][:]) for key in keys[:num_bg]]),
                        np.ones_like(events[sig]['weight'][:])))
    weights = np.concatenate([events[key]['finalWeight'][:] for key in keys])

    X_train, X_test, X_Txbb_train, X_Txbb_test, y_train, y_test, weights_train, weights_test = train_test_split(X, X_Txbb, Y, weights, test_size=test_size, random_state=seed)

    if save:
        np.save(f'{save_dir}/X_train.npy', X_train)
        np.save(f'{save_dir}/X_test.npy', X_test)
        np.save(f'{save_dir}/X_Txbb_train.npy', X_Txbb_train)
        np.save(f'{save_dir}/X_Txbb_test.npy', X_Txbb_test)
        np.save(f'{save_dir}/y_train.npy', y_train)
        np.save(f'{save_dir}/y_test.npy', y_test)
        np.save(f'{save_dir}/weights_train.npy', weights_train)
        np.save(f'{save_dir}/weights_test.npy', weights_test)

    return X_train, X_test, X_Txbb_train, X_Txbb_test, y_train, y_test, weights_train, weights_test


def load_training_data(data_path: str):
    """ Load pre-processed data directly if already saved """
    X_train = np.load(f'{data_path}/X_train.npy')
    X_test = np.load(f'{data_path}/X_test.npy')
    X_Txbb_train = np.load(f'{data_path}/X_Txbb_train.npy')
    X_Txbb_test = np.load(f'{data_path}/X_Txbb_test.npy')
    y_train = np.load(f'{data_path}/y_train.npy')
    y_test = np.load(f'{data_path}/y_test.npy')
    weights_train = np.load(f'{data_path}/weights_train.npy')
    weights_test = np.load(f'{data_path}/weights_test.npy')
    return X_train, X_test, X_Txbb_train, X_Txbb_test, y_train, y_test, weights_train, weights_test


def train_model(X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array, model_dir: str, early_stopping_rounds: int = 5, **classifier_params):
    """ Trains BDT. `classifier_params` are hyperparameters for the classifier """
    model = xgb.XGBClassifier(**classifier_params)
    trained_model = model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)])  # , sample_weight=weights_train)
    trained_model.save_model(f'{model_dir}/trained_bdt.model')
    return model


def evaluate_model(model: xgb.XGBClassifier, model_dir: str, X_test: np.array, y_test: np.array, weights_test: np.array, txbb_threshold: float = 0.98):
    """ """
    preds = model.predict_proba(X_test)

    # sorting by importance
    feature_importance = np.array(list(zip(bdtVars, model.feature_importances_)))[np.argsort(model.feature_importances_)[::-1]]
    np.savetxt(f"{model_dir}/feature_importance.txt", )

    print("Feature importance")
    for feature, imp in feature_importance:
        print(f"{feature}: {imp}")


    fpr, tpr, thresholds = roc_curve(y_test, preds[:, 1], sample_weight=weights_test)
    plotting.rocCurve(fpr, tpr, title="ROC Curve", plotdir=model_dir, name="bdtroccurve")

    np.savetxt(f"{model_dir}/fpr", fpr)
    np.savetxt(f"{model_dir}/tpr", tpr)
    np.savetxt(f"{model_dir}/thresholds", thresholds)

    if txbb_threshold:
        preds_txbb_threshold = preds[:, 1].copy()
        preds_txbb_threshold[X_test < txbb_threshold] = 0

        fpr_txbb_threshold, tpr_txbb_threshold, thresholds_txbb_threshold = roc_curve(y_test, preds_txbb_threshold, sample_weight=weights_test)

        fpr_txbb_threshold[np.argmin(np.abs(tpr_txbb_threshold - 0.15))]
        thresholds_txbb_threshold[np.argmin(np.abs(tpr_txbb_threshold - 0.15))]

        plotting.rocCurve(fpr_txbb_threshold, tpr_txbb_threshold, title="ROC Curve after Txbb {txbb_threshold} Cut", plotdir=model_dir, name="bdtroccurve_txbb_cut")

        np.savetxt(f"{model_dir}/fpr_txbb_threshold", fpr_txbb_threshold)
        np.savetxt(f"{model_dir}/tpr_txbb_threshold", tpr_txbb_threshold)
        np.savetxt(f"{model_dir}/thresholds_txbb_threshold", thresholds_txbb_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-dir', default="./", help="directory in which to save model and evaluation output", type=str)

    parser.add_argument('--num-events', default=0, help="Num events per sample to train on - if 0 train on all", type=int)
    utils.add_bool_arg(parser, "preselection", "Apply preselection on events before training", default=True)

    parser.add_argument('--test-size', default=0.3, help="testing/training split", type=float)
    parser.add_argument('--seed', default=4, help="seed for testing/training split", type=int)

    utils.add_bool_arg(parser, "evaluate-only", "Only evaluation, no training", default=False)

    args = parser.parse_args()
    main(args)
