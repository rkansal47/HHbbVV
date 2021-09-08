"""
Takes the skimmed pickles (output of bbVVSkimmer) and trains a BDT using xgboost.

Author(s): Raghav Kansal
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import xgboost as xgb
import sys

# import utils
# import plotting
# import matplotlib.pyplot as plt

# data_path = '../../data/2017_combined/'
data_path = sys.argv[1]
model_dir = sys.argv[2]

import os
os.system(f'mkdir -p {model_dir}')

# import importlib
# importlib.reload(utils)
# importlib.reload(plotting)


##################################################################################
# Load and process data
##################################################################################

# # backgrounds listed first and plotted in order
# keys = ['V', 'Top', 'QCD', 'HHbbVV4q']
# labels = ['VV/V+jets', 'ST/TT', 'QCD', 'HHbbVV4q']
# num_bg = 3  # up to this label for bg
# sig = 'HHbbVV4q'
#
# import pickle
#
# events = {}
# for key in keys:
#     # if key != sig: continue
#     print(key)
#     with open(f'{data_path}{key}.pkl', 'rb') as file:
#         events[key] = pickle.load(file)['skimmed_events']
#
# # Just for checking
# for key in keys:
#     print(f"{key} events: {np.sum(events[key]['finalWeight']):.2f}")
#
# # Features to use for BDT
# bdtVars = [
#     'MET_pt',
#
#     'DijetEta',
#     'DijetPt',
#     'DijetMass',
#
#     'bbFatJetPt',
#
#     'VVFatJetEta',
#     'VVFatJetPt',
#     'VVFatJetMsd',
#     'VVFatJetParticleNet_Th4q',
#
#     'bbFatJetPtOverDijetPt',
#     'VVFatJetPtOverDijetPt',
#     'VVFatJetPtOverbbFatJetPt',
# ]
#
# # NUM_EVENTS = 10000
# TEST_SIZE = 0.3
# SEED = 4
#
# X = np.concatenate([np.concatenate([events[key][var][:, np.newaxis] for var in bdtVars], axis=1) for key in keys], axis=0)
# Y = np.concatenate((np.concatenate([np.zeros_like(events[key]['weight'][:]) for key in keys[:num_bg]]),
#                     np.ones_like(events[sig]['weight'][:])))
# weights = np.concatenate([events[key]['finalWeight'][:] for key in keys])
# X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, Y, weights, test_size=TEST_SIZE, random_state=SEED)
#
#
# save_dir = '../../data/2017_bdt_training/'
# np.save(f'{save_dir}/X_train.npy', X_train)
# np.save(f'{save_dir}/X_test.npy', X_test)
# np.save(f'{save_dir}/y_train.npy', y_train)
# np.save(f'{save_dir}/y_test.npy', y_test)
# np.save(f'{save_dir}/weights_train.npy', weights_train)
# np.save(f'{save_dir}/weights_test.npy', weights_test)


##################################################################################
# Load pre-processed data
##################################################################################

X_train = np.load(f'{data_path}/X_train.npy')
X_test = np.load(f'{data_path}/X_test.npy')
y_train = np.load(f'{data_path}/y_train.npy')
y_test = np.load(f'{data_path}/y_test.npy')

##################################################################################
# Train model
##################################################################################

model = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=400, verbosity=2, n_jobs=4, reg_lambda=1.0)
trained_model = model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)])  # , sample_weight=weights_train)
trained_model.save_model(f'{model_dir}/bdt.model')
