"""
Takes the skimmed pickles (output of bbVVSkimmer) and trains a BDT using xgboost.

Author(s): Raghav Kansal
"""

import numpy as np

import utils
import plotting

# load the data

import pickle

# backgrounds listed first and plotted in order
keys = ['V', 'Top', 'QCD', 'HHbbVV4q']
labels = ['VV/V+jets', 'ST/TT', 'QCD', 'HHbbVV4q']
num_bg = 3  # up to this label for bg
sig = 'HHbbVV4q'
data_path = '../../data/2017_combined/'

plotdir = '../plots/BDTPlots/'

# import os
# os.system(f'mkdir -p {plotdir}')


# import importlib
# importlib.reload(utils)
# importlib.reload(plotting)

events = {}

for key in keys:
    # if key != sig: continue
    print(key)
    with open(f'{data_path}{key}.pkl', 'rb') as file:
        events[key] = pickle.load(file)['skimmed_events']

events[key]

for key in keys:
    print(f"{key} events: {np.sum(events[key]['finalWeight']):.2f}")

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

key

NUM_EVENTS = 10000
TEST_SIZE = 0.3
SEED = 4

X = np.concatenate([np.concatenate([events[key][var][:NUM_EVENTS, np.newaxis] for var in bdtVars], axis=1) for key in keys], axis=0)
Y = np.concatenate((np.concatenate([np.zeros_like(events[key]['weight'][:NUM_EVENTS]) for key in keys[:num_bg]]),
                    np.ones_like(events[sig]['weight'][:NUM_EVENTS])))
weights = np.concatenate([events[key]['finalWeight'][:NUM_EVENTS] for key in keys])


from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import roc_curve

x_train, x_test, y_train, y_test, weights_train, weights_test = train_test_split(X, Y, weights, test_size=TEST_SIZE, random_state=SEED)

model = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=400, verbosity=2, n_jobs=4, reg_lambda=1.0)
model.fit(x_train, y_train, sample_weight=weights_train)


model



preds = model.predict(x_test)

preds

roc_curve(y_test, preds)
