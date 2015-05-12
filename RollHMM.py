from tabulate import tabulate
import TestStrat as ts
import Metrics as m
import MLData as mld
import FeatureFinder as ff
import MLCombinedStrats as mcs
import StratAnalysis as sa
import numpy as np
import sklearn, warnings, random
from sklearn.hmm import GaussianHMM
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Perceptron, Ridge
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# SETTINGS ####################
HOLDING_DAYS      = 20
LOOKBACK_WINDOW   = 252
n_components      = 5
perform_pca       = False
pca_components    = 20
################################

# Obtain Data
d      = mld.getData()
X, y, FeatureNames, returns, actual_y = mld.createMLData(HOLDING_DAYS)
dates  = d.dates[-len(y):]
values = d.sub_data[-len(y):,0]

top_features = ff.getTopFeatures(Ridge(alpha=0.0001, normalize=True, fit_intercept=False),30,False)
X            = X[:,top_features]
FeatureNames = np.array(FeatureNames)[top_features]

# Scale the data for training and PCA
if perform_pca:
    scaler = StandardScaler()
    X      = scaler.fit_transform(X)

def predictWithHMM(index, window = 252):
    training_X = X[range(index-window,index),:]
    training_y = actual_y[range(index-window,index)]
    testing_X  = X[index,:].reshape(1,training_X.shape[1])
    testing_y  = y[index]

    # PCA DATA
    if perform_pca:
        pca        = PCA(n_components= pca_components)
        pca.fit(training_X)
        training_X = pca.transform(training_X)
        testing_X  = pca.transform(testing_X)


    model = GaussianHMM(n_components, "diag",n_iter=1000)
    model.fit([training_X])

    hidden_states          = model.predict(training_X)
    predicted_hidden_state = model.predict(testing_X)

    # DO PROBALISTIC APPROACH
    # pr = model.predict_proba(testing_X)
    # print pr

    prob = 0
    state_idx  = (hidden_states == predicted_hidden_state)
    median_val = np.mean(training_y[state_idx])

    return int(median_val>0), testing_y, prob

def runRollingHMMPrediction(window=252):
    y_pred = []
    y_true = []
    indices = range(window, X.shape[0])
    for index in indices:
        print float(index-window+1)/(X.shape[0]-window+1), '%'
        y_p, y_t, prob = predictWithHMM(index,window)
        y_pred.append(y_p)
        y_true.append(y_t)

    return np.array(y_pred), np.array(y_true), []

y_pred, y_true, probs = runRollingHMMPrediction(LOOKBACK_WINDOW)


window_length = y_pred.shape[0]

returns        = returns[-window_length:]
dates          = d.dates[-window_length:]

CURRENT_PRED = y_pred[0]

for i in range(y_pred.shape[0]):
    if i%HOLDING_DAYS == 0:
        CURRENT_PRED = y_pred[i]
    else:
         y_pred[i]  = CURRENT_PRED


ts.runBackTest(y_true, y_pred, probs, returns, dates, True, "HMM")


