
import scipy
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn import metrics
import sklearn, warnings, random
import numpy as np
import Metrics as m
import MLData as mld
import FeatureFinder as ff
import TestStrat as ts
import MLCombinedStrats as mcs
import StratAnalysis as sa
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoLars,ElasticNet, SGDRegressor, lasso_path, enet_path


from sklearn import tree
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import gaussian_process
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy import stats
import time
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import statsmodels.api as sm

import regressionData as regData
from collections import Counter


random.seed(1234)

# define regressions
names = ["OLS", "Ridge", "Lasso", "Lasso Lars", "Elastic Net", "SGD", "Decision Tree", "Random Forest"]

regressors   = np.array([
    LinearRegression(),
  	Ridge(alpha=0.5, normalize=True, fit_intercept=True, max_iter=1000000),
    Lasso(alpha=0.005, max_iter=1000000),
    LassoLars(alpha=0.005),
    ElasticNet(alpha=0.005, l1_ratio=0.5),
    SGDRegressor(alpha=0.005, epsilon=0.1, eta0=0.01),
    tree.DecisionTreeRegressor(),
    RandomForestRegressor(random_state=0,n_estimators=100)
    ])


############################################################################

HOLDING_DAYS    = 10
LOOKBACK_WINDOW = 252
reg_ind         = 0
###############################
# Obtain Data
d = mld.getData()
X, _ , FeatureNames, returns, y = regData.createRegressionData(HOLDING_DAYS)

scaler = StandardScaler()
X      = scaler.fit_transform(X)
# top_features = ff.getTopFeatures(ff.clfs[0],10,False)
# X            = X[:,top_features]
# FeatureNames = np.array(FeatureNames)[top_features]
# Only Correlation Surprise Values
# X = X[:,:15]

def runCurrentPrediction(index, window = 252, reg_ind=1):
    training_X = X[range(index-window,index),:]
    training_y = y[range(index-window,index)]
    testing_X  = X[index,:]
    testing_y  = y[index]
    clf        = regressors[reg_ind]

    ## DO TRANSFORMATION?
    clf.fit(training_X, training_y)
    r_squared = clf.score(training_X, training_y)

    top_features = sa.findPredictiveCharacteristicsReg(clf,FeatureNames, 3) 

    return clf.predict(testing_X), testing_y, r_squared, top_features

def runRollingPrediction(window=252, reg_ind=1):
    y_pred = []
    y_true = []
    r_squared = []
    feats     = []

    indices = range(window, X.shape[0])
    for index in indices:
        y_p, y_t, r_s, feat = runCurrentPrediction(index,window,reg_ind)
        y_pred.append(y_p)
        y_true.append(y_t)
        r_squared.append(r_s)
        feats.append(feat)

        # if proba[0].shape[0] != 2:
        #     if y_p == 1:
        #         proba = np.array([0,1])
        #     else:
        #         proba = np.array([1,0])
        #     probs.append(proba)
        # else:
        #     probs.append(proba[0])

    return np.array(y_pred), np.array(y_true), np.array(r_squared), np.array(feats)


############################################################################


y_pred, y_true, r_squared, reg_top_features = runRollingPrediction(LOOKBACK_WINDOW, reg_ind)



if len(reg_top_features) > 0:
    print Counter(np.ravel(reg_top_features).tolist())




window_length = y_pred.shape[0]

returns        = returns[-window_length:]
dates          = d.dates[-window_length:]

TEST_WINDOW = 252
# ts.runBackTest(y_true, y_pred, [], returns, dates,True)
# ts.runBackTest(y_true[-TEST_WINDOW:], y_pred[-TEST_WINDOW:], [], returns[-TEST_WINDOW:], dates[-TEST_WINDOW:],True)


CURRENT_PRED = y_pred[0]
#CURRENT_PROB = probs[0,:]
for i in range(y_pred.shape[0]):
    if i%HOLDING_DAYS == 0:
        CURRENT_PRED = y_pred[i]
        #CURRENT_PROB = probs[i,:]
    else:
         y_pred[i]  = CURRENT_PRED
        #probs[i,:] = CURRENT_PROB



def convertToClass(ret, threshold = 0):
    if ret >= threshold:
        classification = 1
    else:
        classification = 0
    return (classification)

## High sharpe ratio and precision if raise threshold 


# y_true = np.array([mld.convertToClass(y_i) for y_i in y_true])
# y_pred = np.array([mld.convertToClass(y_i) for y_i in y_pred])


## convert 
y_true = np.array([convertToClass(y_i) for y_i in y_true])

y_pred = np.array([convertToClass(y_i) for y_i in y_pred])


print "r_squared", np.mean(r_squared)
print "accuracy score", metrics.accuracy_score(y_true,y_pred)
print metrics.classification_report(y_true, y_pred)

ts.runBackTest(y_true, y_pred,[], returns, dates,True)




 







