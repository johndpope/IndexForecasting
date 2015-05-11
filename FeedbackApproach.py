import TestStrat as ts
import Metrics as m
import MLData as mld
import FeatureFinder as ff
import MLCombinedStrats as mcs
import StratAnalysis as sa
import numpy as np
import sklearn, warnings, random
from sklearn import cross_validation, linear_model
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron, Ridge
from sklearn.qda import QDA
from sklearn import metrics
from collections import Counter
import matplotlib.pyplot as plt

random.seed(1234)

# Define Classifiers
names = ["KNN","Logistic", "LIN_SVC","RBF_SVM","DTree",
         "RandForest", "AdaBoost", "NB_Gauss", "NB_Multinomial", "Perceptron", "QDA"]

classifiers = np.array([
    KNeighborsClassifier(3),                                                           # 0
    linear_model.LogisticRegression(C=0.1),                                            # 1
    sklearn.svm.LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0),      # 2
    SVC(gamma=0.001, kernel='rbf', C=1, probability=True),                             # 3
    DecisionTreeClassifier(),                                                          # 4
    RandomForestClassifier(n_estimators=15,max_features=1,max_depth=5,random_state=9), # 5
    AdaBoostClassifier(base_estimator = SVC(gamma=0.005, C=1, probability=True), n_estimators=20),   # 6
    GaussianNB(),                                                                                    # 7
    MultinomialNB(alpha=0.01, fit_prior=False),                                                      # 8
    Perceptron(penalty='l2', fit_intercept=False, n_iter=20),                                        # 9
    QDA() # 10
    ])


# SETTINGS ####################
HOLDING_DAYS    = 20
LOOKBACK_WINDOW = 252
CLF_IND         = 3
# 40 day holding, 180 lookback window is good with RF  (5)
# 20 day holding, 252 lookback window is good with SVC (3)
###############################
# Obtain Data
d = mld.getData()
X, y, FeatureNames, returns, actual_y = mld.createMLData(HOLDING_DAYS)

# top_features = ff.getTopFeatures(ff.clfs[0],30,False)
# X            = X[:,top_features]
# FeatureNames = np.array(FeatureNames)[top_features]
# Only Correlation Surprise Values
# X = X[:,:15]

def runCurrentPrediction(index, window = 252, clf_ind=5):
    training_X = X[range(index-window,index),:]
    training_y = y[range(index-window,index)]
    testing_X  = X[index,:]
    testing_y  = y[index]
    clf        = classifiers[clf_ind]

    ## DO TRANSFORMATION?
    clf.fit(training_X, training_y)
    if clf_ind == 5:
        top_features = sa.findPredictiveCharacteristics(clf,FeatureNames, False, 3)
        return clf.predict(testing_X)[0], testing_y, clf.predict_proba(testing_X), top_features
    return clf.predict(testing_X)[0], testing_y, clf.predict_proba(testing_X)

def runRollingPrediction(window=252, clf_ind=5):
    y_pred = []
    y_true = []
    probs  = []
    feats  = []
    indices = range(window, X.shape[0])
    for index in indices:
        if clf_ind == 5:
            y_p, y_t, proba, feat = runCurrentPrediction(index,window,clf_ind)
            feats.append(feat)
        else:
            y_p, y_t, proba = runCurrentPrediction(index,window,clf_ind)

        y_pred.append(y_p)
        y_true.append(y_t)

        if proba[0].shape[0] != 2:
            if y_p == 1:
                proba = np.array([0,1])
            else:
                proba = np.array([1,0])
            probs.append(proba)
        else:
            probs.append(proba[0])

    return np.array(y_pred), np.array(y_true), np.array(probs), np.array(feats)

y_pred, y_true, probs, clf_top_features = runRollingPrediction(LOOKBACK_WINDOW, CLF_IND)


print metrics.accuracy_score(y_true,y_pred)
print metrics.classification_report(y_true, y_pred)

if len(clf_top_features) > 0:
    print Counter(np.ravel(clf_top_features).tolist())



window_length = y_pred.shape[0]

returns        = returns[-window_length:]
dates          = d.dates[-window_length:]

TEST_WINDOW = 252
# ts.runBackTest(y_true, y_pred, [], returns, dates,True)
# ts.runBackTest(y_true[-TEST_WINDOW:], y_pred[-TEST_WINDOW:], [], returns[-TEST_WINDOW:], dates[-TEST_WINDOW:],True)


CURRENT_PRED = y_pred[0]
CURRENT_PROB = probs[0,:]
for i in range(y_pred.shape[0]):
    if i%HOLDING_DAYS == 0:
        CURRENT_PRED = y_pred[i]
        CURRENT_PROB = probs[i,:]
    else:
         y_pred[i]  = CURRENT_PRED
         probs[i,:] = CURRENT_PROB

ts.runBackTest(y_true, y_pred, probs, returns, dates, True, names[CLF_IND])


def violin_dist():
    zeros_ind = np.where(y_true == 0)[0]
    ones_ind  = np.where(y_true == 1)[0]
    vio_data  = [np.array(probs[zeros_ind,0]), np.array(probs[ones_ind,1])]

    return vio_data


a = violin_dist()
# plt.violinplot(a, [0,1], points=20, widths=0.1, showmeans=True, showextrema=True, showmedians=True)
# plt.show()
np.savetxt("violin.csv", a[0], delimiter=",")
np.savetxt("violin1.csv", a[1], delimiter=",")
