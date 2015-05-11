from tabulate import tabulate
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

# random.seed(1234)

# Define Classifiers
names = ["KNN","Logistic", "LIN_SVC","RBF_SVM","DTree",
         "RandForest", "AdaBoost", "NB_Gauss", "NB_Multinomial", "Perceptron"]

classifiers = np.array([
    KNeighborsClassifier(3),                                                           # 0
    linear_model.LogisticRegression(C=0.1),                                            # 1
    sklearn.svm.LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0),      # 2
    SVC(gamma=0.001, kernel='rbf', C=1, probability=True),                             # 3
    DecisionTreeClassifier(),                                                          # 4
    RandomForestClassifier(n_estimators=15,max_features=1,max_depth=5,random_state=9), # 5
    AdaBoostClassifier(base_estimator = SVC(gamma=0.001, C=1, probability=True), n_estimators=10),   # 6
    GaussianNB(),                                                                                    # 7
    MultinomialNB(alpha=0.01, fit_prior=False),                                                      # 8
    Perceptron(penalty='l2', fit_intercept=False, n_iter=20)                                         # 9
    ])

# Obtain Data
HOLDING_DAYS = 5

d = mld.getData()
X, y, FeatureNames, returns, actual_y = mld.createMLData(HOLDING_DAYS)

# Only Correlation Surprise Values
# X = X[:,:15]

# Select Top Features
top_features = ff.getTopFeatures(Ridge(alpha=0.0001, normalize=True, fit_intercept=False),30,False)
X            = X[:,top_features]
FeatureNames = np.array(FeatureNames)[top_features]

SELECT_FEATURES = False
# Define classifier
class_id        = 5
clf             = classifiers[class_id]

# Scale the data for training
scaler = StandardScaler()
# scaler = MinMaxScaler()
X      = scaler.fit_transform(X)

# Splice into training and testing set
# USE K-FOLD!!
CUTOFF = int(252 * 1.5)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)
X_train = X[0:-CUTOFF,:]
X_test  = X[-CUTOFF:,:]
y_train = y[0:-CUTOFF]
y_test  = y[-CUTOFF:]

if SELECT_FEATURES:
    # Feature Selection
    clf.fit(X_train, y_train)
    X_train_new = clf.transform(X_train)
    X_test_new  = clf.transform(X_test)
    X_train = X_train_new
    X_test = X_test_new

# combined = mcs.combineStrat(classifiers[[0,1,3,5,6,7,8]],X_train,y_train,X_test)
# ts.runBackTest(y_test, combined, [], returns[-CUTOFF:], d.dates[-CUTOFF:])

# Fit Data
clf.fit(X_train, y_train)
predicted       = clf.predict(X_test)
right_predict_p = clf.score(X_test,y_test)

print "--------------------------------------------------------------------------------"
print names[class_id]
# print "cross val score", np.mean(np.array(cross_validation.cross_val_score(clf, X, y, cv=10)))

try:
    proba = clf.predict_proba(X_test)
except Exception, e:
    proba = []

print "right prob score", right_predict_p

# DO HOLDING PERIOD
y_pred = predicted

# CURRENT_PRED = y_pred[0]
# CURRENT_PROB = proba[0,:]
# for i in range(y_pred.shape[0]):
#     if i%HOLDING_DAYS == 0:
#         CURRENT_PRED = y_pred[i]
#         CURRENT_PROB = proba[i,:]
#     else:
#          y_pred[i]  = CURRENT_PRED
#          proba[i,:] = CURRENT_PROB


ts.runBackTest(y_test, y_pred, proba, returns[-CUTOFF:], d.dates[-CUTOFF:], True)
print "--------------------------------------------------------------------------------"

sa.findPredictiveCharacteristics(clf, FeatureNames)

