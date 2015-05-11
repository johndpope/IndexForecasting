import Metrics as m
import MLData as mld
import numpy as np
import sklearn
import warnings
from sklearn import cross_validation
from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from tabulate import tabulate
from sklearn import metrics

names = ["KNN","Logistic", "LIN_SVC","RBF_SVM", "DTree",
         "RandForest", "AdaBoost", "NB_Gauss", "LDA", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    linear_model.LogisticRegression(C=1e5),
    sklearn.svm.LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0),
    SVC(gamma=0.001, kernel='rbf', C=1, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=15,max_features=1,max_depth=5),
    AdaBoostClassifier(base_estimator = SVC(gamma=0.001, kernel='rbf', C=10.0, probability=True), n_estimators=10),
    GaussianNB(),
    LDA(),
    QDA()]

# Get Machine Learning Data
X, y, FeatureNames, returns, actual_y = mld.createMLData()

# Scale the data for training
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Splice into training and testing set
CUTOFF = int(y.shape[0]*0.9)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)
X_train = X[0:CUTOFF,:]
X_test  = X[CUTOFF:,:]
y_train = y[0:CUTOFF]
y_test  = y[CUTOFF:]

# Compare all
headers = ["Classifier", "Score", "Precision 0", "Precision 1", "Recall 0", "Recall 1"]
stats = []
i = 0
for clf in classifiers:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        clf.fit(X_train, y_train)
        right_predict_p = clf.score(X_test,y_test)

        y_pred = clf.predict(X_test)
        precision, recall, _, _ = metrics.precision_recall_fscore_support(y_test,y_pred)
        stats.append([names[i], right_predict_p, precision[0], precision[1], recall[0], recall[1]])

        # 10-Fold
        # stats.append([names[i], np.mean(np.array(cross_validation.cross_val_score(clf, X, y, cv=10)))])
        i = i+1

print tabulate(stats, headers, tablefmt="simple")





