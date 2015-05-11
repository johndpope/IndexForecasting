import numpy as np
import sklearn
import scipy

#AdaBoostClassifier, BernoulliNB, DecisionTreeClassifier, ExtraTreeClassifier, ExtraTreesClassifier,
#MultinomialNB, NuSVC, Perceptron, RandomForestClassifier, RidgeClassifierCV, SGDClassifier, SVC

def combineStrat(clfs, X_train, y_train, X_test):
    num_classifiers = len(clfs)
    predictions = np.zeros([X_test.shape[0],num_classifiers])
    proba       = np.zeros([X_test.shape[0],len(np.unique(y_train))])
    i = 0

    for clf in clfs:
        clf.fit(X_train, y_train)
        predictions[:,i]       = clf.predict(X_test)
        # right_predict_p = clf.score(X_test,y_test)
        print i
        proba = proba + clf.predict_proba(X_test)


        i = i+1

    # predictions = (np.sum(combined,axis=1)>=3).astype(int)
    predictions = np.transpose(np.array(scipy.stats.mstats.mode(predictions,axis=1)[0]))[0]

    print proba/float(num_classifiers)
    return predictions

