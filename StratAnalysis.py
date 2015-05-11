import numpy as np
import pandas
import sklearn
from tabulate import tabulate


def findPredictiveCharacteristics(clf, feature_names, print_stats = True, top_feats = 3):
    # For Random Forest
    importances  = np.array(clf.feature_importances_)
    sorted_index = np.argsort(-importances)


    statistics   = np.transpose(np.vstack((np.arange(1,len(sorted_index)+1),
                                feature_names[sorted_index],importances[sorted_index])))
    if print_stats:
        headers      = ["Rank.","Feature", "Importance"]
        print tabulate(statistics[:top_feats,:], headers, tablefmt="simple")

    return statistics[:top_feats,1]


def findPredictiveCharacteristicsReg(clf, feature_names, top_feats = 3):

    # Sort
    sorted_index = sorted(range(len(clf.coef_)), key=lambda k: abs(clf.coef_[k]))[::-1]
    return feature_names[sorted_index[:top_feats]]


