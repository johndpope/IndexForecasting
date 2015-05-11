import numpy as np
import MLData as mld
from sklearn import linear_model
from sklearn.linear_model import Ridge, Lasso, ElasticNet, lasso_path, enet_path
from scipy import stats
from tabulate import tabulate
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Elastic Net - for highly correlated vars
# Correlation of returns vs market
def printOLSStats():
    # CHECK 33 FEATURES
    use_added_features = False
    X, _, FeatureNames, returns, y = mld.createMLData()

    X = sm.add_constant(X)
    est = sm.OLS(y, X)
    est = est.fit()
    print est.summary()

def getTopFeaturesOLS(n=10,showStats=False):

    d = mld.getData()
    X, _, FeatureNames, returns, y = mld.createMLData()

    scaler = StandardScaler()
    X      = scaler.fit_transform(X)

    headers      = ["No.","Feature", "Slope", "R2", "P Value", "Standard Error"]
    statistics   = []
    slopes       = []

    for i in range(X.shape[1]):
        slope, intercept, r_value, p_value, std_err = stats.linregress(X[:,i],y)
        statistics.append([i+1, FeatureNames[i], slope, r_value, p_value, std_err])
        slopes.append(slope)

    # Sort
    sorted_index = sorted(range(len(slopes)), key=lambda k: abs(slopes[k]))[::-1]
    if showStats:
        statistics = [statistics[i] for i in sorted_index]

        print
        print tabulate(statistics[:n], headers, tablefmt="simple", floatfmt=".5f")

    return sorted_index[:n]

def getTopFeatures(clf, n=10, showStats=False):

    d = mld.getData()
    X, _, FeatureNames, returns, y = mld.createMLData()

    scaler = StandardScaler()
    X      = scaler.fit_transform(X)

    headers      = ["No.","Feature", "Slope"]

    clf.fit(X,y)

    # Sort
    sorted_index = sorted(range(len(clf.coef_)), key=lambda k: abs(clf.coef_[k]))[::-1]

    headers      = ["No.","Feature", "Slope"]
    statistics   = []

    if showStats:

        for i in sorted_index:
            statistics.append([i+1, FeatureNames[i],clf.coef_[i]])

        statistics = np.array(statistics)
        print
        print tabulate(statistics[:n,:], headers, tablefmt="simple", floatfmt=".5f")
        print clf.score(X,y)

    return sorted_index[:n]
    # return clf.score(X,y)

def pathPlot():
    X, _, FeatureNames, returns, y = mld.createMLData()

    X /= X.std(axis=0)  # Standardize data (easier to set the l1_ratio parameter)

    # Compute paths
    eps = 5e-3  # the smaller it is the longer is the path

    print("Computing regularization path using the lasso...")
    alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept=False)
    print coefs_lasso.shape
    print("Computing regularization path using the positive lasso...")
    alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(
        X, y, eps, positive=True, fit_intercept=False)
    print("Computing regularization path using the elastic net...")
    alphas_enet, coefs_enet, _ = enet_path(
        X, y, eps=eps, l1_ratio=0.8, fit_intercept=False)

    print("Computing regularization path using the positve elastic net...")
    alphas_positive_enet, coefs_positive_enet, _ = enet_path(
        X, y, eps=eps, l1_ratio=0.8, positive=True, fit_intercept=False)

    # Display results

    plt.figure(1)
    ax = plt.gca()
    ax.set_color_cycle(2 * ['b', 'r', 'g', 'c', 'k'])
    l1 = plt.plot(-np.log10(alphas_lasso), coefs_lasso[:20].T)
    l2 = plt.plot(-np.log10(alphas_enet), coefs_enet[:20].T, linestyle='--')

    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    plt.title('Lasso and Elastic-Net Paths')
    plt.legend((l1[-1], l2[-1]), ('Lasso', 'Elastic-Net'), loc='lower left')
    plt.axis('tight')

    # plt.figure(2)
    # ax = plt.gca()
    # ax.set_color_cycle(2 * ['b', 'r', 'g', 'c', 'k'])
    # l1 = plt.plot(-np.log10(alphas_lasso), coefs_lasso.T)
    # l2 = plt.plot(-np.log10(alphas_positive_lasso), coefs_positive_lasso.T,
    #               linestyle='--')

    # plt.xlabel('-Log(alpha)')
    # plt.ylabel('coefficients')
    # plt.title('Lasso and positive Lasso')
    # plt.legend((l1[-1], l2[-1]), ('Lasso', 'positive Lasso'), loc='lower left')
    # plt.axis('tight')


    # plt.figure(3)
    # ax = plt.gca()
    # ax.set_color_cycle(2 * ['b', 'r', 'g', 'c', 'k'])
    # l1 = plt.plot(-np.log10(alphas_enet), coefs_enet.T)
    # l2 = plt.plot(-np.log10(alphas_positive_enet), coefs_positive_enet.T,
    #               linestyle='--')

    # plt.xlabel('-Log(alpha)')
    # plt.ylabel('coefficients')
    # plt.title('Elastic-Net and positive Elastic-Net')
    # plt.legend((l1[-1], l2[-1]), ('Elastic-Net', 'positive Elastic-Net'),
    #            loc='lower left')
    # plt.axis('tight')
    plt.show()

clf_names = [
            "Ridge Regression",
            "Lasso Regression",
            "Elastic Net",
            ]

clfs = [
        Ridge     (alpha=0.01, normalize=True, fit_intercept=False),
        Lasso     (alpha=0.00002, normalize=True, fit_intercept=False),
        ElasticNet(alpha=0.0005, normalize=True, fit_intercept=False),
        ]


# clf_ind = 1
# getTopFeatures(clfs[clf_ind], 20, True)
# printOLSStats()
# pathPlot()

