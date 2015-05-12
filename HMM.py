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
from sklearn.linear_model import Perceptron, Ridge
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# SETTINGS ####################
# HOLDING_DAYS    = 20
# LOOKBACK_WINDOW = 252
n_components = 10
# ###############################

# Obtain Data
d      = mld.getData()
X, y, FeatureNames, returns, actual_y = mld.createMLData()
dates  = d.dates[-len(y):]
values = d.sub_data[-len(y):,0]

# Scale the data for training and PCA

# scaler = StandardScaler()
# X      = scaler.fit_transform(X)

# pca    = PCA(n_components= 30)
# X      = pca.fit_transform(X)


# Select Top Features
# top_features = ff.getTopFeatures(Ridge(alpha=0.0001, normalize=True, fit_intercept=False),30,False)
# X            = X[:,top_features]
# FeatureNames = np.array(FeatureNames)[top_features]

print "fitting to HMM and decoding ...",

# make an HMM instance and execute fit
model = GaussianHMM(n_components, "diag", n_iter=1000)
model.fit([X])

hidden_states = model.predict(X)

print "done\n"

###############################################################################
# print trained parameters and plot
print "Transition matrix"
print model.transmat_
print ""

print "means and vars of each hidden state"
for i in xrange(n_components):
    print "%dth hidden state" % i
    print "mean = ", model.means_[i]
    print "var = ", np.diag(model.covars_[i])
    print ""

fig = plt.figure()
ax  = fig.add_subplot(111)

for i in xrange(n_components):
    # use fancy indexing to plot data in each state
    idx = (hidden_states == i)
    ax.plot_date(dates[idx], values[idx], 'o', label="hidden state: %d" % (i+1))

plt.legend(loc=0,prop={'size':8})
plt.xlabel('Date')
plt.ylabel('Level')
plt.title('HMM fit of S&P 500 Index')
ax.autoscale_view()

# format the coords message box
ax.grid(True)

fig.autofmt_xdate()

plt.show()


