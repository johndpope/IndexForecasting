import os
import sys
import time
os.chdir('/Users/Eddie/Dropbox/IndexForecasting/dev/')

from scipy.spatial.distance import cdist, pdist
import sklearn, warnings, random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn import metrics
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation as AP
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import ProjectedGradientNMF as NMF
from sklearn.decomposition import FactorAnalysis as FA
import MLData as mld
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
from matplotlib.axes import Axes as Ax
from matplotlib import cm
from operator import add
from tabulate import tabulate
import scipy

random.seed(1234)
HOLDING_PERIOD = 20
d = mld.getData()
X, y, FeatureNames, returns, actual_y  = mld.createMLData(HOLDING_PERIOD)

clusters = ["KMeans","GMM", "SpectralClustering","AffinityProp","PCA","ICA", "NMF"] 
n = 8
clustering = np.array([
    KMeans(n_clusters = n,max_iter = 500),                                                           
    GMM(n_components = n, n_iter = 1000,n_init=15,random_state=1),                                            
    SpectralClustering(n_clusters = n),
    AP(damping=0.5, max_iter=200),                                         
    ])
num = 10
reduction = np.array([PCA(n_components= num),                                                          
    FastICA(n_components= num),
    NMF(n_components= num,max_iter=500),
    FA(n_components = num, max_iter=500) ])

# Define classifier
class_id        = 0
red_id          = 3
clf             = clustering[class_id]
red             = reduction[red_id]

# Scale the data for training
scaler = StandardScaler()
#scaler = MinMaxScaler()
X      = scaler.fit_transform(X)

# Splice into training and testing set
CUTOFF = int(252 * 1.5)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)
X_train = X[0:-CUTOFF,:]
X_test  = X[-CUTOFF:,:]
y_train = y[0:-CUTOFF]
y_test  = y[-CUTOFF:]
actual_y_train = actual_y[0:-CUTOFF]
actual_y_test = actual_y[-CUTOFF:]

## Plot of Variance Explained
# k_range = range(1,60)
# k_means_var =[KMeans(n_clusters=k).fit(X_train) for k in k_range]

# centroids = [X.cluster_centers_ for X in k_means_var]
# k_euclid =[cdist(X_train,cent,'euclidean') for cent in centroids] 
# dist = [np.min(ke,axis=1) for ke in k_euclid]
# cIdx = [np.argmin(ke,axis=1) for ke in k_euclid]
# wcss = [sum(d**2) for d in dist]
# tss = sum(pdist(X_train)**2)/X_train.shape[0]
# bss = tss -wcss
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(k_range, bss/tss*100, 'b*-')
# ax.set_ylim((0,100))
# plt.grid(True)
# plt.xlabel('Number of clusters')
# plt.ylabel('Percentage of variance explained (%)')
# plt.title('KMeans clustering')
# plt.show()

## Plot of Cluster Representation
# num_cluster = 5
# clr = cm.spectral( np.linspace(0,1,num_cluster+1) ).tolist()
# mrk = 'os^p<dvh8>+x.'
# fig = plt.figure()
# ax = fig.add_subplot(121)
# for i in range(num_cluster+1):
#     ind = (y_train==i)
#     ax.scatter(X_train[ind,0],X_train[ind,1], s=35, c=clr[i], marker=mrk[i], label='%d'%i)
# plt.legend()
# plt.title('Actual Digits')
# ax = fig.add_subplot(122)
# for i in range(num_cluster+1):
#     ind = (cIdx[num_cluster]==i)
#     ax.scatter(X_train[ind,0],X_train[ind,1], s=35, c=clr[i], marker=mrk[i], label='C%d'%i)
# plt.legend()
# plt.title('K=%d clusters'%k_range[num_cluster])
# plt.show()

# Fit Data
def performDimRed(X_train,X_test,red,temp_reduced_data,temp_reduced_test):
    temp_reduced_data = red.fit_transform(X_train)
    temp_reduced_test = red.transform(X_test)
    return [temp_reduced_data,temp_reduced_test]
    
#fitted = clf.fit(dim_reduced_data,actual_y_train)
#predicted = clf.predict(reduced_test)

def rumGMM(X_train,X_test,actual_y_train,returns):
    fitted = clf.fit(X_train,actual_y_train)
    predicted       = clf.predict(X_test)
    train_returns = returns[0:-CUTOFF]
    test_returns  = returns[-CUTOFF:]
    number = range(0,n)


    if class_id ==0:
        cluster_mean =[0] * (np.max(number)+1.0)
        cluster_std = [0] * (np.max(number)+1.0)
        for i in number:
            cluster_mean[i] = np.mean(returns[np.where(fitted.labels_ == i)])
            cluster_std[i] = np.std(returns[np.where(fitted.labels_ == i)])
    elif class_id == 1:
        cluster_std = np.sqrt(fitted.covars_[:,0])
        cluster_mean = fitted.means_[:,0]
        true_cluster = clf.predict(X_train)
        #true_cluster = clf.predict(dim_reduced_data)

    # fig = plt.figure()
    # ax = fig.add_subplot(121)
    # ax.plot(number,cluster_mean,'r--')
    # plt.title('Mean Return of Cluster')
    # ax = fig.add_subplot(122)
    # ax.plot(number,cluster_std,'r--')
    # plt.title('Std of Cluster Return')
    # plt.show()

    sorted_clusters = np.argsort(cluster_mean).astype(str)
    # cluster_mean[np.argsort(cluster_mean)]
    # width = [1*.7]*(np.max(number)+1)
    # fig,ax = plt.subplots()
    # rects = ax.bar(number,cluster_mean[np.argsort(cluster_mean)],.7,color='g')
    # #rects2 = ax.bar(map(add,number,width),cluster_std[np.argsort(cluster_mean)],.7, color='r')
    # ax.set_xticks(map(add,number,width))
    # ax.set_xticklabels(tuple([sorted_clusters[i] for i in number]))
    # ax.set_title('Sorted Means by Cluster')
    # ax.set_ylabel('Cluster Mean')
    # ax.set_xlabel('Cluster')
    # plt.show()

    predicted_train = clf.predict(X_train)
    returns_of_interest = [X_train[np.where(predicted_train == i),0] for i in  sorted_clusters.astype(float)]
    #returns_of_interest = [train_returns[np.where(predicted_train == i)] for i in  sorted_clusters.astype(float)]
    percentage_neg = [sum(sum(returns_of_interest[i]<0))/float(returns_of_interest[i].shape[1]) for i in number]
    #percentage_neg = [sum(returns_of_interest[i]<0)/float(len(returns_of_interest[i])) for i in number]
    num_in_cluster = [float(returns_of_interest[i].shape[1]) for i in number]
    print percentage_neg

    print sorted_clusters[0], percentage_neg[0], num_in_cluster[0]



    headers      = ["Cluster Number", "Perc Neg", "Perc Pos", "Cluster Size"]
    stats = np.transpose(np.vstack((sorted_clusters,percentage_neg,1-np.array(percentage_neg),num_in_cluster)))
    print tabulate(stats, headers, tablefmt="simple")

    unsorted_returns_of_interest = [X_train[np.where(predicted_train == i),0] for i in  number]
    cluster_median = [np.median(unsorted_returns_of_interest[i]) for i in number]
    sorted_median_index = np.argsort(cluster_median)
    sorted_median= [unsorted_returns_of_interest[i] for i in sorted_median_index]
    
    #sorted_median_index = np.argsort([np.median(returns_of_interest[i]) for i in number])
    #sorted_median = [returns_of_interest[i] for i in sorted_median_index]

    ## Box and Whisker Plot Clustered by Mean or Median
    # fig,ax = plt.subplots()
    # ##Mean
    # #plt.boxplot(returns_of_interest)
    # #Median
    # plt.boxplot(sorted_median)
    # #plt.boxplot(sorted_median[0:25])
    # #plt.boxplot(sorted_median[25:50])
    # ##Mean Tick Labels
    # #tick_labels = tuple([sorted_clusters[i] for i in number])
    # #Median Tick Labels
    # tick_labels = tuple([sorted_median_index[i] for i in number])
    # ax.set_xticklabels(tick_labels)
    # # print plt.axis()
    # plt.axis((.5,np.max(number)+.5,-.06,.06))
    # plt.axhline(0, xmin=0, xmax=1, color='g')
    # plt.title('Box and Whisker Plot of Clusters')
    # plt.ylabel('Returns')
    # plt.xlabel('Clusters')
    # plt.show()


    cluster_10 = [np.percentile(unsorted_returns_of_interest[i],10) for i in number]
    cluster_25 = [np.percentile(unsorted_returns_of_interest[i],25) for i in number]
    cluster_75 = [np.percentile(unsorted_returns_of_interest[i],75) for i in number]
    cluster_90 = [np.percentile(unsorted_returns_of_interest[i],90) for i in number]

    predict_matrix = np.transpose(np.vstack((number,cluster_mean,cluster_std,cluster_median,cluster_10,cluster_25,cluster_75,cluster_90)))

    print predict_matrix

    #print clf.predict_proba(X_test).shape
    test_predictions = clf.predict(X_test)
    actual_prediction_returns = actual_y_test

    cluster_decisions = [0] * len(number)
    test_decision = [0] * len(test_predictions)
    tree = 2.0

    if tree == 2.0:
        threshold = .000
        std_threshold = .0041
        if class_id == 1:
            cluster_decisions = cluster_mean > threshold
            test_decision = cluster_decisions[test_predictions]

        if class_id == 0:
            cluster_decisions = [cluster_mean[i] > threshold for i in number]
            test_decision = [cluster_decisions[test_predictions[i]] for i in range(0,len(test_predictions))]

        actual_prediction_returns[actual_prediction_returns > threshold] = 1
        actual_prediction_returns[actual_prediction_returns < -threshold] = 0
    # else:
    #     up_threshold = .0025
    #     down_threshold = .0025
    #     std_threshold = .0041

    #     for i in number:
    #         if ((cluster_mean[i] > up_threshold)):# & (cluster_median[i]>0.0)): 
    #             cluster_decisions[i] = 1 #& (cluster_mean > threshold) & (cluster_std < std_threshold)
    #         elif ((cluster_mean[i] < -down_threshold)): #& (cluster_median[i] < 0.0)):
    #             cluster_decisions[i] = -1
    #         else:
    #             cluster_decisions[i] = 0

    #     for i in range(0,len(test_predictions)):
    #         test_decision[i] =cluster_decisions[test_predictions[i]]

    #     truth_value = (actual_prediction_returns > -down_threshold) & (actual_prediction_returns < up_threshold)

    #     actual_prediction_returns[actual_prediction_returns > up_threshold] = 1
    #     actual_prediction_returns[actual_prediction_returns < -down_threshold] = -1
    #     actual_prediction_returns[truth_value] = 0

    print metrics.accuracy_score(actual_prediction_returns,test_decision)

    CURRENT_PRED = test_decision[0]
    if class_id == 1:
        for i in range(test_decision.shape[0]):
            if i%HOLDING_PERIOD == 0:
                CURRENT_PRED = test_decision[i]
            else:
                test_decision[i]  = CURRENT_PRED

    if class_id == 0:
        for i in range(len(test_decision)):
            if i%HOLDING_PERIOD == 0:
                CURRENT_PRED = test_decision[i]
            else:   
                test_decision[i]  = CURRENT_PRED



    #print sum((cluster_mean > up_threshold) )#& (cluster_std < std_threshold))

    runBackTest(np.array(actual_prediction_returns), np.array(test_decision),[],returns[-CUTOFF:], d.dates[-CUTOFF:],True)

#[reduced_train, reduced_test] = performDimRed(X_train,X_test,red,[],[])
[X_train, X_test] = performDimRed(X_train,X_test,red,[],[])
#rumGMM(reduced_train,reduced_test,actual_y_train,returns)
rumGMM(X_train,X_test,actual_y_train,returns)