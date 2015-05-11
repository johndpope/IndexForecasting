import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

def checkTesting(clf, clfType, testingX, testingY, trainingSampleSize, plotDetails=False):

    predicted = clf.predict(testingX)
    # Get % of right spam predictions
    diffs  =  np.array([p_i - a_i for p_i, a_i in zip(predicted, testingY)])

    right_predict_p = clf.score(testingX,testingY)

    print "Number of training samples:",trainingSampleSize
    print "Number of testing samples :",len(testingY)
    print "Percent of right predictions:",right_predict_p
    print metrics.classification_report(testingY, predicted)

    print "Number of training samples:",trainingSampleSize
    print "Number of testing samples :",len(testingY)
    print "Percent of right predictions:",right_predict_p
    print metrics.classification_report(testingY, predicted)

    noise_p = np.random.normal(0,0.05,len(predicted))
    noise_a = np.random.normal(0,0.05,len(predicted))

    if plotDetails:
        try:
            probas_ = clf.predict_proba(testingX)
            plt.subplot(1,2,1)
            plt.scatter(np.array([p_i - a_i for p_i, a_i in zip(predicted, noise_p)]),
                        np.array([p_i - a_i for p_i, a_i in zip(testingY, noise_a)]),
                        c=np.absolute(diffs))
            plt.xlabel('Predictions')
            plt.ylabel('Actuals')
            plt.title(clfType+' Predictions vs Actuals (0 = Spam, 1 = Not Spam)')

            plt.subplot(1,2,2)


            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(testingY, probas_[:, 1])
            # log scale
            # fpr = np.log(fpr) / np.log(10)
            roc_auc = auc(fpr, tpr)

            print "Area under the ROC curve : %f" % roc_auc

            # Plot ROC curve
            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')

            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic')
            plt.legend(loc="lower right")
            plt.show()

        except Exception, e:
            plt.scatter(np.array([p_i - a_i for p_i, a_i in zip(predicted, noise_p)]),
                        np.array([p_i - a_i for p_i, a_i in zip(testingY, noise_a)]),
                        c=np.absolute(diffs))
            plt.xlabel('Predictions')
            plt.ylabel('Actuals')
            plt.title(clfType+' Predictions vs Actuals (0 = Spam, 1 = Not Spam)')
            plt.show()

    return(right_predict_p)
