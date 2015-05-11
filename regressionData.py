import DataContainer as dc
import numpy as np
import pickle
import FeaturesHelper as ft

import DataContainer as dc
import numpy as np
import pickle
import FeaturesHelper as ft

def createData():
    with open('data/datacontainer.pkl','wb') as output:
        d = dc.DataContainer()
        pickle.dump(d, output, pickle.HIGHEST_PROTOCOL)

def getData():
    with open('data/datacontainer.pkl','rb') as input:
        d = pickle.load(input)
        return (d)


def convertToClass(ret, threshold = 0):
    if ret >= threshold:
        classification = 1
    else:
        classification = 0
    return (classification)


def createRegressionData(ret_days = 1):
    IND =  range(11, 35)
    data = getData()
    indicators_num = data.surprises_ind.shape[0]
    # Full list of features - X variable
    features = np.array(data.daily_ret[-indicators_num:-ret_days,IND])
    X = features
    #X = np.transpose(features)
    # X = np.concatenate((np.transpose(features),data.corr_mat_sp500[-indicators_num:-ret_days,1:11]),axis=1)
    # X = np.concatenate((X,data.sub_dollar_vol[-indicators_num:-ret_days,:]),axis=1)

    # Names of Features
    FeatureNames = [
                    "S5AUCO Index",	"S5CODU Index",	"S5HOTR Index",	"S5MEDA Index",	"S5RETL Index",	"S5FDSR Index",	"S5FDBT Index",	"S5HOUS Index",	"S5ENRSX Index",	"S5BANKX Index",	"S5DIVF Index",	"S5INSU Index",	"S5REAL Index",	"S5HCES Index",	"S5PHRM Index",	"S5CPGS Index",	"S5COMS Index",	"S5TRAN Index",	"S5SFTW Index",	"S5TECH Index",	"S5SSEQ Index",	"S5MATRX Index",	"S5TELSX Index",	"S5UTILX Index",
                    ]

    # corrs = ["corr_" + corr for corr in data.fid[2:12]]
    # volume_data = ["volume_"+vol for vol in data.fid[1:]]
    # FeatureNames = FeatureNames + corrs + volume_data

    # Targets - Y variable
    daily_ret    = data.daily_ret[-indicators_num+ret_days:,0]
  
    #actual_y     = data.daily_ret[-indicators_num+1:,0]
    actual_y     = ft.getCumRet(data.daily_ret, 0.2 * ret_days)[-indicators_num+ret_days:,0]
    y            = np.array([convertToClass(y_i) for y_i in actual_y])


    return X, y, np.array(FeatureNames), daily_ret, actual_y

X, y, featureNames, daily_ret, actual_y = createRegressionData()

