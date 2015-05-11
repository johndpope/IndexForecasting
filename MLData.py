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

def createMLData(ret_days = 1):
    IND = 0
    data = getData()
    indicators_num = data.surprises_ind.shape[0]
    # Full list of features - X variable
    features = np.array([
                        data.daily_ret[-indicators_num:-ret_days,IND],
                        data.vol_10[-indicators_num:-ret_days,IND],
                        data.vol_20[-indicators_num:-ret_days,IND],
                        data.cum_ret_1[-indicators_num:-ret_days,IND],
                        data.cum_ret_4[-indicators_num:-ret_days,IND],
                        data.cum_ret_13[-indicators_num:-ret_days,IND],
                        data.cum_ret_52[-indicators_num:-ret_days,IND],
                        data.MA_20[-indicators_num:-ret_days,IND],
                        data.MA_50[-indicators_num:-ret_days,IND],
                        data.EMA_20[-indicators_num:-ret_days,IND],
                        data.EMA_50[-indicators_num:-ret_days,IND],
                        data.mom_ind[-indicators_num:-ret_days,IND],
                        data.surprises_ind[-indicators_num:-ret_days,0],
                        data.surprises_ind[-indicators_num:-ret_days,1],
                        data.surprises_ind[-indicators_num:-ret_days,2],
                        data.macd[-indicators_num:-ret_days],
                        data.rsi[-indicators_num:-ret_days],
                        ])


    X = np.concatenate((np.transpose(features),data.corr_mat_sp500[-indicators_num:-ret_days,1:11]),axis=1)
    X = np.concatenate((X,data.sub_dollar_vol[-indicators_num:-ret_days,:]),axis=1)

    # Names of Features
    FeatureNames = [
                    "daily_ret",
                    "vol_10",
                    "vol_20",
                    "cum_ret_1",
                    "cum_ret_4",
                    "cum_ret_13",
                    "cum_ret_52",
                    "MA_20",
                    "MA_50",
                    "EMA_20",
                    "EMA_50",
                    "mom_ind",
                    "turbulence",
                    "magnitude_suprise",
                    "correlation_surprise",
                    "MACD",
                    "RSI"
                    ]

    corrs = ["corr_" + corr for corr in data.fid[2:12]]
    volume_data = ["volume_"+vol for vol in data.fid[1:]]
    FeatureNames = FeatureNames + corrs + volume_data

    # Targets - Y variable
    daily_ret    = data.daily_ret[-indicators_num+ret_days:,IND]
    actual_y     = ft.getCumRet(data.daily_ret, 0.2 * ret_days)[-indicators_num+ret_days:,IND]
    y            = np.array([convertToClass(y_i) for y_i in actual_y])

    # INCLUDE MACD, RSI, T-BILLS, VIX ETC
    return X, y, np.array(FeatureNames), daily_ret, actual_y

# X, y, feats , daily_ret, actual_y = createMLData()
# X1, y1, feats1 , daily_ret1, actual_y1 = createMLData(10)
# print X1.shape
# print y1.shape
# print daily_ret1.shape
# print actual_y1.shape

