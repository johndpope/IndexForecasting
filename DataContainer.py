import FeaturesHelper as ft
import numpy as np
import csv, shelve, datetime

class DataContainer:

    # READ ALL DATA
    SUB_FILE         = 'data/sub_returns.csv'
    SUB_VOLUME       = 'data/sub_volumes.csv'
    sub_data         = np.genfromtxt(SUB_FILE, delimiter=',')[1:,1:]
    sub_vol_data     = (np.genfromtxt(SUB_VOLUME, delimiter=',')[1:,1:] + 1) # ADD 1 FOR SMOOTHING
    sub_dollar_vol   = np.multiply(sub_vol_data, sub_data)                   # Dollar volume
    sub_gold         = 'data/Gold.csv'
    sub_gold_data    = np.genfromtxt(SUB_VOLUME, delimiter=',')[1:,1:]  


    # GET INDEX/COLUMN NAMES
    reader = csv.reader(open(SUB_FILE, 'rU'), dialect=csv.excel_tab)
    fid    = reader.next()[0].split(',')

    # GET THE DATES
    with open(SUB_FILE, 'rU') as f:
        reader = csv.reader(f, delimiter=',')
        dt     = [row[0] for row in reader]
    dt    = dt[1:]
    dates = np.array([datetime.datetime.strptime(d, '%m/%d/%y') for d in dt])


    # MATRIX INDICES
    sectors      = range(1, 11)
    sub_indicies = range(11, sub_data.shape[1])

    total_days   = sub_data.shape[0]
    trading_days = 252;

    ## GET FEATURES OF DATA

    # Get daily returns + volatility stats
    daily_ret        = np.divide(sub_data[1:,:], sub_data[0:-1,:])-1;

    overall_ret      = ft.getOverallRet(daily_ret)
    vol_10           = ft.getVol(daily_ret, 10)
    vol_20           = ft.getVol(daily_ret, 20)

    # Get the 1/4/13/52 cumulative week returns:
    cum_ret_1        = ft.getCumRet(daily_ret, 1)
    cum_ret_4        = ft.getCumRet(daily_ret, 4)
    cum_ret_13       = ft.getCumRet(daily_ret, 13)
    cum_ret_52       = ft.getCumRet(daily_ret, 52)

    # Get 20 day and 50 day Simple Moving Averages
    MA_20            = ft.getMA(sub_data, 20)
    MA_50            = ft.getMA(sub_data, 50)
    EMA_20           = ft.getEMA(sub_data, 20)
    EMA_50           = ft.getEMA(sub_data, 50)

    # GET CORRELATION
    corr_mat_sp500   = ft.getCorrMatrix(daily_ret, 20, 0)

    # Get the Correlation Surprise index
    surprises_ind    = ft.getCorrSurprise(daily_ret, 3)

    # Get Momentum Indicator 10 day
    mom_ind          = ft.getMomInd(sub_data, 10)

    # Get the MACD indicator
    macd             = np.array(ft.getMACD(sub_data)[2])[:,0]

    # Get RSI indicator
    rsi              = np.array(ft.getRSI(sub_data[:,0]))



    # Get GOLD returns
    gold_daily_ret   = np.divide(sub_gold_data[1:,:], sub_gold_data[0:-1,:])-1;
    goldRet          = ft.getCumRet(gold_daily_ret, 5)

    ##




