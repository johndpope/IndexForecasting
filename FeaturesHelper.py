import numpy as np
from scipy import stats

trading_days = 252

# Get overall returns
def getOverallRet(daily_ret):
    daily_ret = np.array(daily_ret)
    return (pow(stats.gmean(daily_ret+1), trading_days) - 1);


# Get the volatilities over entire period with given lag
def getVol(daily_ret, lag):
    total_days   = daily_ret.shape[0]
    vol = np.zeros(shape=(total_days-lag+1,daily_ret.shape[1]))

    for i in range(lag-1, total_days):
        start = i - lag + 1
        current_period_vol = np.std(daily_ret[start:(i+1),:],axis=0) * np.sqrt(trading_days)
        vol[start,:] = current_period_vol

    return (vol)


# Get cumulative returns
def getCumRet(daily_ret, week):
    lag = int(5.0 * week)
    total_days   = daily_ret.shape[0]
    cum_ret = np.zeros(shape=(total_days-lag+1,daily_ret.shape[1]))

    for i in range(lag-1, total_days):
        start = i - lag + 1
        current_cumret = np.prod(daily_ret[start:(i+1),:]+1,axis=0) - 1

        cum_ret[start,:] = current_cumret

    return (cum_ret)


# Get Moving Averages
def getMA(prices, lag):
    total_days = prices.shape[0]
    MA = np.zeros(shape=(total_days-lag+1,prices.shape[1]))

    for i in range(lag-1, total_days):
        start = i - lag + 1
        current_period_MA = np.mean(prices[start:(i+1),:],axis=0)
        MA[start,:] = current_period_MA

    return (MA)


# Get Exponential Moving Averages
def getEMA(prices, lag):
    total_days = prices.shape[0]
    alpha = 2.0/(lag+1);
    EMA = np.zeros(shape=(total_days-lag+1,prices.shape[1]))

    for i in range(lag-1, total_days):
        start = i - lag + 1

        if (i == (lag-1)):
            current_period_EMA = np.mean(prices[start:(i+1),:],axis=0)
        else:
            current_period_EMA = EMA[start-1,:] + alpha * (prices[i,:]-EMA[start-1,:])

        EMA[start,:] = current_period_EMA

    return (EMA)


# Get the correlation over entire period with given lag
def getCorrMatrix(daily_ret, lag, target_index):
    np.seterr(divide='ignore', invalid='ignore')
    total_days = daily_ret.shape[0]
    corrMAt = np.zeros(shape=(total_days-lag+1,daily_ret.shape[1]))

    for i in range(lag-1, total_days):
        start = i - lag + 1

        for item in range(daily_ret.shape[1]):
            corrMAt[start,item] =  np.corrcoef(daily_ret[start:(i+1),target_index],daily_ret[start:(i+1),item])[0,1]

    return (corrMAt)

# Turbulance function
def getTurbulance( mean_rets, current_rets, cov):
    # Obtain covariance matrix from vols and correlation
    n        = mean_rets.shape[0]
    cov_diag = np.multiply(np.eye(n), cov)

    # Turbulance Factor from Kinlaw & Turkington (2012)
    dt = np.transpose(np.dot(np.dot((current_rets-mean_rets),np.linalg.inv(cov+np.eye(n)*0.000001)),(current_rets-mean_rets)))/n

    # Magnitude Surprise
    mag_sup = np.dot(np.dot((current_rets-mean_rets),np.linalg.inv(cov_diag)),(current_rets-mean_rets))/n

    # Correlation Surprise
    corr_sup = dt/mag_sup

    return (dt, mag_sup, corr_sup)

# Get correlation surprise index
def getCorrSurprise(daily_ret, look_back):
    lag        = look_back * trading_days
    total_days = daily_ret.shape[0]
    surprises  = np.zeros(shape=(total_days-lag+1,3))

    for i in range(lag-1, total_days):
        start = i - lag + 1
        mean_rets = np.mean(daily_ret[start:(i+1),:],axis=0)
        cov_rets  = np.cov(np.transpose(daily_ret[start:(i+1),:]))
        dt, mag_sup, corr_sup = getTurbulance( mean_rets, daily_ret[i,:], cov_rets);
        surprises[start,:] =  [dt, mag_sup, corr_sup]

    return (surprises)


# Get the Momentum Indicator over entire period with given lag
def getMomInd(prices, lag):
    total_days = prices.shape[0]
    alpha = 2.0/(lag+1);
    mom_values = np.zeros(shape=(total_days-lag+1,prices.shape[1]))

    for i in range(lag-1, total_days):
        start = i - lag + 1
        current_period_mom = prices[i,:] - prices[start,:]
        mom_values[start,:] = current_period_mom

    return (mom_values)


# Get the MACD indicator
def getMACD(prices, slow=26, fast=12):
    emaslow = getEMA(prices, slow)
    emafast = getEMA(prices, fast)
    length  = min(emaslow.shape[0],emafast.shape[0])

    emaslow = emaslow[-length:,:]
    emafast = emafast[-length:,:]
    return emaslow, emafast, emafast - emaslow

# Get RSI indicator
def getRSI(prices, n=10):
    deltas  = np.diff(prices)
    seed    = deltas[:n+1]
    up      = seed[seed>=0].sum()/n
    down    = -seed[seed<0].sum()/n
    rs      = up/down
    rsi     = np.zeros_like(prices)
    rsi[:n] = 100.0 - 100.0/(1.0+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval   = delta
            downval = 0.0
        else:
            upval   = 0.0
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100.0 - 100.0/(1.0+rs)

    return rsi

# Get T-bill rate
# Get LIBOR rate
# Get Gold price
