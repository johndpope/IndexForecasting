import numpy as np
import scipy
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn import metrics

TRADING_DAYS = 252
rf = 0.01

# EXPECT y_pred to be 1 or 0 ONLY
def runBackTest(y_true, y_pred, proba, returns, dates, plotGraph=True, ML_Name = ''):

    prob_threshold = 0.525
    headers = ["Strategy", "Annual Return", "Outperformance", "Volatility", "Sharpe Ratio", "Max DrawDown"]
    strategies = ["Market", "Long Only", "Long Short", "Kelly Long Only", "Kelly Long Short", "Kelly Long Only High-Prob", "Kelly Long Short High-Prob"]
    # Set up strategies
    market_strat     = np.zeros(len(y_pred)) + 1
    long_only_strat  = y_pred
    long_short_strat = y_pred * 2 -1

    if len(proba) == 0:
        # Load strats vectors
        all_strats = [market_strat, long_only_strat, long_short_strat]

    else:
        # Kelly Strats - Probability Based
        high_prob_index = np.max(proba,axis=1) > prob_threshold
        high_prob_index = [int(p) for p in high_prob_index]
        prob_diff = abs(proba[:,0]-proba[:,1])
        weighting = 1+prob_diff
        kelly_long_only_strat    = np.multiply(weighting, long_only_strat)
        kelly_long_only_strat_h  = np.multiply(kelly_long_only_strat, high_prob_index)
        kelly_long_short_strat   = np.multiply(weighting, long_short_strat)
        kelly_long_short_strat_h = np.multiply(kelly_long_short_strat, high_prob_index)
        # Load strats vectors
        all_strats = [market_strat, long_only_strat, long_short_strat, kelly_long_only_strat,
                      kelly_long_short_strat, kelly_long_only_strat_h, kelly_long_short_strat_h]

    strats_range = range(len(all_strats))

    # Obtain performance metrics
    stats = []
    Performances = []

    for i in strats_range:
        Strat_Prices, Strat_Annual_ret, Strat_OutPerformance, Strat_Volatility, Strat_SR, Strat_MaxDD = getPerformanceStats(returns, all_strats[i])
        stats.append([strategies[i], Strat_Annual_ret, Strat_OutPerformance,Strat_Volatility,Strat_SR,Strat_MaxDD])
        Performances.append(Strat_Prices)


    # print
    # print tabulate(stats, headers, tablefmt="simple", floatfmt=".4f")

    if plotGraph:
        lines = []
        for i in strats_range:
            plt.plot(dates, Performances[i], label = strategies[i])

        plot_title = 'Returns of Strats vs Market'
        if ML_Name != '':
            plot_title = plot_title + ' for ' + ML_Name
        plt.xlabel('Date')
        plt.ylabel('Returns')
        plt.title(plot_title)
        plt.legend(loc=0,prop={'size':8})
        plt.grid()
        plt.show()

    #print metrics.classification_report(y_true, y_pred)
    Matrix=stats
    return Matrix

# EXPECT y_pred to be 1, 0, -1...
def runBackTestTri(y_true, y_pred, returns, dates, plotGraph=True):

    prob_threshold = 0.525
    headers = ["Strategy", "Annual Return", "Outperformance", "Volatility", "Sharpe Ratio", "Max DrawDown"]
    strategies = ["Market", "Long Only", "Long Short"]
    # Set up strategies
    market_strat     = np.zeros(len(y_pred)) + 1

    long_only_strat  = y_pred[y_pred==-1]=0
    long_short_strat = y_pred

    all_strats = [market_strat, long_only_strat, long_short_strat]

    strats_range = range(len(all_strats))

    # Obtain performance metrics
    stats = []
    Performances = []

    for i in strats_range:
        Strat_Prices, Strat_Annual_ret, Strat_OutPerformance, Strat_Volatility, Strat_SR, Strat_MaxDD = getPerformanceStats(returns, all_strats[i])
        stats.append([strategies[i], Strat_Annual_ret, Strat_OutPerformance,Strat_Volatility,Strat_SR,Strat_MaxDD])
        Performances.append(Strat_Prices)

    print
    print tabulate(stats, headers, tablefmt="simple")

    if plotGraph:
        lines = []
        for i in strats_range:
            plt.plot(dates, Performances[i], label = strategies[i])

        plt.xlabel('Date')
        plt.ylabel('Returns')
        plt.title('Returns of Strat vs Market')
        plt.legend(loc=0)
        plt.grid()
        plt.show()

    print metrics.classification_report(y_true, y_pred)





def getPerformanceStats(returns, strat):
    # Return all important performance metrics
    Strat_Prices         = np.cumprod(np.multiply(returns, strat)+1)
    Strat_Annual_ret     = np.power(scipy.stats.mstats.gmean(np.multiply(returns, strat)+1),TRADING_DAYS) - 1
    Strat_Volatility     = np.std(np.multiply(returns, strat)) * np.sqrt(TRADING_DAYS)
    Strat_SR             = (Strat_Annual_ret - rf)/ Strat_Volatility
    Strat_MaxDD          = getMaxDrawDown(Strat_Prices)

    Market_Annual_Ret    = np.power(scipy.stats.mstats.gmean(returns+1),TRADING_DAYS) - 1
    Strat_OutPerformance = Strat_Annual_ret - Market_Annual_Ret

    return Strat_Prices, Strat_Annual_ret, Strat_OutPerformance, Strat_Volatility, Strat_SR, Strat_MaxDD


def getDrawDown(Prices):
    return (np.max(Prices) - Prices[-1]) / np.max(Prices)

def getMaxDrawDown(Prices):
    # Max DrawDowns
    drawDowns = []
    for i in range(len(Prices))[1:]:
        drawDowns.append(getDrawDown(Prices[range(i)]))
    return np.max(drawDowns)

