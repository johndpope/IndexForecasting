import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import DataContainer as dc
import statsmodels.tsa.arima_model as ari

# dta = sm.datasets.sunspots.load_pandas().data[['SUNACTIVITY']]
# dta.index = pd.DatetimeIndex(start='1700', end='2009', freq='A')
# res = sm.tsa.ARMA(dta, (3, 0)).fit()
# fig, ax = plt.subplots()
# ax = dta.ix['1950':].plot(ax=ax)
# fig = res.plot_predict('1990', '2012', dynamic=True, ax=ax,
#                         plot_insample=False)
# plt.show()


data  = dc.DataContainer.daily_ret
dates = dc.DataContainer.dates
# # plt.plot(dates,data[:,0])
# # plt.show()



# df = pd.DataFrame(data[:,0],index=dates)


# # plt.plot(df)
# # plt.show


# # dta = sm.datasets.sunspots.load_pandas().data[['SUNACTIVITY']]
# # dta.index = pd.DatetimeIndex(start='1700', end='2009', freq='A')

# dta = df
# res = sm.tsa.ARMA(dta, (3, 0)).fit()
# fig, ax = plt.subplots()
# ax = dta.ix['1950':].plot(ax=ax)
# fig = res.plot_predict('2015', '2016', dynamic=True, ax=ax,
#                         plot_insample=False)
# plt.show()


# from statsmodels.tsa.arima_model import _arima_predict_out_of_sample
# res = sm.tsa.ARIMA(data[:,0], (3,1, 2)).fit()

model=ari.ARIMA(data[:,0],order=(3,0,2))
ar_res=model.fit()
preds=ar_res.predict(3000,3900, dynamic=True)
plt.plot(preds)
plt.show()

# get what you need for predicting one-step ahead
# params = res.params
# residuals = res.resid
# p = res.k_ar
# q = res.k_ma
# k_exog = res.k_exog
# k_trend = res.k_trend
# steps = 1

# statsmodels.tsa.arima_model.ARIMA.predict(params, start=len(data[:,0]))
