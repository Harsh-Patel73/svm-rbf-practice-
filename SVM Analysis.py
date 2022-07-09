from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import csv
plt.style.use('fivethirtyeight')

disData = open("SPY.csv", mode = "r")
df = pd.read_csv('SPY.csv')
stockName = "S&P 500"
df = df.head(len(df))

days = list()
adj_close_prices = list()

df_days = df.loc[:, 'Date']
df_adj_close = df.loc[:, 'Adj Close']

count = 0
for day in df_days:
    days.append([count])
    count += 1
    
for adj_close_price in df_adj_close:
    adj_close_prices.append(float(adj_close_price))

rbf_svr = SVR(kernel = 'rbf', C = 1000.0, gamma = 0.1)
rbf_svr.fit(days, adj_close_prices)

plt.figure(figsize = (16,8))
plt.scatter(days,adj_close_prices, color = "black", label = 'Data')
plt.plot(days, rbf_svr.predict(days), color = "green", label = 'rbf model')
plt.xlabel('Days')
plt.ylabel('Adj Close Price')
plt.legend()
plt.show()

weeklyPredictionRbf = []
weeklyPredictionRbfFloat = []
weeklyPredictionPoly = []
weeklyPredictionPolyFloat = []
for i in range(252, 257):
    a = rbf_svr.predict([[i]])
    weeklyPredictionRbf.append(a)
for i in weeklyPredictionRbf:
    weeklyPredictionRbfFloat.append(float(i))

x = datetime.datetime.now()
print(x)

chartInfo = pd.DataFrame([[weeklyPredictionRbfFloat], [weeklyPredictionPolyFloat]], index = ["RBF", "Placeholder"])
chartInfo.columns = [f"{stockName} Updated at {x}"]
chartInfo.to_csv('CSA.csv')
chartInfo

