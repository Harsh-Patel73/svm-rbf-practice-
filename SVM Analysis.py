#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Predicts price of stocks using the SVM analysis and creating a RBF model


# In[2]:


from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import csv
plt.style.use('fivethirtyeight')


# In[3]:


# getting data
disData = open("SPY.csv", mode = "r")
df = pd.read_csv('SPY.csv')
df
stockName = "S&P 500"


# In[4]:


# Getting all data but last row

df = df.head(len(df))
df


# In[5]:


#Create empty lists to hold data

days = list()
adj_close_prices = list()


# In[6]:


# Pulling data and adj close prices

df_days = df.loc[:, 'Date']
df_adj_close = df.loc[:, 'Adj Close']


# In[7]:


#Create the independent data set of dates, create the dependent data of adj close prices
count = 0
for day in df_days:
    days.append([count])
    count += 1
    
for adj_close_price in df_adj_close:
    adj_close_prices.append(float(adj_close_price))


# In[8]:


# Creating RBF model
rbf_svr = SVR(kernel = 'rbf', C = 1000.0, gamma = 0.1)
rbf_svr.fit(days, adj_close_prices)


# In[9]:


#Plot models
plt.figure(figsize = (16,8))
plt.scatter(days,adj_close_prices, color = "black", label = 'Data')
plt.plot(days, rbf_svr.predict(days), color = "green", label = 'rbf model')
plt.xlabel('Days')
plt.ylabel('Adj Close Price')
plt.legend()
plt.show()


# In[10]:


# Show predicted price for a given day after the given data set

day = [[253]]
print('The RBF SVR predicted price: ', rbf_svr.predict(day))


# In[11]:


# Weekly Predictions

weeklyPredictionRbf = []
weeklyPredictionRbfFloat = []
weeklyPredictionPoly = []
weeklyPredictionPolyFloat = []
for i in range(252, 257):
    a = rbf_svr.predict([[i]])
    weeklyPredictionRbf.append(a)
for i in weeklyPredictionRbf:
    weeklyPredictionRbfFloat.append(float(i))


# In[12]:


#Make a chart with pandas to contain all of the data 
x = datetime.datetime.now()
print(x)

chartInfo = pd.DataFrame([[weeklyPredictionRbfFloat], [weeklyPredictionPolyFloat]], index = ["RBF", "Placeholder"])
chartInfo.columns = [f"{stockName} Updated at {x}"]
chartInfo.to_csv('CSA.csv')
chartInfo


# In[ ]:




