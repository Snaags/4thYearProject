#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 00:00:46 2020

@author: alexissofias
"""


import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy.stats as sp
import os
path = os.getcwd()

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVR


MSFTC = pandas.read_csv(path+"/StockData/AAPL.csv").loc[:,"Close"]
#MSFTC = np.asarray(MSFTC)#convert to numpy array
AAPLC = pandas.read_csv(path+"/StockData/MSFT.csv").loc[:,"Close"]
#AAPLC = np.asarray(AAPLC)
dataframe= pandas.concat([MSFTC,AAPLC],axis=1)
dataset = dataframe.values
dataset = dataset.astype('float32')


train_size = int(len(dataset) * 0.85)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]



train = pandas.DataFrame(train,columns=['Open_MICROSOFT','Open_NASDAQ'])
test = pandas.DataFrame(test,columns=['Open_MICROSOFT','Open_NASDAQ'])




x=train[['Open_MICROSOFT','Open_NASDAQ']][:-1]
y=train['Open_MICROSOFT'][1:]

RidgeReg = Ridge()
LinrReg = LinearRegression()

SVR = LinearSVR() 


regressor = Ridge()
regressor.fit(x, y) #training the algorithm

print('Intercept: \n', regressor.intercept_)
print('Coefficients: \n', regressor.coef_)


x=test[['Open_MICROSOFT','Open_NASDAQ']][:-1]
y=test['Open_MICROSOFT'][1:]

prediction = []
print(x)

prediction = (regressor.predict(x))


for c,i in zip(prediction, y):
	print(c,"    ",i)



from sklearn.metrics import mean_squared_error

mean=mean_squared_error(prediction,y)
print("Error from prediction: ",mean)
mean=mean_squared_error(x['Open_MICROSOFT'],y)
print("Error from input: ",mean)


total_error = 0
for i,c in zip(prediction,y):
	test_lost_score = abs(c - i)
	test_lost_score = test_lost_score/c
	total_error += test_lost_score

total_error = total_error/ len(y)
total_error = total_error*100

print("MAPE: ", total_error)

plt.plot(y,label = "MSFT")
plt.plot(prediction, label = "Prediction")
#plt.plot(x, label = "input")
plt.legend()
plt.show()




















