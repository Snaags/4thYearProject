import numpy
import pandas
import os
import sys
sys.path.append(r'C:\Users\chris\Anaconda3\Lib\site-packages')
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPRegressor

import StockData

path = os.getcwd()



def ConcatData(filename1, filename2):
    
	file1 = pandas.read_csv(filename1)
	file2 = pandas.read_csv(filename2)

	file1data = file1.loc[:,"Open":]
	file2data = file2.loc[:,"Open":]

	#file1date = file1.loc[:,"Date"]
	#file2date = file2.loc[:,"Date"]

	return pandas.concat([file1data,file2data],axis = 1)



dataset = ConcatData("AAPL.csv","GOOGL.csv")
y = list(dataset.iloc[:,1])
y.pop(0)
dataset = dataset.drop(dataset.index[[2766]])
NN  = MLPRegressor()
LinrReg = LinearRegression()
SVR = SVR(kernel = "linear") 
rfe = RFE(estimator=LinrReg,n_features_to_select = 1, step=1)

rfe.fit(dataset, y)
for i,c in zip(rfe.ranking_,list(dataset)):
    print(str(c)+": "+ str(i))

