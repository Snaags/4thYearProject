import numpy
import pandas
import os
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt


path = os.getcwd()
os.chdir(path +"\StockData")


def ConcatData(filename1, filename2):

	file1 = pandas.read_csv(filename1)
	file2 = pandas.read_csv(filename2)

	file1data = file1.loc[:,"Open":]
	file2data = file2.loc[:,"Open":]

	#file1date = file1.loc[:,"Date"]
	#file2date = file2.loc[:,"Date"]

	return pandas.concat([file1data,file2data],axis = 1)



dataset = ConcatData("GOOGL.csv","AAPL.csv")
svc = SVR(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)

y = list(dataset.iloc[:,1])


x= rfe.fit(dataset, y)

