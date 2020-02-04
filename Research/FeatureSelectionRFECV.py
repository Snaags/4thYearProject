import numpy as np
import pandas

import os
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold


def ConcatData(files,shape):
	hold = pandas.DataFrame()

	for i in files:

		data = pandas.read_csv(i)

		if data.shape == shape and data.isnull().any().any() == False:
			data = data.loc[:,"Open":]
			hold = pandas.concat([hold,data],axis = 1)

		##if all(pandas.notna(np.min(data))):

		
	return hold
        
 

os.chdir(os.getcwd()+"/StockData")

files = os.listdir()
files.insert(0, files.pop(files.index("AAPL.csv")))

##files = ["GOOGL.csv","AAPL.csv"]
shape = (2769,8)

dataset = ConcatData(files[0:50],shape)
print(dataset.shape)



y = list(dataset.iloc[:,1])
y.pop(0)
dataset = dataset.drop(dataset.index[[2766]])

LinrReg = LinearRegression()
rfecv = RFECV(estimator=LinrReg, step=1, cv = 5)

rfecv.fit(dataset, y)
print(rfecv.grid_scores_)
print(rfecv.support_+
	)
for i,c in zip(rfecv.ranking_,list(dataset)):
    print(str(c)+": "+ str(i))

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(0, len(rfecv.grid_scores_)), rfecv.grid_scores_)
plt.show(block = True)

