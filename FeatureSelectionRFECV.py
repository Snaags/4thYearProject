import numpy as np
import pandas

import os
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold


def ConcatData(files):
    hold = pandas.DataFrame()
    
    for i in files:
        data = pandas.read_csv(i)
        if all(pandas.notna(np.min(data))):
                   data = data.loc[:,"Open":]
                   hold = pandas.concat([hold,data],axis = 1)
    return hold
        
 




os.chdir(os.getcwd()+"\\StockData")
files = os.listdir()
dataset = ConcatData(files)
y = list(dataset.iloc[:,1])
y.pop(0)
dataset = dataset.drop(dataset.index[[2766]])

LinrReg = LinearRegression()
rfecv = RFECV(estimator=LinrReg, step=1, cv = 5)

VarT = VarianceThreshold(0.9)
VarT.fit_transform(dataset)

rfecv.fit(dataset, y)
print(rfecv.grid_scores_)

for i,c in zip(rfecv.ranking_,list(dataset)):
    print(str(c)+": "+ str(i))

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(0, len(rfecv.grid_scores_)), rfecv.grid_scores_)
plt.show()

