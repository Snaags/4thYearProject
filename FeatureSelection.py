import numpy as np
import pandas
import os
import sys
sys.path.append(r'C:\Users\chris\Anaconda3\Lib\site-packages')

from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPRegressor
import multiprocessing
from sklearn.preprocessing import MinMaxScaler
#import StockData

path = os.getcwd()


def SMA(file,N):
	output = []
	n = 1
	for i in range(len(file)):
		if (i - N) < 0:		
			m = i
			if i == 0:
				n = 1
			else:
				n = m
		else:
			n = N
			m = N
		output.append(sum(file[(i-m):i])/n)

	return output

def RSI(file,N):
	output = []
	U = []
	D = []
	RSI = 0
	i_old = 0
	

	for i in file:
		v = i - i_old

		if v > 0:
			U.append(v)
			D.append(0)
		else:
			D.append(-v)
			U.append(0)

		i_old = i

		if len(U) > 14:
			U.pop(0)
			D.pop(0)
			RS = SMMA(U,N)/SMMA(D,N)
			RSI = 100 - (100/(1+RS))
		
		output.append(RSI)

	return output 

def SMMA(data,N):

	MMA = 0

	def step(MMA, new_sample, N):

		return ((N - 1)*MMA + new_sample)/N

	
	for i in data:
		MMA = step(MMA,i,N)

	return MMA

def SMMA_Seq(data,N):
	output = []
	MMA = 0

	def step(MMA, new_sample, N):

		return ((N - 1)*MMA + new_sample)/N

	
	for i in data:
		MMA = step(MMA,i,N)
		output.append(MMA)

	return output

def ConcatData(filename1, filename2):
    
	file1 = pandas.read_csv(filename1)
	file2 = pandas.read_csv(filename2)

	file1data = file1.loc[:,"Open":]
	file2data = file2.loc[:,"Open":]

	#file1date = file1.loc[:,"Date"]
	#file2date = file2.loc[:,"Date"]

	return pandas.concat([file1data,file2data],axis = 1)


##Takes dataframe in the shape (n,2):date,data

def MatchDate(data1,data2):
	output = []
	outputb = []
	hold2 = 0
	datelist = []
	for c in data2:
		flag = False
		for i in data1:
			if i[0] == c[0]:
				datelist.append([i[0],c[1]])
				flag = True
				break

		if flag == False:
			hold = list(c[0])
			if "".join(hold[8:]) == "31" or "".join(hold[8:]) == "30":
				if "".join(hold[5:7]) == "12":
					hold[3] = str(int(hold[3]) + 1)
					hold[5:7] = "01"
				hold[8:] = "02"

			hold[9] = str(int(hold[9]) + 1)	
			for i in data1:
				if i[0] == "".join(hold):
					datelist.append([i[0],c[1]])
					flag = True

		if flag == False:
			hold = list(c[0])
			if "".join(hold[8:]) == "31" or "".join(hold[8:]) == "30":
				if "".join(hold[5:7]) == "12":
					hold[3] = str(int(hold[3]) + 1)
					hold[5:7] = "01"
				hold[8:] = "03"

			hold[9] = str(int(hold[9]) + 2)	
			for i in data1:
				if i[0] == "".join(hold):
					datelist.append([i[0],c[1]])
					flag = True

	for i in data1:
		for c in datelist:
			if i[0] == c[0]:


				hold2 = c[1]
				if type(hold2) == str:
					hold2 = hold2.strip("$")
					hold2 = float(hold2)
		output.append(hold2)
	output = np.asarray(output)
	return output

def PercentChange(data,n):
	data = list(data)
	output = [0]*n


	for i in data[n:]:
		output.append(((i-data[data.index(i)-n])/i)*100)
	return output


def DailyChange(open,close):
	output = []
	for i,c in zip(open,close):
		output.append(((i-c)/i)*100)
	return output



###Import and scale
Features = []
FeaturesNames = []

APPLC = pandas.read_csv(path+"/StockData/AAPL.csv").loc[:,"Close"]
APPLC = np.asarray(APPLC)#convert to numpy array
Features.append(APPLC)
FeaturesNames.append("APPL Close")


APPLO = pandas.read_csv(path+"/StockData/AAPL.csv").loc[:,"Open"]
APPLO = np.asarray(APPLO)#convert to numpy array
Features.append(APPLO)
FeaturesNames.append("APPL Open")


APPLD = pandas.read_csv(path+"/StockData/AAPL.csv")
APPLD = APPLD[["Date","Close"]]
APPLD = np.asarray(APPLD)#convert to numpy array


MSFTC = pandas.read_csv(path+"/StockData/MSFT.csv").loc[:,"Close"]
MSFTC = np.asarray(MSFTC)#convert to numpy array
Features.append(MSFTC)
FeaturesNames.append("MSFT Close")

GOOGLC = pandas.read_csv(path+"/StockData/GOOGL.csv").loc[:,"Close"]
GOOGLC = np.asarray(GOOGLC)#convert to numpy array
Features.append(GOOGLC)
FeaturesNames.append("GOOGL Close")

DJIC = pandas.read_csv(path+"/StockData/^DJI.csv").loc[:,"Close"]
DJIC = np.asarray(DJIC)#convert to numpy array
Features.append(DJIC)
FeaturesNames.append("DJI Close")

CCI = pandas.read_csv(path+"/StockData/^CCI.csv")
CCI = np.asarray(CCI)#convert to numpy array
CCI = MatchDate(APPLD,CCI)
Features.append(CCI)
FeaturesNames.append("Consumer Confidence Index (US)")

APPLAST = pandas.read_csv(path+"/StockData/AAPLAST.csv")
APPLAST = np.asarray(APPLAST)#convert to numpy array
APPLAST = MatchDate(APPLD,APPLAST)
Features.append(APPLAST)
FeaturesNames.append("APPL Asset Value")

APPLPSR = pandas.read_csv(path+"/StockData/AAPLPSR.csv")
APPLPSR = np.asarray(APPLPSR)#convert to numpy array
APPLPSR = MatchDate(APPLD,APPLPSR)
Features.append(APPLPSR)
FeaturesNames.append("APPL Price to Share Ratio")

USD = pandas.read_csv(path+"/StockData/USD.csv").loc[:,"Date":"Price"]
USD = np.asarray(USD)#convert to numpy array
USD = MatchDate(APPLD,USD)
Features.append(USD)
FeaturesNames.append("USD Exchange")

OIL = pandas.read_csv(path+"/StockData/OIL.csv")
OIL = OIL[["Date","Price"]]
OIL = np.asarray(OIL)#convert to numpy array

OIL = MatchDate(APPLD,OIL)
Features.append(OIL)
FeaturesNames.append("OIL")

APPLPSP = pandas.read_csv(path+"/StockData/AAPLSPS.csv")
APPLPSP = np.asarray(APPLPSP)#convert to numpy array
APPLPSP = MatchDate(APPLD,APPLPSP)
Features.append(APPLPSP)
FeaturesNames.append("APPL Sales Per Share")

APPLEPS = pandas.read_csv(path+"/StockData/AAPLEPS.csv")
APPLEPS = np.asarray(APPLEPS)#convert to numpy array
APPLEPS = MatchDate(APPLD,APPLEPS)
Features.append(APPLEPS)
FeaturesNames.append("APPL Earnings Per Share")


AAPLEMPLY = pandas.read_csv(path+"/StockData/AAPLEMPLY.csv")
AAPLEMPLY = np.asarray(AAPLEMPLY)#convert to numpy array
AAPLEMPLY = MatchDate(APPLD,AAPLEMPLY)
Features.append(AAPLEMPLY)
FeaturesNames.append("AAPL Employees")

AAPLEREV = pandas.read_csv(path+"/StockData/AAPLREV.csv")
AAPLEREV = np.asarray(AAPLEREV)#convert to numpy array
AAPLEREV = MatchDate(APPLD,AAPLEREV)
Features.append(AAPLEREV)
FeaturesNames.append("AAPLE Revenue")

INTEREST = pandas.read_csv(path+"/StockData/INTEREST.csv")
INTEREST = np.asarray(INTEREST)#convert to numpy array
INTEREST = MatchDate(APPLD,INTEREST)
Features.append(INTEREST)
FeaturesNames.append("US Interest Rates")


APPLV = pandas.read_csv(path+"/StockData/AAPL.csv").loc[:,"Volume"]
APPLV = np.asarray(APPLV)#convert to numpy array
Features.append(APPLV)
FeaturesNames.append("APPL Volume")




RawTickers = [APPLC]
RawTickersN = ["APPL"]


for i,n in zip(RawTickers,RawTickersN):

	Features.append(RSI(i,14))
	print("RSI length", 14, "completed for ", i)
	FeaturesNames.append(str(str(n)+" RSI "+str(14)+" Days"))




#APPLVAR = np.var(APPLC)
#file = np.stack((APPLC,APPLRSI,APPLCD,GOOGLC,MSFTC,APPLSMA200,APPLSMA50,APPLSMMA200,APPLSMMA50),1)
#file = np.stack((APPLC,APPLRSI,APPLCD,GOOGLC,MSFTC,APPLSMA200,APPLSMA50,APPLSMMA200,APPLSMMA50),1)



scalers = []
scaled_data = []
for i,c in zip(Features,FeaturesNames):
	print(c,": ",len(i))
	Features[Features.index(i)] = i[:-1]
Features = np.asarray(Features)

"""
for i in range(len(Features[0,:])):
	scalers.append(MinMaxScaler(feature_range=(-1, 10)))
#scaler = MinMaxScaler(feature_range=(-1, 1))	#scale data
#scaler1 = MinMaxScaler(feature_range=(-1, 1))	#scale data	
#scale2 = MinMaxScaler(feature_range=(-1, 1))	#scale data
#scaler3 = MinMaxScaler(feature_range=(-1, 1))	#scale data	
#scaler4 = MinMaxScaler(feature_range=(-1, 1))	#scale data
index_count = 0
for i in scalers:
	scaled_data.append(i.fit_transform(Features[:,index_count].reshape(-1, 1)))
	index_count +=1 

Features = np.stack((scaled_data),1)

"""

y = APPLC[1:]


"""
y = scalers[0].fit_transform(y.reshape(-1, 1))
"""

#dataset = np.asarray(Features)
#print(dataset.shape)

Results = pandas.DataFrame(index = FeaturesNames)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

def test_train_split2(file,train_size):
	x = file.shape
	print(x)
	x = int(x[1]*train_size)
	X_train = file[:,:x]
	X_test = file[:,x:]

	return X_train, X_test

def test_train_split(file,train_size):

	X_train = file[:int(len(file)*train_size)]
	X_test = file[int(len(file)*train_size):]

	return X_train, X_test





Features = np.matrix(Features)
Features = np.transpose(Features)





from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr

K = SelectKBest(score_func=f_regression, k="all")
K.fit(Features,y)
x = np.arange(len(FeaturesNames))
plt.title("Regressor Corrolation")
plt.xlabel("Features")
plt.ylabel("F-Score")
plt.bar(x,height = K.scores_)
plt.xticks(x, FeaturesNames,rotation=90)
plt.show()
plt.bar(x,height = K.pvalues_)
plt.xticks(x, FeaturesNames,rotation=90)
plt.show()

K = SelectKBest(score_func=f_regression, k="all")
K.fit(Features[:,2:],y)
x = np.arange(len(FeaturesNames[2:]))
plt.title("Regressor Corrolation")
plt.xlabel("Features")
plt.ylabel("F-Score")
plt.bar(x,height = K.scores_)
plt.xticks(x, FeaturesNames[2:],rotation=90)
plt.show()
plt.bar(x,height = K.pvalues_)
plt.xticks(x, FeaturesNames[2:],rotation=90)
plt.show()


K = SelectKBest(score_func=mutual_info_regression, k="all")
K.fit(Features,y)
x = np.arange(len(FeaturesNames))
plt.title("Mutual Information Estimate")
plt.bar(x,height = K.scores_)
plt.xlabel("Features")
plt.ylabel("Information Estimate")
plt.xticks(x, FeaturesNames,rotation=90)
plt.show()














scores = pandas.DataFrame(index = FeaturesNames)
CV = True
if CV == True:



	RidgeReg = Ridge()
	LinrReg = LinearRegression(normalize = True)
	lasso = Lasso()
	SVR = LinearSVR(max_iter = 100000, dual = False, loss ="squared_epsilon_insensitive") 
	Tree =DecisionTreeRegressor()

		##Linear Regression
	rfe = RFECV(estimator=LinrReg, step=1,min_features_to_select = 8,cv = 5)
	rfe.fit(Features, y)
	for i,c in zip(rfe.ranking_,list(FeaturesNames)):
	    print(str(c)+": "+ str(i))
	print(rfe.support_)
	scores["LinearRegression"] = rfe.ranking_



	plt.figure()
	plt.title("LinearRegression")
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross validation score (nb of correct classifications)")
	plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
	plt.show()

	##Ridge Regression
	rfe = RFECV(estimator=RidgeReg, step=1,min_features_to_select = 8,cv = 5)
	rfe.fit(Features, y)
	for i,c in zip(rfe.ranking_,list(FeaturesNames)):
	    print(str(c)+": "+ str(i))
	print(rfe.support_)
	scores["RidgeRegression"] = rfe.ranking_



	plt.figure()
	plt.title("Ridge Regression")
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross validation score (nb of correct classifications)")
	plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
	plt.show()



	rfe = RFECV(estimator=lasso, step=1,min_features_to_select = 8,cv = 5)
	rfe.fit(Features, y)
	for i,c in zip(rfe.ranking_,list(FeaturesNames)):
	    print(str(c)+": "+ str(i))
	print(rfe.support_)
	scores["Lasso"] = rfe.ranking_
	plt.figure()
	plt.title("Lasso")
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross validation score (nb of correct classifications)")
	plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
	plt.show()




	rfe = RFECV(estimator=Tree, step=1,min_features_to_select = 8,cv = 15)
	rfe.fit(Features, y)


	print(rfe.support_)
	print(rfe.grid_scores_[0])
	scores["Tree"] = rfe.ranking_

	plt.figure()
	plt.title("Tree")
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross validation score (nb of correct classifications)")
	plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
	plt.show()

	rfe = RFECV(estimator=SVR,min_features_to_select = 8, step=1,cv = 10)
	rfe.fit(Features, y)


	print(rfe.support_)
	print(rfe.grid_scores_[0])
	scores["SVR"] = rfe.ranking_

	plt.figure()
	plt.title("SVR")
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross validation score (nb of correct classifications)")
	plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
	plt.show()
	mean =  abs(scores.mean(axis = 1))
	scores["Times Selected"] = scores.isin([1]).sum(axis=1)
	scores["Mean"] = mean 
	scores = scores.sort_values(by = ["Times Selected","Mean"],ascending = [False,True])
	scores.to_csv("REFCV.csv")
	print(scores)



		#####Linear Regression#########

regressor = LinearRegression()
regressor.fit(Features, y) #training the algorithm
Results["Linear Regression"] = abs(regressor.coef_)




		#####Lasso#########


regressor = Lasso(positive = True)
regressor.fit(Features, y) #training the algorithm
Results["Lasso"] = abs(regressor.coef_)



		#####Ridge Regression#########

regressor = Ridge()
regressor.fit(Features, y) #training the algorithm

Results["Ridge"] = abs(regressor.coef_)


"""
regressor = LinearSVR()
regressor.fit(Features, y) #training the algorithm

Results["SVR"] = abs(regressor.coef_)

regressor = LinearSVR(max_iter = 100000, dual = False, loss ="squared_epsilon_insensitive")
regressor.fit(Features, y) #training the algorithm

Results["SVR2"] = abs(regressor.coef_)
"""

	###Tree

regressor = DecisionTreeRegressor()
regressor.fit(Features, y) #training the algorithm

Results["Tree"] = abs(regressor.feature_importances_)


Results["Mean"] = abs(Results.mean(axis = 1))


Results = Results.sort_values(by = ["Mean"],ascending = False)

Results.to_csv("FeatureSelectionModelsResults.csv")
print(Results)

