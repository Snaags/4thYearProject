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
	hold = 0
	for i in data1:
		for c in data2:
			if i[0] == c[0]:
				hold = c[1]
				if type(hold) == str:
					hold = hold.strip("$")
					hold = float(hold)
		output.append(hold)
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
APPLC = pandas.read_csv(path+"/StockData/AAPL.csv")
APPLC = APPLC[["Date","Close"]]
APPLC = np.asarray(APPLC)#convert to numpy array


MSFTC = pandas.read_csv(path+"/StockData/MSFT.csv").loc[:,"Close"]
MSFTC = np.asarray(MSFTC)#convert to numpy array
Features.append(MSFTC)
FeaturesNames.append("MSFTC")

GOOGLC = pandas.read_csv(path+"/StockData/GOOGL.csv").loc[:,"Close"]
GOOGLC = np.asarray(GOOGLC)#convert to numpy array
Features.append(GOOGLC)
FeaturesNames.append("GOOGLC")

DJIC = pandas.read_csv(path+"/StockData/^DJI.csv").loc[:,"Close"]
DJIC = np.asarray(DJIC)#convert to numpy array
Features.append(DJIC)
FeaturesNames.append("DJIC")

APPLEPS = pandas.read_csv(path+"/StockData/AAPLEPS.csv")
APPLEPS = np.asarray(APPLEPS)#convert to numpy array
APPLEPS = MatchDate(APPLC,APPLEPS)
Features.append(APPLEPS)
FeaturesNames.append("APPLEPS")

AAPLEMPLY = pandas.read_csv(path+"/StockData/AAPLEMPLY.csv")
AAPLEMPLY = np.asarray(AAPLEMPLY)#convert to numpy array
AAPLEMPLY = MatchDate(APPLC,AAPLEMPLY)
Features.append(AAPLEMPLY)
FeaturesNames.append("AAPLEMPLY")


APPLC = pandas.read_csv(path+"/StockData/AAPL.csv").loc[:,"Close"]
APPLC = np.asarray(APPLC)#convert to numpy array
Features.append(APPLC)
FeaturesNames.append("APPLC")

APPLV = pandas.read_csv(path+"/StockData/AAPL.csv").loc[:,"Volume"]
APPLV = np.asarray(APPLV)#convert to numpy array
Features.append(APPLV)
FeaturesNames.append("APPLV")

APPLO = pandas.read_csv(path+"/StockData/AAPL.csv").loc[:,"Open"]
APPLO = np.asarray(APPLO)#convert to numpy array

AAPLEREV = pandas.read_csv(path+"/StockData/AAPLREV.csv")
AAPLEREV = np.asarray(AAPLEREV)#convert to numpy array
AAPLEREV = MatchDate(APPLC,AAPLEREV)
Features.append(AAPLEREV)
FeaturesNames.append("AAPLEREV")



RawTickers = [APPLC,DJIC,GOOGLC,MSFTC]
RawTickersN = ["APPLC","DJIC","MSFTC","GOOGLC"]


for i,n in zip(RawTickers,RawTickersN):
	for c in range(10,22,4):
		Features.append(RSI(i,c))
		print("RSI length", c, "completed for ", i)
		FeaturesNames.append(str("RSI"+str(n)+str(c)))
	for c in range(2,10,2):
		Features.append(PercentChange(i,c))
		print("Percent change length", c, "completed for ", i)
		FeaturesNames.append(str("PrcntChge"+str(n)+str(c)))

	Features.append(DailyChange(APPLO,APPLC))
	FeaturesNames.append(str("DC"+str(n)))


#APPLVAR = np.var(APPLC)
#file = np.stack((APPLC,APPLRSI,APPLCD,GOOGLC,MSFTC,APPLSMA200,APPLSMA50,APPLSMMA200,APPLSMMA50),1)
#file = np.stack((APPLC,APPLRSI,APPLCD,GOOGLC,MSFTC,APPLSMA200,APPLSMA50,APPLSMMA200,APPLSMMA50),1)



scalers = []
scaled_data = []
for i in Features:
	Features[Features.index(i)] = i[:-1]
Features = np.asarray(Features)


"""
for i in range(len(Features[0,:])):
	scalers.append(MinMaxScaler(feature_range=(-1, 1)))
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
scalers[6].fit_transform(y.reshape(-1, 1))
"""

#dataset = np.asarray(Features)
#print(dataset.shape)

Results = pandas.DataFrame(index = FeaturesNames)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVR
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

Features, Test = test_train_split2(Features, 0.85)

y, yTest = test_train_split(y, 0.85)


Features = np.matrix(Features)
Features = np.transpose(Features)

regressor = LinearRegression()
regressor.fit(Features, y) #training the algorithm

print('Intercept: \n', regressor.intercept_)
for i,c in zip(FeaturesNames,regressor.coef_):
	print(i,": ",c)

Results["Linear Regression"] = regressor.coef_



Test = np.matrix(Test)
Test = np.transpose(Test)

prediction = (regressor.predict(Test))




from sklearn.metrics import mean_squared_error

mean=mean_squared_error(prediction,yTest)
print("Error from prediction: ",mean)


plt.plot(yTest,label = "AAPL")
plt.plot(prediction, label = "Prediction")
#plt.plot(x, label = "input")
plt.legend()
plt.show()



regressor = Lasso(positive = True)
regressor.fit(Features, y) #training the algorithm

print('Intercept: \n', regressor.intercept_)
for i,c in zip(FeaturesNames,regressor.coef_):
	print(i,": ",c)

Results["Lasso"] = regressor.coef_
prediction = (regressor.predict(Test))



mean=mean_squared_error(prediction,yTest)
print("Error from prediction: ",mean)

total_error = 0
for i,c in zip(prediction,yTest):
	test_lost_score = abs(c - i)
	test_lost_score = test_lost_score/c
	total_error += test_lost_score

total_error = total_error/ len(yTest)
total_error = total_error*100

print("MAPE: ", total_error)

error = total_error*100

plt.plot(yTest,label = "AAPL")
plt.plot(prediction, label = "Prediction")
#plt.plot(x, label = "input")
plt.legend()
plt.show()


regressor = LinearSVR(max_iter = 100000, dual = False, loss ="squared_epsilon_insensitive")
regressor.fit(Features, y) #training the algorithm

print('Intercept: \n', regressor.intercept_)
for i,c in zip(FeaturesNames,regressor.coef_):
	print(i,": ",c)

Results["SVR"] = regressor.coef_
prediction = (regressor.predict(Test))


regressor = Ridge()
regressor.fit(Features, y) #training the algorithm

print('Intercept: \n', regressor.intercept_)
for i,c in zip(FeaturesNames,regressor.coef_):
	print(i,": ",c)

Results["Ridge"] = regressor.coef_
prediction = (regressor.predict(Test))




Results["Mean"] = abs(Results.mean(axis = 1))


Results = Results.sort_values(by = ["Mean"])

print(Results)











scores = {}
CV = True
if CV == True:



	RidgeReg = Ridge(normalize = True)
	LinrReg = LinearRegression(normalize = True)
	Lasso = Lasso(alpha = 0.75,normalize = True)
	SVR = LinearSVR(max_iter = 100000, dual = False, loss ="squared_epsilon_insensitive") 

		##Linear Regression
	rfe = RFECV(estimator=LinrReg, step=1,cv = 5)
	rfe.fit(Features, y)
	for i,c in zip(rfe.ranking_,list(FeaturesNames)):
	    print(str(c)+": "+ str(i))
	print(rfe.support_)
	scores["LinearRegression"] = rfe.ranking_



	plt.figure()
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross validation score (nb of correct classifications)")
	plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
	plt.show()

	##Ridge Regression
	rfe = RFECV(estimator=RidgeReg, step=1,cv = 5)
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



	rfe = RFECV(estimator=Lasso, step=1,cv = 5)
	rfe.fit(Features, y)
	for i,c in zip(rfe.ranking_,list(FeaturesNames)):
	    print(str(c)+": "+ str(i))
	print(rfe.support_)
	scores["Lasso"] = rfe.ranking_
	plt.figure()
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross validation score (nb of correct classifications)")
	plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
	plt.show()






	rfe = RFECV(estimator=SVR, step=1,cv = 5)

	rfe.fit(Features, y)
	for i,c in zip(rfe.ranking_,list(FeaturesNames)):
	    print(str(c)+": "+ str(i))
	print(rfe.ranking_)
	print(rfe.support_)
	print(rfe.grid_scores_[0])
	scores["SVR"] = rfe.ranking_
	plt.figure()
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross validation score (nb of correct classifications)")
	plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
	plt.show()

else:

	lasso = Lasso(normalize = True, alpha = 0.85)
	lasso.fit(Features, y)

	for c,i in zip(FeaturesNames,np.abs(lasso.coef_)):

		print(c,": ", i)
	plt.plot(lasso.predict(Features))
	RidgeReg = Ridge(normalize = True)
	LinrReg = LinearRegression(normalize = True)
	#Lasso = Lasso(normalize = True)
	SVR = LinearSVR(max_iter = 100000, dual = False, loss ="squared_epsilon_insensitive") 


	##Linear Regression
	rfe = RFE(estimator=LinrReg, step=1)
	rfe.fit(Features, y)
	for i,c in zip(rfe.ranking_,list(FeaturesNames)):
		print(str(c)+": "+ str(i))
	print( "Features sorted by their rank:")
	print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), FeaturesNames)))
	print(rfe.score(Features,y))
	linrank = rfe.ranking_

	##Ridge Regression
	rfe = RFE(estimator=RidgeReg, step = 1)
	rfe.fit(Features, y)
	for i,c in zip(rfe.ranking_,list(FeaturesNames)):
	    print(str(c)+": "+ str(i))


	ridgerank = rfe.ranking_


	rfe = RFE(estimator=lasso, step=1)
	rfe.fit(Features, y)
	for i,c in zip(rfe.ranking_,list(FeaturesNames)):
	    print(str(c)+": "+ str(i))
	print(rfe.ranking_)


	lassorank = rfe.ranking_

	rfe = RFE(estimator=SVR, step=1)
	rfe.fit(Features, y)
	for i,c in zip(rfe.ranking_,list(FeaturesNames)):
	    print(str(c)+": "+ str(i))
	print(rfe.ranking_)

	SVRrank = rfe.ranking_
	plt.figure()
	plt.plot(linrank, label = "Linear Regression")
	plt.plot(lassorank,label = "Lasso Regression")
	plt.plot(ridgerank,label = "Ridge Regression")
	plt.plot(SVRrank,label = "Support Vector Regression")
	plt.legend()
	plt.show()

