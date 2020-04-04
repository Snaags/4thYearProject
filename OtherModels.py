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

APPLD = pandas.read_csv(path+"/StockData/AAPL.csv")
APPLD = APPLD[["Date","Close"]]
APPLD = np.asarray(APPLD)#convert to numpy array

"""
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
APPLEPS = MatchDate(APPLD,APPLEPS)
Features.append(APPLEPS)
FeaturesNames.append("APPLEPS")

AAPLEMPLY = pandas.read_csv(path+"/StockData/AAPLEMPLY.csv")
AAPLEMPLY = np.asarray(AAPLEMPLY)#convert to numpy array
AAPLEMPLY = MatchDate(APPLD,AAPLEMPLY)
Features.append(AAPLEMPLY)
FeaturesNames.append("AAPLEMPLY")
"""

APPLC = pandas.read_csv(path+"/StockData/AAPL.csv").loc[:,"Close"]
APPLC = np.asarray(APPLC)#convert to numpy array
Features.append(APPLC)
FeaturesNames.append("APPLC")




"""

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
"""

#APPLVAR = np.var(APPLC)
#file = np.stack((APPLC,APPLRSI,APPLCD,GOOGLC,MSFTC,APPLSMA200,APPLSMA50,APPLSMMA200,APPLSMMA50),1)
#file = np.stack((APPLC,APPLRSI,APPLCD,GOOGLC,MSFTC,APPLSMA200,APPLSMA50,APPLSMMA200,APPLSMMA50),1)



scalers = []
scaled_data = []
for i in Features:
	Features[Features.index(i)] = i[:-8]
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


y = APPLC[8:]

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
print 



from sklearn.metrics import mean_squared_error

mean=mean_squared_error(prediction[:-1],yTest[1:])
print("Error from prediction: ",mean)


plt.plot(yTest[1:],label = "AAPL")
plt.plot(prediction[:-1], label = "Prediction")
plt.title("Linear Regression Prediction of AAPL Stock Value")
plt.ylabel("Price ($)")
plt.xlabel("Trading Days")
#plt.plot(x, label = "input")
plt.legend()
plt.show()



total_error = 0
for i,c in zip(prediction[:-1],yTest[1:]):
	test_lost_score = abs(c - i)
	test_lost_score = test_lost_score/c
	total_error += test_lost_score

total_error = total_error/ len(yTest)
total_error = total_error*100

print("MAPE: ", total_error)


t_old = prediction[0]
tprime_old = yTest[1]
d = 0
for t,tprime in zip(prediction[1:-1],yTest[2:]):



	if ((t - t_old)*(tprime - tprime_old)) >0:
		d += 1

	t_old = t
	tprime_old = tprime 

DS = d*(100/(len(yTest)-1))

print("DS: ",DS)



regressor = Lasso(positive = True)
regressor.fit(Features, y) #training the algorithm

print('Intercept: \n', regressor.intercept_)
for i,c in zip(FeaturesNames,regressor.coef_):
	print(i,": ",c)

Results["Lasso"] = regressor.coef_
prediction = (regressor.predict(Test))



mean=mean_squared_error(prediction[:-1],yTest[1:])
print("Error from prediction: ",mean)

total_error = 0
for i,c in zip(prediction[:-1],yTest[1:]):
	test_lost_score = abs(c - i)
	test_lost_score = test_lost_score/c
	total_error += test_lost_score

total_error = total_error/ len(yTest)
total_error = total_error*100

print("MAPE: ", total_error)

error = total_error*100


t_old = prediction[0]
tprime_old = yTest[1]
d = 0
for t,tprime in zip(prediction[1:-1],yTest[2:]):



	if ((t - t_old)*(tprime - tprime_old)) >0:
		d += 1

	t_old = t
	tprime_old = tprime 

DS = d*(100/(len(yTest)-1))

print("DS: ",DS)


plt.plot(yTest[1:],label = "AAPL")
plt.plot(prediction[:-1], label = "Prediction")
#plt.plot(x, label = "input")
plt.legend()
plt.show()






regressor = LinearSVR(max_iter = 100000, dual = False, loss ="squared_epsilon_insensitive")
regressor.fit(Features, y) #training the algorithm


prediction = (regressor.predict(Test))


plt.plot(yTest[1:],label = "AAPL")
plt.plot(prediction[:-1], label = "Prediction")
plt.title("Support Vector Regression Prediction of AAPL Stock Value")
plt.ylabel("Price ($)")
plt.xlabel("Trading Days")
#plt.plot(x, label = "input")
plt.legend()
plt.show()




total_error = 0
for i,c in zip(prediction[:-1],yTest[1:]):
	test_lost_score = abs(c - i)
	test_lost_score = test_lost_score/c
	total_error += test_lost_score

total_error = total_error/ len(yTest)
total_error = total_error*100

print("MAPE: ", total_error)

t_old = prediction[0]
tprime_old = yTest[1]
d = 0
for t,tprime in zip(prediction[1:-1],yTest[2:]):



	if ((t - t_old)*(tprime - tprime_old)) >0:
		d += 1

	t_old = t
	tprime_old = tprime 

DS = d*(100/(len(yTest)-1))

print("DS: ",DS)


regressor = Ridge()
regressor.fit(Features, y) #training the algorithm

print('Intercept: \n', regressor.intercept_)
for i,c in zip(FeaturesNames,regressor.coef_):
	print(i,": ",c)

Results["Ridge"] = regressor.coef_
prediction = (regressor.predict(Test))

plt.plot(yTest[1:],label = "AAPL")
plt.plot(prediction[:-1], label = "Prediction")
plt.title("Ridge Regression Prediction of AAPL Stock Value")
plt.ylabel("Price ($)")
plt.xlabel("Trading Days")
#plt.plot(x, label = "input")
plt.legend()
plt.show()


total_error = 0
for i,c in zip(prediction[:-1],yTest[1:]):
	test_lost_score = abs(c - i)
	test_lost_score = test_lost_score/c
	total_error += test_lost_score

total_error = total_error/ len(yTest)
total_error = total_error*100

print("MAPE: ", total_error)

t_old = prediction[0]
tprime_old = yTest[1]
d = 0
for t,tprime in zip(prediction[1:-1],yTest[2:]):



	if ((t - t_old)*(tprime - tprime_old)) >0:
		d += 1

	t_old = t
	tprime_old = tprime 

DS = d*(100/(len(yTest)-1))

print("DS: ",DS)

Results["Mean"] = abs(Results.mean(axis = 1))


Results = Results.sort_values(by = ["Mean"])

print(Results)

mean=mean_squared_error(prediction[:-1],yTest[1:])
print("Error from prediction: ",mean)

