from Trainer import RunModel
import os
import pandas
import numpy as np
import random
from Utils import MatchDate
path = os.getcwd()

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

def PercentChange(data,n):
	data = list(data)
	output = [0]*n


	for i in data[n:]:
		output.append(((i-data[data.index(i)-n])/i)*100)
	return output

def SMMA_Seq(data,N):
	output = []
	MMA = 0

	def step(MMA, new_sample, N):

		return ((N - 1)*MMA + new_sample)/N

	
	for i in data:
		MMA = step(MMA,i,N)
		output.append(MMA)

	return output

hyperparameters = [0.001,20,2,1,4,0.0000001,100]
toggle = 0
x = 20
error = []

APPLC = pandas.read_csv(path+"/StockData/AAPL.csv")
APPLC = np.asarray(APPLC.loc[:,"Close"])

APPLO = pandas.read_csv(path+"/StockData/AAPL.csv")
APPLO = np.asarray(APPLO.loc[:,"Volume"])

APPLD = pandas.read_csv(path+"/StockData/AAPL.csv")
APPLD = APPLD[["Date","Close"]]
APPLD = np.asarray(APPLD)#convert to numpy array


INTEREST = pandas.read_csv(path+"/StockData/INTEREST.csv")
INTEREST = np.asarray(INTEREST)#convert to numpy array
INTEREST = MatchDate(APPLD,INTEREST)

OIL = pandas.read_csv(path+"/StockData/OIL.csv")
OIL = OIL[["Date","Price"]]
OIL = np.asarray(OIL)#convert to numpy array
OIL = MatchDate(APPLD,OIL)
print(len(APPLC))
print(len(OIL))


CCI = pandas.read_csv(path+"/StockData/^CCI.csv")
CCI = np.asarray(CCI)#convert to numpy array
CCI = MatchDate(APPLD,CCI)

USD = pandas.read_csv(path+"/StockData/USD.csv").loc[:,"Date":"Price"]
USD = np.asarray(USD)#convert to numpy array
USD = MatchDate(APPLD,USD)

APPLEPS = pandas.read_csv(path+"/StockData/AAPLEPS.csv")
APPLEPS = np.asarray(APPLEPS)#convert to numpy array
APPLEPS = MatchDate(APPLD,APPLEPS)

APPLPSR = pandas.read_csv(path+"/StockData/AAPLPSR.csv")
APPLPSR = np.asarray(APPLPSR)#convert to numpy array
APPLPSR = MatchDate(APPLD,APPLPSR)

DJI = pandas.read_csv(path+"/StockData/^DJI.csv")
DJI = np.asarray(DJI.loc[:,"Close"])


DJI = pandas.read_csv(path+"/StockData/MSFT.csv")
DJI = np.asarray(DJI.loc[:,"Close"])

"""
while x > 0:

	file = pandas.read_csv(path+"/StockData/"+os.listdir("StockData")[random.randint(0,len(os.listdir("StockData")))])
	if len(file) !=  2769:
		continue
	if toggle > 10:
		file = APPLC#convert to numpy array
		x -=1
		toggle =0
		hyperparameters[6] = 20

	else:
		hyperparameters[6] = 20
	toggle +=1
	
	file = np.asarray(file.loc[:,"Close"])#convert to numpy array
	file = np.hstack((file,OIL))
	hyperparameters.insert(0,file)

	results = RunModel(*hyperparameters)

	hyperparameters = results["HyperParameters"]
	print(results["EvalScore"])
	if toggle == 1:
		error.append(results["EvalScore"])

APPLC = pandas.read_csv(path+"/StockData/AAPL.csv")
APPLC = np.asarray(APPLC)#convert to numpy array
hyperparameters[0] = APPLC


"""

file = np.stack((APPLC,USD),axis = 1)
#file = APPLC
error = {}




hyperparameters = [
	
	0.006,		#"lr" 
	224,			#"hiddenDimension" 
	14,				#"seq_length" 
	1,				#"numberLayers"	
	32,				#"batch_size"
	0,			#Regularization
	
	120		#"num_epochs"
	#0.0005			#dropout
					]
counter = 0

for i in range(15):

	counter +=1
	hyperparameters.insert(0,file)
	results = RunModel(*hyperparameters)

	hyperparameters = results["HyperParameters"]
	hyperparameters[6] = 10

	print(results["EvalScore"])

	error[results["EvalScore"]] = counter
	print(error)

