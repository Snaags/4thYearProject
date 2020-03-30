from Trainer import RunModel
import os
import pandas
import numpy as np
import random
from Utils import MatchDate
path = os.getcwd()

hyperparameters = [0.0001,104,24,1,8,0.00000003,30]
toggle = 0
x = 20
error = []

APPLD = pandas.read_csv(path+"/StockData/AAPL.csv")
APPLD = APPLD[["Date","Close"]]
APPLD = np.asarray(APPLD)#convert to numpy array


APPLC = pandas.read_csv(path+"/StockData/AAPL.csv")
APPLC = np.asarray(APPLC.loc[:,"Close"])
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

file = np.stack((APPLC,USD,CCI,OIL),axis = 1)
file = APPLC
for i in range(20):
	hyperparameters.insert(0,file)
	results = RunModel(*hyperparameters)

	hyperparameters = results["HyperParameters"]
	print(results["EvalScore"])
	error.append(results["EvalScore"])
	print(error)