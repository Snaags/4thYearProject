from Trainer import RunModel
import os
import pandas
import numpy as np
import random
from Utils import MatchDate
path = os.getcwd()
APPLC = pandas.read_csv(path+"/StockData/AAPL.csv")
hyperparameters = [0.005,60,20,8,256,0.0000003,5]
toggle = 0
x = 20
error = []

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

for i in range(10):
	results = RunModel(*hyperparameters)

	hyperparameters = results["HyperParameters"]
	hyperparameters[0] = APPLC
	print(results["EvalScore"])
	error.append(results["EvalScore"])
	print(error)