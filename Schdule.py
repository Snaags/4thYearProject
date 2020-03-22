from Trainer import RunModel
import os
import pandas
import numpy as np
path = os.getcwd()
APPLC = pandas.read_csv(path+"/StockData/AAPL.csv").loc[:,"Close"]
hyperparameters = [0.0006,260,35,1,16,0.00001,5]
toggle = False
for i in os.listdir("StockData"):

	file = pandas.read_csv(path+"/StockData/"+i)
	if len(file) < 1000:
		continue
	if toggle == True:
		file = np.asarray(APPLC)#convert to numpy array
	
	toggle = not toggle
	
	file = np.asarray(file.loc[:,"Close"])#convert to numpy array
	print(file)

	hyperparameters.insert(0,file)
	print(hyperparameters)
	print("Running training with",i)
	results = RunModel(*hyperparameters)

	hyperparameters = results["HyperParameters"]
	print(results["EvalScore"])


APPLC = pandas.read_csv(path+"/StockData/AAPL.csv").loc[:,"Close"]
APPLC = np.asarray(APPLC)#convert to numpy array
hyperparameters[0] = APPLC

for i in range(10):
	results = RunModel(*hyperparameters)

	hyperparameters = results["HyperParameters"]
	hyperparameters[0] = APPLC
	print(results["EvalScore"])