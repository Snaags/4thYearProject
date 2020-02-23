import torch
import os
import pandas
from model import LSTMModel
from PBTTrainer import RunModel
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import math
import threading
import itertools 
import concurrent.futures
import os
from multiprocessing import Pool
import multiprocessing
from Utils import RandomRange,CreateSets,CreateRandomSets

#########HyperParameter Search Settings###########

#Comment out section to select search type



SearchType = "Random"


hyperparameters = [
	
	[0.0000001,0.001,"log"],			#"lr" 
	[10,200,"int"],			#"hiddenDimension" [
	[1,100,"int"],			#"seq_length" 
	[1,1,"int"],			#"numberLayers"
	[1,1,"int"],			#predict_distance	
	[100,100,"int"],			#"batch_size"
	[25,25,"int"]			#"num_epochs"
					]

"""

SearchType = "Grid"

hyperparameters = {

	"lr" :[0.00001],
	"hiddenDimension": [2],
	"seq_length": [3],
	"numberLayers":[1]
	}
"""
####################################################







path = os.getcwd()


###Import and scale
file = pandas.read_csv(path+"/StockData/AAPL.csv").loc[:,"Open"]
file = np.asarray(file)#convert to numpy array



if SearchType == "Random":

	searchSpace = CreateRandomSets(file,hyperparameters,20)

elif SearchType == "Grid":

	searchSpace = CreateSets(file,hyperparameters)

else:
	print("No search selected, select type from top of file")
	exit()

print(len(searchSpace)," iterations of model to run")


threads = []

start = time.time()

SearchResults = {}



print(os.cpu_count()," CPU logical cores detected")
concurrentTasks = 3 
print("Running",concurrentTasks,"models in parrallel")




if __name__ == "__main__":

	p = Pool(concurrentTasks,maxtasksperchild = 2)
	results = p.starmap(RunModel,searchSpace)
	p.close()
	p.join()





print("Total runtime:",time.time()- start)


##Create input sets##

		






