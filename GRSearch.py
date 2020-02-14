import torch
import os
import pandas
from model import LSTMModel
from functions import RunModel
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


hyperparameters = {
	
	"lr" :[0.000001,0.01,"log"],
	"hiddenDimension": [150,250,"int"],
	"seq_length": [2,15,"int"],
	"numberLayers":[1,2,"int"]

}

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

	p = Pool(concurrentTasks,maxtasksperchild = 10)
	results = p.starmap(RunModel,searchSpace)
	p.close()
	p.join()


scores = {}

parms = list(hyperparameters.keys())

for valueSet,result in zip(searchSpace,results):
	name = []

	for value, key in zip(valueSet[1:], parms):

		x = (str(key)+": "+str(value))

		name.append(x)


	name = ",".join(name)
	scores[name] = result

print("#####Scores from search#####")






x = len(scores)
ranking = 1

while x > 0:

	score = min(scores.values())
	name = list(scores.keys())[list(scores.values()).index(score)]

	print(ranking,". ",name," -- MSRE: ", score)
	ranking += 1 
	scores.pop(name)
	if ranking > 3:
		os.remove("Graphs/"+str(score)+".png") 
	x -= 1


"""
for i in searchSpace:
	processes.append(Process(target = RunModel,args= (file,i[0],i[1],i[2],i[3])))

	process = processes[-1]
	process.start()
	process.join()

	print(len(multiprocessing.active_children()))
	while int(active[-2]) > 6:
		time.sleep(3)
		print(len(multiprocessing.active_children()))
"""

"""
for i in searchSpace[0:4]:

	if threading.active_count() < 1:
		threads.append(threading.Thread(target = RunModel, args = (file,i[0],i[1],i[2],i[3])))
		threads[-1].start()
		print("starting thread",len(threads))
	else:
		print("starting main thread")
		RunModel(file,i[0],i[1],i[2],i[3])

	print("current number of threads",threading.active_count())
	
"""

print("Total runtime:",time.time()- start)


##Create input sets##

		






