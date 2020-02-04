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

def CreateSets(Hyperparms):
	sets = []
	subset = []

	parameters = len(Hyperparms)
	for i in list(itertools.permutations(range(len(list(Hyperparms.values())[0])))):

		for parms,indx in zip(Hyperparms,i):
			subset.append(Hyperparms[parms][int(indx)])
		
		subset.insert(0,file)
		subset = tuple(subset)
		sets.append(subset)
		subset = []

	return sets


path = os.getcwd()


###Import and scale
file = pandas.read_csv(path+"/StockData/AAPL.csv").loc[:,"Open"]
file = np.asarray(file)#convert to numpy array






print("ready")

hyperparameters = {

	"lr" :[0.00001,0.001,0.01],
	"hiddenDimension": [175,200,225],
	#"numberLayers": [1,2,3],
	"seq_length": [2,4,8]
	}

searchSpace = CreateSets(hyperparameters)
print(len(searchSpace))

threads = []

start = time.time()

SearchResults = {}



print(os.cpu_count())





if __name__ == "__main__":

	p = Pool(4,maxtasksperchild = 10)
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

		






