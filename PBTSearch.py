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
import random
from Utils import CreateRandomSets
path = os.getcwd()
#########HyperParameter Search Settings###########


SearchType = "Random"


hyperparameters = {
	
	"lr" :[0.0000001,0.001,"log"],
	"hiddenDimension": [10,250,"int"],
	"seq_length": [2,15,"int"],
	"numberLayers":[1,1,"int"],
	"predict_distance":[1,1,"int"],
	"num_epochs":[2,2,"int"]
}



def Exploit(Network):
	#Find high proforming networks
	#Select network based on a probability distrobution
	#Extract weights and hyperparameters
	#Apply to the targeted network
	pass

def Explore():

	pass


###Import and scale
file = pandas.read_csv(path+"/StockData/AAPL.csv").loc[:,"Open"]
file = np.asarray(file)#convert to numpy array


if SearchType == "Random":

	searchSpace = CreateRandomSets(file,hyperparameters,4)

print(searchSpace)
#searchSpace[0].append('/home/snaags/4thYearProject/Models/7e-06.pth')
print(searchSpace)




print(len(searchSpace)," iterations of model to run")

threads = []

start = time.time()

SearchResults = {}

print(searchSpace)

print(os.cpu_count()," CPU logical cores detected")
concurrentTasks = 4
print("Running",concurrentTasks,"models in parrallel")



if __name__ == "__main__":


	##Intial random search steps
	p = Pool(concurrentTasks,maxtasksperchild = 1)
	results = p.starmap(RunModel,searchSpace)
	p.close()
	p.join()

	min_error = 100000
	pos = "NA"
	for i in results:
		if i["EvalScore"] < min_error:
			min_error = i["EvalScore"]
			pos = results.index(i)
	Search = []
	search = list(results[pos]["HyperParameters"].values())
	search.append(results[pos]["ModelState"])
	search.insert(0,file)
	Search.append(search)
	print(search)

	p = Pool(concurrentTasks,maxtasksperchild = 1)
	results2 = p.starmap(RunModel,Search)
	p.close()
	p.join()
	print(results2)

scores = {}

parms = list(hyperparameters.keys())


exit()

for valueSet,result in zip(searchSpace,results):
	name = []

	for value, key in zip(valueSet[1:], parms):

		x = (str(key)+": "+str(value))

		name.append(x)


	name = ",".join(name)
	scores[name] = result

print("#####Scores from search#####")



exit()


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

