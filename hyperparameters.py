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
import random



def Exploit(Network):
	#Find high proforming networks
	#Select network based on a probability distrobution
	#Extract weights and hyperparameters
	#Apply to the targeted network
	pass

def Explore():

	pass



##Creates a range of values between 2 boundries. Types = log, int 
def RandomRange(min, max, steps,types):
	output = []

	if types == "int":
		while steps >= 1:
			output.append(random.randint(min,max))
			steps -=1

	if types == "log":
		minexp = np.log10(np.abs(min))
		print(minexp)
		maxexp = np.log10(np.abs(max))
		print(maxexp)
		while steps >= 1:
			output.append(random.randint(1,9)*10**random.randint(minexp,maxexp))
			steps -=1			

	return output

def CreateSets(Hyperparms):

	set_new = []
	subsets = []
	names = list(Hyperparms.keys())
	set_old = []
	for i in Hyperparms[names[0]]:
		set_old.append([i])
	del names[0]

	for name in names: 
		#Append each value to each element in list
		for i in Hyperparms[name]:


			for c in set_old:
				x = c+[i]

				set_new.append(x)
		set_old = set_new
		set_new = []
	
	#Adds dataset to the hyperparameter sequence
	for x in set_old:
		x.insert(0,file)

	return set_old


#(name:(min max steps types))
def CreateRandomSet(preparameters):
	
	hyperparameters = dict()
	for i in preparameters:
		hyperparameters[i] = RandomRange(*preparameters[i])

	print(hyperparameters)
	return CreateSets(hyperparameters)


path = os.getcwd()


###Import and scale
file = pandas.read_csv(path+"/StockData/AAPL.csv").loc[:,"Open"]
file = np.asarray(file)#convert to numpy array

#print(RandomRange(0.000001,0.01,5,"log"))



print("ready")


parameters = {
	
	"lr" :[0.000001,0.01,3,"log"],
	"hiddenDimension": [150,250,3,"int"],
	"seq_length": [2,15,3,"int"]

}



hyperparameters = {

	"lr" :[0.00001],
	"hiddenDimension": [2],
	"seq_length": [3],
	"numberLayers":[1]
	}

#searchSpace = CreateRandomSet(parameters)

searchSpace = CreateSets(hyperparameters)
print(len(searchSpace))
#del searchSpace[0]
print(searchSpace)

threads = []

start = time.time()

SearchResults = {}



print(os.cpu_count())





if __name__ == "__main__":

	p = Pool(3,maxtasksperchild = 10)
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

		






