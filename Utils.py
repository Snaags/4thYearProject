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




	#Find high proforming networks
	#Select network based on a probability distrobution
	#Extract weights and hyperparameters
	#Apply to the targeted network

def Explore(hyperparameters,file,Mutation = 0.4, number = None):
	mutation = Mutation
	output = [file,0,0,0,0,0,0,0,hyperparameters["ID"],number]
	for i in hyperparameters:


		if i == "lr":

			if mutation != 0:
				x = hyperparameters[i] + hyperparameters[i]*((mutation*10)**random.uniform(-1,1))
			
			else:
				x = hyperparameters[i]
			output[1] = x

		elif i == "dropout":
			x = hyperparameters[i] + hyperparameters[i]*random.uniform(-1,1)*mutation

		elif i == "hiddenDimension":

			#if random.uniform(mutation-1,mutation) > 0:
			#	x = int(hyperparameters[i])+math.ceil(random.uniform(0,1*(mutation))*hyperparameters[i])
			#else:
			x = hyperparameters[i]
			output[2] = x
		elif i == "seq_length":
			if hyperparameters[i] == 1:
				x = int(hyperparameters[i]+random.randint(0,1))
			else:
				x = int(hyperparameters[i]+math.ceil(random.uniform(-1,1)*mutation*hyperparameters[i]))
			output[3] = x

		elif i == "numberLayers":
			x = int(hyperparameters[i])#+math.ceil(random.uniform((1*mutation-1),1*mutation)*hyperparameters[i]))
			output[4] = x

		elif i == "batch_size":
			
			if hyperparameters[i] != 1 and random.uniform(mutation-1,mutation)>0:
				x = math.ceil(2**(math.log2(hyperparameters[i])+(random.choice([-1,1]))))

			else:
				x = hyperparameters[i]
			output[5] = x

		if i == "l2":

			if mutation != 0:
				x = hyperparameters[i] + hyperparameters[i]*((mutation*10)**random.uniform(-1,1))
			
			else:
				x = hyperparameters[i]
			output[6] = x

		elif i == "num_epochs":
			output[7] = hyperparameters[i]

	return output



##Creates a range of values between 2 boundries. Types = log, int 	for scales of number distrobution
def RandomRange(min, max,types):
	output = []

	if types == "int":
		
		output = int(random.randint(min,max))
	

	if types == "log":
		minexp = np.log10(np.abs(min))
		maxexp = np.log10(np.abs(max))

		output = random.randint(1,9)*10**random.randint(minexp,maxexp)

	if types == "Po2":
		minpow = math.log2(min)
		maxpow = math.log2(max)
		output = 2**random.randint(minpow,maxpow)

		
	return output

"""
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
"""

def CreateSets(file,Hyperparms,SearchSize):
	output = [ [] for i in range(SearchSize)]

	print(len(output))
	sets = [ [] for i in range(len(Hyperparms))]
	print(sets)
	indexs = []
	for i in Hyperparms:

		if (i[1] - i[0]) != 0:
			indexs.append(Hyperparms.index(i))
		else:
			sets[Hyperparms.index(i)].append(i[0])

	steps = math.ceil(SearchSize**(1/len(indexs)))
	print(steps)
	for i in indexs:
		##Integer values
		if Hyperparms[i][2] == "int":
			increments = int((Hyperparms[i][1] - Hyperparms[i][0])/(steps -1))
			sets[i].append([Hyperparms[i][0] + increments* c for c in range(steps)])
		##log scale
		elif Hyperparms[i][2] == "log":

			increments = (math.log(Hyperparms[i][1]) -math.log(Hyperparms[i][0]))/(steps - 1)

			sets[i].append(np.exp([math.log(Hyperparms[i][0]) + increments * c for c in range(steps)]))
		elif Hyperparms[i][2] == "Po2":
			increments = int((Hyperparms[i][1] - Hyperparms[i][0])/(steps -1))
			sets[i].append([Hyperparms[i][0] + increments* c for c in range(steps)])

		else:
			increments = (Hyperparms[i][1] - Hyperparms[i][0])/(steps -1)
			sets[i].append([Hyperparms[i][0] + increments* c for c in range(steps)])


	counter = 0
	count = 1

	for i in sets:
		a = []
			
		if type(i[0]) is int:
			hold = [i[0]]*SearchSize

		else:
			for c in i[0]:
				for h in range(count):

					a.append(c)
			hold = [a]*int((SearchSize/len(i)))
			y = []
			for i in hold:
				y +=i
			hold = y


			count = count*steps

		for c,z in zip(hold,range(SearchSize)):
			output[z].append(c)
		counter +=1
	for x in output:
		print(x)

	for i in output:
		i.insert(0, file)
	return output

"""
[1,2]
[10,15]
[2,4]

1 10 2
2 10 2
1 15 2
2 15 2 
1 10 4
2 10 4
1 15 4
2 15 4
"""

#(name:(min max steps types))
def CreateRandomSets(file,preparameters,size):
	
	setin = list()
	setout = list()
	#Loop through the keys in the input dict, passing the values as arguements to the function for each iteration
	for c in range(size):
		for i in preparameters:
			setin.append(RandomRange(*i))
		setin.insert(0,file)
		setout.append(setin)
		setin = []
	return setout
