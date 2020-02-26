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

def Explore(hyperparameters,file,mutation = 0.05, number = None):
	output = [file,0,0,0,0,0,0,hyperparameters["ID"],number]
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

		elif i == "num_epochs":
			output[6] = hyperparameters[i]

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

def CreateSets(file,Hyperparms):

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
