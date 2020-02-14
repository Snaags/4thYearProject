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


##Creates a range of values between 2 boundries. Types = log, int 	for scales of number distrobution
def RandomRange(min, max,types):
	output = []

	if types == "int":
		
		output = random.randint(min,max)
	

	if types == "log":
		minexp = np.log10(np.abs(min))
		maxexp = np.log10(np.abs(max))


		output = random.randint(1,9)*10**random.randint(minexp,maxexp)

	return int(output)

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
	
	hyperparameters = dict()
	setin = list()
	setout = list()
	#Loop through the keys in the input dict, passing the values as arguements to the function for each iteration
	for c in range(size):
		for i in preparameters:
			setin.append(RandomRange(*preparameters[i]))
		setin.insert(0,file)
		setout.append(setin)
		setin = []
	return setout
