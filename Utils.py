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

def Exploit(probability, models,score,file):	##Selection offset
	
	score = float(score)

	num_healthy = int(len(models)*probability)

	#Sets healthy set to 1 when calculation produces a number smaller than this
	if num_healthy < 1:
		num_healthy = 1


	modelrank = int(num_healthy)
	
	#loops through models to find the score passed to functions placement.
	for i in models:
		if score > float(i):
			modelrank -= 1
		

		#Finds replacement model if model is deemed unfit.
		if modelrank <= 0:
			#Randomly select healthy model to exploit.
			x = random.randint(1,num_healthy)

			modelscores = []
			for i in models:
				modelscores.append(float(i))
			print(modelscores)
			for c in range(x):
				output = modelscores.pop(modelscores.index(min(modelscores)))
			print(output)
			
			
			output = Explore(models[output],file)
			return output

	#return the same model if within the number of healthy models 
	return Explore(models[score],file, 0)

	#Find high proforming networks
	#Select network based on a probability distrobution
	#Extract weights and hyperparameters
	#Apply to the targeted network
	pass

def Explore(hyperparameters,file,mutation = 0.05):
	mutablesf = ["lr"]
	mutablesi =["seq_length"]
	mutablesh = ["hiddenDimension","numberLayers"]
	output = [file]
	for i in hyperparameters:
		if i != "ID":
			print(i,"pre mutation: ", hyperparameters[i])
		if i == "lr" or i == "dropout":
			x = hyperparameters[i] + hyperparameters[i]*random.uniform(-1,1)*mutation
		elif i == "seq_length":
			if hyperparameters[i] == 1:
				x = int(hyperparameters[i]+random.randint(0,1))
			else:
				x = int(hyperparameters[i]+math.ceil(random.randint(-1,1)*mutation*hyperparameters[i]))

		elif i == "hiddenDimension": #or i == "numberLayers":
			x = int(hyperparameters[i]+math.ceil(random.uniform(0,1)*(mutation)*hyperparameters[i]))
		else:
			x = hyperparameters[i]
		if i != "ID":
			print(i," post mutation: ",x)
		output.append(x)
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
