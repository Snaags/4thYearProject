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

def MatchDate(data1,data2):
	output = []
	outputb = []
	hold2 = 0
	datelist = []
	for c in data2:
		flag = False
		for i in data1:
			if i[0] == c[0]:
				datelist.append([i[0],c[1]])
				flag = True
				break

		if flag == False:
			hold = list(c[0])
			if "".join(hold[8:]) == "31" or "".join(hold[8:]) == "30":
				if "".join(hold[5:7]) == "12":
					hold[3] = str(int(hold[3]) + 1)
					hold[5:7] = "01"
				hold[8:] = "02"

			hold[9] = str(int(hold[9]) + 1)	
			for i in data1:
				if i[0] == "".join(hold):
					datelist.append([i[0],c[1]])
					flag = True

		if flag == False:
			hold = list(c[0])
			if "".join(hold[8:]) == "31" or "".join(hold[8:]) == "30":
				if "".join(hold[5:7]) == "12":
					hold[3] = str(int(hold[3]) + 1)
					hold[5:7] = "01"
				hold[8:] = "03"

			hold[9] = str(int(hold[9]) + 2)	
			for i in data1:
				if i[0] == "".join(hold):
					datelist.append([i[0],c[1]])
					flag = True

	for i in data1:
		for c in datelist:
			if i[0] == c[0]:


				hold2 = c[1]
				if type(hold2) == str:
					hold2 = hold2.strip("$")
					hold2 = float(hold2)
		output.append(hold2)
	output = np.asarray(output)
	return output

def Explore(hyperparameters,file,Mutation = 0.4, number = None):
	mutation = Mutation
	output = [file,0,0,0,0,0,0,0,hyperparameters["ID"],number]
	for i in hyperparameters:


		if i == "lr":

			if mutation != 0:
				x = hyperparameters[i]*((mutation*10)**random.uniform(-1,1))
			
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
				x = hyperparameters[i]*((mutation*10)**random.uniform(-1,1))
			
			else:
				x = hyperparameters[i]
			output[6] = x

		elif i == "num_epochs":
			output[7] = math.ceil(hyperparameters[i])

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
		#Find variables being searched
		if (i[1] - i[0]) != 0:
			indexs.append(Hyperparms.index(i))
		#Constant variables
		else:
			sets[Hyperparms.index(i)].append(i[0])

	##Find the number of steps based on the search size and the number of variables being searched
	steps = math.ceil(SearchSize**(1/len(indexs)))

	##Step through the range of values based on the scale given 
	for i in indexs:
		
		##Integer values
		if Hyperparms[i][2] == "int":
			increments = int((Hyperparms[i][1] - Hyperparms[i][0])/(steps -1))
			sets[i].append([Hyperparms[i][0] + increments* c for c in range(steps)])
		
		##log scale
		elif Hyperparms[i][2] == "log":
			increments = (math.log(Hyperparms[i][1]) -math.log(Hyperparms[i][0]))/(steps - 1)
			sets[i].append(np.exp([math.log(Hyperparms[i][0]) + increments * c for c in range(steps)]))
		#Power of two
		elif Hyperparms[i][2] == "Po2":
			increments = int((Hyperparms[i][1] - Hyperparms[i][0])/(steps -1))
			sets[i].append([Hyperparms[i][0] + increments* c for c in range(steps)])
		#Catch
		else:
			increments = (Hyperparms[i][1] - Hyperparms[i][0])/(steps -1)
			sets[i].append([Hyperparms[i][0] + increments* c for c in range(steps)])


	counter = 0
	count = 1

	###Sorting the values into the different possible combinations 
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
