import torch
import os
import pandas
from model import LSTMModel
from PBTTrainer import RunModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
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
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.axes as axes 
import multiprocessing
import random
from Utils import CreateRandomSets, Explore
path = os.getcwd()


def Exploit(probability, models,score,file, distribution, number = None):	##probability: percentage of population deemed healthy, 
												##models: list of scores from population, Score: score of currently testing model
												##file: dataset for training, distribution: type of selection from healthy models "Guassian", "Uniform"

	
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

			if distribution == "Uniform":
				x = random.randint(1,num_healthy)

			if distribution == "Gaussian":
				x = math.ceil(abs(random.gauss(0,0.2))*num_healthy)

			modelscores = []
			for i in models:
				modelscores.append(float(i))

			for c in range(x):
				output = modelscores.pop(modelscores.index(min(modelscores)))

			#lineage[number].append(str("Mutate "+str(models[output]["number"])))
			number_new =max(lineage.keys())+1
			lineage[number_new] = [lineage[models[output]["number"]][-1]]
			number = number_new

			print(output)
			
			
			output = Explore(models[output],file,0,number)
			return output

	#return the same model if within the number of healthy models 
	return Explore(models[score],file,0,number)














#########HyperParameter Search Settings###########



SearchType = "Random"


hyperparameters = [
	
	[0.00001,0.001,"log"],	#"lr" 
	[1,30,"int"],				#"hiddenDimension" [
	[1,30,"int"],				#"seq_length" 
	[1,1,"int"],				#"numberLayers"	
	[128,128,"int"],			#"batch_size"
	[1,1,"int"]					#"num_epochs"
					]



###Import and scale
file = pandas.read_csv(path+"/StockData/AAPL.csv").loc[:,"Open"]
file = np.asarray(file)#convert to numpy array

lineage = {}
searchsize = 4
if SearchType == "Random":
	numbers = 0
	searchSpace = CreateRandomSets(file,hyperparameters,searchsize)
	for i in searchSpace:
		i.append(None)
		i.append(numbers)
		lineage[numbers] = []
		numbers +=1



print(len(searchSpace)," iterations of model to run")
start = time.time()
print(os.cpu_count()," CPU logical cores detected")
concurrentTasks = 4
print("Running",concurrentTasks,"models in parrallel")








if __name__ == "__main__":


	"""
	##Intial random search steps
	p = Pool(concurrentTasks,maxtasksperchild = 1)
	results = p.starmap(RunModel,searchSpace)
	p.close()
	p.join()
	"""
	run = True
	counter = 3
	

	while run == True:
		Alive = []
		Models = {}
		ModelsAlive = {}
		with Pool(processes=4) as pool:
			results = pool.starmap(RunModel,searchSpace)
			pool.close()
			pool.join()

		searchSpace = []
		
		
		#Adds the new models to a dictionary of models labled by Evaluation Score
		for i in results:
			Models[i["EvalScore"]] = i["HyperParameters"]

			ModelsAlive[i["EvalScore"]] = [i["HyperParameters"],i["HyperParameters"]["number"]]
			Alive.append(i["HyperParameters"]["number"])
			HP = {}
			for c in i["HyperParameters"]:
				if c != "ID":
					HP[c] = i["HyperParameters"][c]
			lineage[i["HyperParameters"]["number"]].append((i["EvalScore"],HP))

		##Searches Models for top performers

		counter -= 1
		if counter == 0:
			break 

		for i in ModelsAlive:
			searchSpace.append(Exploit(0.4, Models,i,file,"Gaussian",ModelsAlive[i][1]))

	
		

			
best = 9999999999999

for i in lineage:
	print(i,":  ")

	for c in lineage[i]:
		if c[0] < best:
			best = c[0]
			bestnum = i
			besttuple = c

print(best)
OutputHP = []
for i in lineage:
	print(i,":  ")

"""
while True:

	OutputHP.append(lineage[bestnum][lineage[bestnum].index(besttuple)][1])
	print(OutputHP)
	if lineage[bestnum].index([besttuple]) ==0:
		break

	besttuple = lineage[bestnum][lineage[bestnum].index([besttuple])-1]
	bestnum = besttuple[1]["number"]
"""

print(OutputHP)

initpoints = []
finalpoints = []
finalscores = []
lines = []
line = []



for i in lineage:
	if len(line) > 0:
		lines.append(line)
	line = []
	for c in lineage[i]:
		if lineage[i].index(c) == 0 and i < searchsize -1 :
			initpoints.append([c[1]["lr"],c[1]["seq_length"]])

		if lineage[i].index(c) == len(lineage[i])-1 and i in Alive:

			finalpoints.append([c[1]["lr"],c[1]["seq_length"]])
			finalscores.append(c[0])

		line.append([c[1]["lr"],c[1]["seq_length"]])


init = np.array(initpoints)
final = np.array(finalpoints)



fig = plt.figure(figsize = [19.20,10.80])
for i in lines:
	i = np.array(i)
	plt.plot(i[:,0],i[:,1], alpha = 0.2, c = "b",lw = 0.5)
plt.scatter(init[:,0],init[:,1],c= "r",s = 20)
plt.scatter(final[:,0],final[:,1],s = 20,c = finalscores, cmap = 'Blues', alpha = 0.7)
cbar = plt.colorbar()
cbar.set_label('RME')
plt.xscale("log")
plt.ylabel("Window Sequence Length")
plt.xlabel("Learning Rate")
plt.show()

initpoints = []
finalpoints = []
finalscores = []
lines = []
line = []



for i in lineage:
	if len(line) > 0:
		lines.append(line)
	line = []
	for c in lineage[i]:
		if lineage[i].index(c) == 0 and i < searchsize -1 :
			initpoints.append([c[1]["lr"],c[1]["hiddenDimension"]])

		if lineage[i].index(c) == len(lineage[i])-1 and i in Alive:

			finalpoints.append([c[1]["lr"],c[1]["hiddenDimension"]])
			finalscores.append(c[0])

		line.append([c[1]["lr"],c[1]["hiddenDimension"]])


init = np.array(initpoints)
final = np.array(finalpoints)

fig = plt.figure(figsize = [19.20,10.80])
for i in lines:
	i = np.array(i)
	plt.plot(i[:,0],i[:,1], alpha = 0.2, c = "b",lw = 0.5)
plt.scatter(init[:,0],init[:,1],c= "r",s = 20)
plt.scatter(final[:,0],final[:,1],s = 20,c = finalscores, cmap = 'Blues', alpha = 0.7)
cbar = plt.colorbar()
cbar.set_label('RME')
plt.ylabel("Hidden Layer Size")
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.show()