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
			lineage[number].append(str("Mutate "+str(models[output]["number"])))
			print(output)
			
			
			output = Explore(models[output],file,0.1,number)
			return output

	#return the same model if within the number of healthy models 
	return Explore(models[score],file,0,number)














#########HyperParameter Search Settings###########



SearchType = "Random"


hyperparameters = [
	
	[0.00000001,0.001,"log"],	#"lr" 
	[60,150,"int"],				#"hiddenDimension" [
	[10,40,"int"],				#"seq_length" 
	[1,1,"int"],				#"numberLayers"	
	[200,200,"int"],			#"batch_size"
	[100,100,"int"]					#"num_epochs"
					]



###Import and scale
file = pandas.read_csv(path+"/StockData/AAPL.csv").loc[:,"Open"]
file = np.asarray(file)#convert to numpy array

lineage = {}
if SearchType == "Random":
	numbers = 0
	searchSpace = CreateRandomSets(file,hyperparameters,8)
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
	end = True
	counter = 3
	

	while end == True:

		Models = {}
		ModelsAlive = {}
		with Pool(processes=8) as pool:
			results = pool.starmap(RunModel,searchSpace)
			pool.close()
			pool.join()

		searchSpace = []
		
		
		#Adds the new models to a dictionary of models labled by Evaluation Score
		for i in results:
			Models[i["EvalScore"]] = i["HyperParameters"]

			ModelsAlive[i["EvalScore"]] = [i["HyperParameters"],i["HyperParameters"]["number"]]
		
			HP = {}
			for c in i["HyperParameters"]:
				if c != "ID":
					HP[c] = i["HyperParameters"][c]

			lineage[i["HyperParameters"]["number"]].append((i["EvalScore"],HP))

		##Searches Models for top performers

		counter -= 1
		if counter == 0:
			end = False 

		for i in ModelsAlive:
			searchSpace.append(Exploit(0.4, Models,i,file,"Gaussian",ModelsAlive[i][1]))
	
		

			


for i in lineage:
	print(i,":  ")

	for c in lineage[i]:


		print(c)

points = []
points1 = []
points2 = []
points3 = []
scores = []
for i in lineage:
	for c in lineage[i]:
		if c[0:6] != "Mutate":
			points.append([c[1]["lr"],c[1]["hiddenDimension"],c[1]["seq_length"]])
			scores.append(c[0])
			points1.append([c[1]["lr"],c[1]["hiddenDimension"]])
			points2.append([c[1]["lr"],c[1]["seq_length"]])
			points3.append([c[1]["hiddenDimension"],c[1]["seq_length"]])

X = np.array(points)
#print(X)
X = TSNE().fit_transform(X)
#scores = np.asarray(scores)
#scaler = MinMaxScaler(feature_range=(0, 10))	#scale data
#scores = scaler.fit_transform(scores.reshape(1, -1))
#print(scores)

points1 = np.array(points1)
points2 = np.array(points2)
points3 = np.array(points3)

#for i in scores:
#	score = i
#	for c in i:
#		print(c)
plt.scatter(X[:,0],X[:,1],s = 20,c = scores,vmin = 0, vmax = 3000 , cmap = 'plasma', alpha = 0.3)
cbar = plt.colorbar()
cbar.set_label('RME')
plt.show()


plt.scatter(points1[:,0],points1[:,1],s = 20,c = scores,vmin = 0, vmax = 3000 , cmap = 'plasma', alpha = 0.3)
cbar = plt.colorbar()
cbar.set_label('RME')
#axes.set_xlim(min(points1[:,1]),max(points1[:,1]))
#axes.set_ylable("Hidden Layer Size")
#axes.set_xlable("learning Rate")
plt.show()
plt.scatter(points2[:,0],points2[:,1],s = 20,c = scores,vmin = 0, vmax = 3000 , cmap = 'plasma', alpha = 0.3)
cbar = plt.colorbar()
cbar.set_label('RME')
#axes.set_xlim(min(points1[:,1]),max(points1[:,1]))

plt.show()
plt.scatter(points3[:,0],points3[:,1],s = 20,c = scores,vmin = 0, vmax = 3000 , cmap = 'plasma', alpha = 0.3)
cbar = plt.colorbar()
cbar.set_label('RME')
plt.show()
print("Total runtime:",time.time()- start)


