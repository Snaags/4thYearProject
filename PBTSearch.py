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
	print(number)
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
			print(modelscores)
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
	[5,300,"int"],				#"hiddenDimension" [
	[1,150,"int"],				#"seq_length" 
	[1,1,"int"],				#"numberLayers"
	[1,1,"int"],				#predict_distance	
	[100,100,"int"],				#"batch_size"
	[5,5,"int"]				#"num_epochs"
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
	counter = 5
	

	while end == True:
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
			HP = {}
			for c in i["HyperParameters"]:
				if c != "ID":
					HP[c] = i["HyperParameters"][c]

			lineage[i["HyperParameters"]["number"]].append((i["EvalScore"],HP))
		

		##Searches Models for top performers

		for i in ModelsAlive:
			searchSpace.append(Exploit(0.4, Models,i,file,"Gaussian",ModelsAlive[i][1]))
	
		
		counter -= 1
		if counter == 0:
			end = False 
			

for i in lineage:
	print(i,":  ")

	for c in lineage[i]:
		print(c)









print("Total runtime:",time.time()- start)


