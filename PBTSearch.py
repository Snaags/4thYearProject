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

def RSI(file,N):
	output = []
	U = []
	D = []
	RSI = 0
	i_old = 0
	

	for i in file:
		v = i - i_old

		if v > 0:
			U.append(v)
			D.append(0)
		else:
			D.append(-v)
			U.append(0)

		i_old = i

		if len(U) > 14:
			U.pop(0)
			D.pop(0)
			RS = SMMA(U)/SMMA(D)
			RSI = 100 - (100/(1+RS))
		
		output.append(RSI)

	return output 


def SMMA(data):

	N = len(data)
	MMA = 0

	def step(MMA, new_sample, N):

		return ((N - 1)*MMA + new_sample)/N

	
	for i in data:
		MMA = step(MMA,i,N)

	return MMA


def Exploit(probability, models,score,file, distribution, number = None):	##probability: percentage of population deemed healthy, 
	##models: list of scores from population, Score: score of currently testing model
	##file: dataset for training, distribution: type of selection from healthy models "Guassian", "Uniform"

	
	score = float(score)

	num_healthy = searchsize*probability

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
			
			
			output = Explore(models[output],file,0.2,number)
			return output

	#return the same model if within the number of healthy models 
	return Explore(models[score],file,0,number)


#########HyperParameter Search Settings###########



SearchType = "Random"
searchsize = 20
cores = 4
mutations = 6
hyperparameters = [
	
	[0.00001,0.001,"log"],	#"lr" 
	[30,80,"int"],				#"hiddenDimension" [
	[4,20,"int"],				#"seq_length" 
	[1,1,"int"],				#"numberLayers"	
	[32,64,"Po2"],				#"batch_size"
	[0.0000001,0.0001,"log"]	,	#Regularization
	[20,20,"int"]					#"num_epochs"
					]



###Import and scale
APPLC = pandas.read_csv(path+"/StockData/AAPL.csv").loc[:,"Close"]
APPLC = np.asarray(APPLC)#convert to numpy array

APPLV = pandas.read_csv(path+"/StockData/AAPL.csv").loc[:,"Volume"]
APPLV = np.asarray(APPLV)#convert to numpy array

MSFTC = pandas.read_csv(path+"/StockData/MSFT.csv").loc[:,"Close"]
MSFTC = np.asarray(MSFTC)#convert to numpy array

GOOGLC = pandas.read_csv(path+"/StockData/GOOGL.csv").loc[:,"Close"]
GOOGLC = np.asarray(GOOGLC)#convert to numpy array

APPLRSI = np.asarray(RSI(APPLC, 14))


file = np.stack((APPLC,APPLV,APPLRSI,MSFTC,GOOGLC),1)


lineage = {}

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
print("Running",cores,"models in parrallel")








if __name__ == "__main__":


	"""
	##Intial random search steps
	p = Pool(concurrentTasks,maxtasksperchild = 1)
	results = p.starmap(RunModel,searchSpace)
	p.close()
	p.join()
	"""
	run = True
	counter = mutations
	
	
	while run == True:
		Alive = []
		ModelsAlive = {}
		Models = {}
		with Pool(processes=cores) as pool:
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
			searchSpace.append(Exploit(0.3, Models,i,file,"Gaussian",ModelsAlive[i][1]))

	
		

			
best = 9999999999999
file = open("SearchLog.txt","w")
file.write("Total run time: "+str(time.time() - start)+"\n")
file.close()
file = open("SearchLog.txt","a")
file.write("lr hidden seq_length \n")


file.write("Hyperparameter Ranges:\n")
for i in hyperparameters:
	file.write(str(i)+"\n")
file.write("Search of size: "+str(searchsize)+"\n")
for i in lineage:
	print(i,":  ")
	file.write(str(i)+":  \n")
	for c in lineage[i]:
		print(c)
		file.write(str(c)+"\n")
		if c[0] < best:
			best = c[0]
			bestnum = i
			besttuple = c
file.write("Lowest error model: \n")
file.write(str(besttuple)+"\n\n")
print(best)


OutputHP = []


while True:
	print(lineage[bestnum])
	OutputHP.append(lineage[bestnum][lineage[bestnum].index(besttuple)])
	print(OutputHP)
	if lineage[bestnum].index(besttuple) ==0:
		break

	besttuple = lineage[bestnum][lineage[bestnum].index(besttuple)-1]
	bestnum = besttuple[1]["number"]


file.write("Hyperparameter schedule for top performing model: \n")

for i in OutputHP:
	print(i)

	file.write(str(i)+"\n")






##Learning Rate /Sequence length


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

stepcounter = float(1/len(lines))
greencounter = 0
fig = plt.figure(figsize = [19.20,10.80])
for i in lines:
	i = np.array(i)
	plt.plot(i[:,0],i[:,1], alpha = 0.2, c = (0,greencounter,1),lw = 0.5)
	greencounter += stepcounter
if mutations != 1:
	plt.scatter(init[:,0],init[:,1],c= "r",s = 20)

plt.scatter(final[:,0],final[:,1],s = 15,c = finalscores,vmin = 0, vmax = 800, cmap = 'summer', alpha = 0.7)
cbar = plt.colorbar()
cbar.set_label('RME')
plt.xscale("log")
plt.ylabel("Window Sequence Length")
plt.xlabel("Learning Rate")
plt.savefig(("Graphs/"+str(float(best))+": lr Seq.pdf"),dpi=1200)
plt.clf()




##Learning Rate /hidden Dimension
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
stepcounter = float(1/len(lines))
greencounter = 0
for i in lines:
	i = np.array(i)
	plt.plot(i[:,0],i[:,1], alpha = 0.2, c = (0,greencounter,1),lw = 0.5)
	greencounter += stepcounter
if mutations != 1:
	plt.scatter(init[:,0],init[:,1],c= "r",s = 20)

plt.scatter(final[:,0],final[:,1],s = 20,vmin = 0, vmax = 800,c = finalscores, cmap = 'summer', alpha = 0.7)
cbar = plt.colorbar()
cbar.set_label('RME')
plt.ylabel("hiddenDimension")
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.savefig(("Graphs/"+str(float(best))+": lr hiddenDimension.pdf"),dpi=1200)
plt.clf()



##Learning Rate / Regularisation

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
			initpoints.append([c[1]["lr"],c[1]["l2"]])

		if lineage[i].index(c) == len(lineage[i])-1 and i in Alive:

			finalpoints.append([c[1]["lr"],c[1]["l2"]])
			finalscores.append(c[0])

		line.append([c[1]["lr"],c[1]["l2"]])


init = np.array(initpoints)
final = np.array(finalpoints)

fig = plt.figure(figsize = [19.20,10.80])
stepcounter = float(1/len(lines))
greencounter = 0
for i in lines:
	i = np.array(i)
	plt.plot(i[:,0],i[:,1], alpha = 0.2, c = (0,greencounter,1),lw = 0.5)
	greencounter += stepcounter
if mutations != 1:
	plt.scatter(init[:,0],init[:,1],c= "r",s = 20)

plt.scatter(final[:,0],final[:,1],s = 20,vmin = 0, vmax = 800,c = finalscores, cmap = 'summer', alpha = 0.7)
cbar = plt.colorbar()
cbar.set_label('RME')
plt.ylabel("l2 Weighting")
plt.yscale("log")
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.savefig(("Graphs/"+str(float(best))+": lr l2.pdf"),dpi=1200)
plt.clf()


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
			initpoints.append([c[1]["lr"],c[1]["batch_size"]])

		if lineage[i].index(c) == len(lineage[i])-1 and i in Alive:

			finalpoints.append([c[1]["lr"],c[1]["batch_size"]])
			finalscores.append(c[0])

		line.append([c[1]["lr"],c[1]["batch_size"]])


init = np.array(initpoints)
final = np.array(finalpoints)

fig = plt.figure(figsize = [19.20,10.80])
stepcounter = float(1/len(lines))
greencounter = 0
for i in lines:
	i = np.array(i)
	plt.plot(i[:,0],i[:,1], alpha = 0.2, c = (0,greencounter,1),lw = 0.5)
	greencounter += stepcounter
if mutations != 1:
	plt.scatter(init[:,0],init[:,1],c= "r",s = 20)

plt.scatter(final[:,0],final[:,1],s = 20,vmin = 0, vmax = 800,c = finalscores, cmap = 'summer', alpha = 0.7)
cbar = plt.colorbar()
cbar.set_label('RME')
plt.ylabel("batch Size")
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.savefig(("Graphs/"+str(float(best))+": lr batch_size.pdf"),dpi=1200)
plt.clf()