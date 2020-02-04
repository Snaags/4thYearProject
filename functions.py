import torch
import os
import pandas
from model import LSTMModel
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from torch.nn.utils.rnn import pack_sequence
import math
import threading
path = os.getcwd()
#torch.cuda.set_device(0)

############TO DO##############
"""





make error measurement a percentage

refactor code into a function

set up batching

set up allowing multiple features






"""
###############################











def Batch(X, batchSize):
	X = list(X)
	while len(X)%batchSize != 0:
		X.pop()

	output = torch.cuda.FloatTensor(X)
	output = output.view(-1,10,batchSize,1)

	return output


def RunModel(X,lr ,hiddenDimension,seq_length=10,numberLayers = 1,predict_distance = 1):

	## transforms time series into samples of length "seq_length" each paired with a lable for trainging "predict_distance" after the last element in the sample 
	def GroupData(X,seq_length,predict_distance):

		x_out = []
		y_out = []

		for i in range(len(X)-(seq_length + predict_distance)):
			hold = X[i:(i+seq_length)]
			sample = tuple(hold)
			lable = tuple(X[i+(seq_length + predict_distance)])
			x_out.append(sample)
			y_out.append(lable)

		return x_out, y_out

	def test_train_split(file,train_size):

		X_train = file[:int(len(file)*train_size)]
		X_test = file[int(len(file)*train_size):]

		return X_train, X_test

	scaler = MinMaxScaler(feature_range=(-1, 1))	#scale data
	X = scaler.fit_transform(X.reshape(-1, 1))

	X_train, X_test = test_train_split(X,0.80) #Training data from 80% of the total data set


	X, y = GroupData(X_train,seq_length,predict_distance)

	##Convert samples and lables
	samples = torch.cuda.FloatTensor(X)
	lables = torch.cuda.FloatTensor(y)



	model = LSTMModel(input_dim = 1, hidden_dim = hiddenDimension,seq= seq_length, output_dim=1, layer_dim=numberLayers)
	model.cuda()


	loss_fn = torch.nn.MSELoss(reduction = "none")

	

	
	optimiser = torch.optim.Adam(model.parameters(), lr=lr)



	startTime = time.time()
	num_epochs = 3
	model.init_hidden()
	for t in range(num_epochs):

		#for param in model.parameters():
		#	print(param.size())
		print(model.state_dict())

		# Clear stored gradient
		model.zero_grad()
		for X,y in zip(samples,lables):

			y_pred = model(X)
			loss = loss_fn(y_pred, y)


			# Backward pass
			loss.backward()

			# Update parameters
			optimiser.step()

			# Zero out gradient, else they will accumulate between epochs
			optimiser.zero_grad()

		print("Epoch",t,"completed in:",time.time()-startTime,"seconds")

	X, y = GroupData(X_test,seq_length,predict_distance)

	##Convert samples and lables
	samples = torch.cuda.FloatTensor(X)
	lables = torch.cuda.FloatTensor(y)
	loss_fn = torch.nn.MSELoss(reduction = "none")
	test_lost_score = 0
	model.init_hidden()
	results = []
	for X,y in zip(samples,lables):

		y_pred = model(X)
		results.append(y_pred)

	##test_lost_score =  list(test_lost_score.cpu())

	lables = lables.cpu()

	results = np.asarray(results)
	results = scaler.inverse_transform(results.reshape(-1,1))

	test_lost_score = np.asarray(test_lost_score)
	test_lost_score = scaler.inverse_transform(test_lost_score.reshape(-1,1))

	lables = np.asarray(lables)
	lables = scaler.inverse_transform(lables.reshape(-1,1))


	results = torch.cuda.FloatTensor(results)
	lables = torch.cuda.FloatTensor(lables)

	##Mean absolute percentage error (MAPE)
	total_error = 0
	for i,c in zip(results,lables):
		test_lost_score = abs(c - i)
		test_lost_score = test_lost_score/c
		total_error += test_lost_score

	total_error = total_error/ len(lables)

	error = total_error*100

	print("Total training and validation time: ",time.time() - startTime)






	plt.plot(results.cpu(), label="Preds")
	plt.plot(lables.cpu(), label=("Data days ahead"))
	plt.legend()
	plt.savefig(("Graphs/"+str(float(error))+".png"))
	#MAPE
	print("lr:",lr ," ;hiddenDimension:",hiddenDimension," ;numberLayers:",numberLayers," ;seq_length:",seq_length, " -- MAPE:",float(error),"%")
	
	#RMSE
	#print("lr:",lr ," ;hiddenDimension:",hiddenDimension," ;numberLayers:",numberLayers," ;seq_length:",seq_length, " -- RMSE:",float(error))
	
	return float(error)








##Inputs
"""
lr
hidden dim
number layers
sequence length
"""

#Size of search
#range of values


