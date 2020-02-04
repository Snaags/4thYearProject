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







def test_train_split(file,train_size):

	X_train = file[:int(len(file)*train_size)]
	X_test = file[int(len(file)*train_size):]

	return X_train, X_test




def Batch(X, batchSize):
	X = list(X)
	while len(X)%batchSize != 0:
		X.pop()

	output = torch.cuda.FloatTensor(X)
	output = output.view(-1,10,batchSize,1)

	return output






###Import and scale
file = pandas.read_csv(path+"/StockData/AAPL.csv").loc[:,"Open"]
file = np.asarray(file)#convert to numpy array

scaler = MinMaxScaler(feature_range=(-1, 1))	#scale data
file = scaler.fit_transform(file.reshape(-1, 1))


X_train, X_test = test_train_split(file,0.80) #Training data from 80% of the total data set






def RunModel(X,lr, costFunction,hiddenDimension,numberLayers,Optimization,seq_length,predict_distance):

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



	X_train, y_train = GroupData(X_train,10,1)
	X_train = torch.cuda.FloatTensor(X_train)
	y_train = torch.cuda.FloatTensor(y_train)
	#print(X_train.size())

	X_test, y_test = GroupData(X_test,10,1)

	X_test = torch.cuda.FloatTensor(X_test)
	y_test = torch.cuda.FloatTensor(y_test) 

	model = LSTMModel(input_dim = 1, hidden_dim =100, output_dim=1, layer_dim=1)
	model.cuda()


	loss_fn = torch.nn.MSELoss(reduction = "none")




	optimiser = torch.optim.ASGD(model.parameters(), lr=0.001)






















	return score



##Inputs

#Hyper-Parameters 
"""

seq_length - length of input seqeunce given for training interval

Optimization algorithm

cost function

hidden dimentions

number of layers

batching * 

learning rate

epochs *

features

"""



x_real = X_test

###Training data
#X = Batch(X_train,10)
#print(X)






print(model.parameters())

#####################
# Train model
#####################

startTime = time.time()

num_epochs = 15
hist = np.zeros(num_epochs)

Predictions = []
trainingout = []
test_input = []
loss = 0
count = 0
model.init_hidden()
batchloop = 0
for t in range(num_epochs):
	# Clear stored gradient
	model.zero_grad()

	# Initialise hidden state
	# Don't do this if you want your LSTM to be stateful
	

	# Forward pass
	for X,y in zip(X_train,y_train):

		#trainingout.append(y)
		y_pred = model(X)
		#Predictions.append(y_pred)
		loss = loss_fn(y_pred, y)



		# Backward pass
		loss.backward()

		# Update parameters
		optimiser.step()
		# Zero out gradient, else they will accumulate between epochs
		optimiser.zero_grad()

	print("Training time:", int(time.time() - startTime))


	if t % 5 == 0:
		print("Epoch ", t, "MSE: ", loss.item() )

	hist[t] = loss.item()




"""
Predictions = np.asarray(Predictions)
Predictions = scaler.inverse_transform(Predictions.reshape(-1,1))

trainingout = np.asarray(trainingout)
trainingout = scaler.inverse_transform(trainingout.reshape(-1,1))


plt.plot(Predictions, label="Preds")
plt.plot(trainingout, label="Data")
plt.legend()
plt.show()
"""
print("Training time:", int(time.time() - startTime),"Seconds")
plt.plot(hist, label="Training loss")
plt.legend()
plt.show()


results = []
realy = y_test.cpu()
realx = X_test.cpu()
count = 0
test_lost_score = 0
model.init_hidden()
test_input = []
for X,y in zip(X_test,y_test):

	y_pred = model(X)
	results.append(y_pred)
	test_lost_score += loss_fn(y_pred, y)	


results = np.asarray(results)
results = scaler.inverse_transform(results.reshape(-1,1))

y_test = np.asarray(realy)
y_test = scaler.inverse_transform(realy.reshape(-1,1))
x_real = np.asarray(x_real)
x_real = scaler.inverse_transform(x_real.reshape(-1,1))


error = test_lost_score*(1/len(y_test))

print("Total Error:",float(error))
x_real = x_real[(10-1):]
plt.plot(results, label="Preds")
plt.plot(y_test, label=("Data days ahead"))
plt.plot(x_real, label="Input data")
plt.legend()
plt.show()




"""

predictdata = file[len(file)-20:]
print(len(predictdata))
realPredictions = []

predictdata, y = GroupData(predictdata,10,1)

X = torch.cuda.FloatTensor(predictdata)
print(X)

for x in X:
	y_pred = model(X)


for i in range(10):

	y_pred = model(X)
	X = list(X)
	X.pop(0)
	X.append(y_pred)
	print(X)
	X = torch.cuda.FloatTensor(X)

	realPredictions.append(y_pred)



predictdata = np.asarray(predictdata)
predictdata = scaler.inverse_transform(predictdata.reshape(-1,1))
realPredictions = np.asarray(realPredictions)
realPredictions = scaler.inverse_transform(realPredictions.reshape(-1,1))

realPredictions = np.append(predictdata,realPredictions)


plt.plot(realPredictions, label="Preds")
plt.plot(predictdata, label=("Data days ahead"))
#plt.plot(x_real, label="Input data")
plt.legend()
plt.show()

"""