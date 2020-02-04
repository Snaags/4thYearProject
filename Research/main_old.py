import torch
import os
import pandas
from model import LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

path = os.getcwd()
file = pandas.read_csv(path+"/StockData/AAPL.csv").loc[:,"Open"]
file = np.asarray(file)#convert to numpy array

scaler = MinMaxScaler(feature_range=(-1, 1))	#scale data
file = scaler.fit_transform(file.reshape(-1, 1))

#Sort data into batches

batch_size = 10

X_train = torch.FloatTensor(file[:int(len(file)*0.8)]) #Training data from 80% of the total data set
y_train = torch.FloatTensor(file[10:int((len(file)*0.8)+10)]) #Predict 5 days ahead

X_test = torch.FloatTensor(file[int(len(file)*0.8):-5]) #Testing data from 20% of the total data set
y_test = file[int((len(file)*0.8)+5):] #Predict 5 days ahead

"""
train = []
for i in file:
	train.append(i)
	if len(train)%30 == 0:
"""



X_train = torch.utils.data.DataLoader(X_train, batch_size= batch_size+1, shuffle = False)
y_train = torch.utils.data.DataLoader(y_train, batch_size= batch_size+1, shuffle = False)
#X_test = torch.utils.data.DataLoader(X_test, batch_size= batch_size+1, shuffle = False)

for data in X_train:
	X = data[:-1]
	y = data[-1]

	print(X)
	print(y)

	#print("y:"+y)

"""
batches = []
dataset = []

for i in file:
	batches.append(i)

	if len(batches) == 100:
		dataset.append(batches)
		batches = []
"""

#run each point through the LSTM
#test at the end of each batch
#repeat from 

#print(len(file))




#X_train = torch.FloatTensor(list(X_train))
#y_train = torch.FloatTensor(list(y_train))






model = LSTM(input_dim = 1, hidden_dim =2, batch_size=1, output_dim=1, num_layers=2)



loss_fn = torch.nn.MSELoss(size_average=False)



optimiser = torch.optim.Adam(model.parameters(), lr=0.05)

#####################
# Train model
#####################



num_epochs = 1
hist = np.zeros(num_epochs)

Predictions = []
trainingout = []

for t in range(num_epochs):
	# Clear stored gradient
	model.zero_grad()
	model.hidden = model.init_hidden()
	# Initialise hidden state
	# Don't do this if you want your LSTM to be stateful
	

	# Forward pass
	for X,y in zip(X_train,y_train):
		X = X[:-1]
		y = y[-1]
		trainingout.append(y)
		y_pred = model(X)

		Predictions.append(y_pred)
		loss = loss_fn(y_pred, y)


		# Zero out gradient, else they will accumulate between epochs
		optimiser.zero_grad()

		# Backward pass
		loss.backward()

		# Update parameters
		optimiser.step()

	if t % 5 == 0:
		print("Epoch ", t, "MSE: ", loss.item())
	hist[t] = loss.item()

	# Zero out gradient, else they will accumulate between epochs
	#optimiser.zero_grad()

	# Backward pass
	#loss.backward()

	# Update parameters
	#optimiser.step()
Predictions = np.asarray(Predictions)
Predictions = scaler.inverse_transform(Predictions.reshape(-1,1))

trainingout = np.asarray(trainingout)
trainingout = scaler.inverse_transform(trainingout.reshape(-1,1))


plt.plot(Predictions, label="Preds")
plt.plot(trainingout, label="Data")
plt.legend()
plt.show()

plt.plot(hist, label="Training loss")
plt.legend()
plt.show()

results = []
real = []
count = 0
model.hidden = model.init_hidden()
test_input = []
for data in X_test:
	if len(test_input) < 30:
		test_input.append(data)
		X = torch.FloatTensor(test_input)
	else:

		y_pred = model(X)
		results.append(y_pred)
		new = list(X)
		new.append(data)
		new.pop(0)
		X = torch.FloatTensor(new)

"""
for i in range(50):

	y_pred = model(X)
	results.append(y_pred)
	x = list(X)
	x.append(y_pred)
	x.pop(0)
	X = torch.FloatTensor(x)
	print(X)
"""
results = np.asarray(results)
results = scaler.inverse_transform(results.reshape(-1,1))

y_test = np.asarray(y_test)
y_test = scaler.inverse_transform(y_test.reshape(-1,1))

plt.plot(results, label="Preds")
plt.plot(y_test, label="Data")
plt.legend()
plt.show()
