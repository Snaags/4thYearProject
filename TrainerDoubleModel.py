import torch
import os
import pandas
from model import LSTMModel
from modelpre import LSTMModelpre
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




def RunModel(X,lr ,hiddenDimension,seq_length=10,numberLayers = 1,batch_size = 100,l2= 1e-5,num_epochs = 5,ID= None, number = None):
	"""
	test = []
	if ID != None:
		for i in ID:
			test.append(ID[i].size())
	print(test)
	"""
#Create dictionary of hyperparameters
	print("starting model",number)
	dropout = 0
	predict_distance = 8
	HyperParameters = [
		
		 lr,
		 hiddenDimension,
		 seq_length,
		numberLayers,
		batch_size,
		num_epochs,
		l2,
		ID,
		number
	]



###Data Processing####################################################################################################

	## transforms time series into samples of length "seq_length" each paired with a lable for trainging "predict_distance" after the last element in the sample 
	def GroupData(X,seq_length,predict_distance):

		x_out = []
		y_out = []

		for i in range(len(X)-(seq_length + predict_distance)):
			hold = X[i:(i+seq_length)]
			sample = tuple(hold)
			lable = tuple(X[i+(seq_length + predict_distance)-1])
			x_out.append(sample)
			y_out.append(lable)

		return x_out, y_out

	def test_train_split(file,train_size):

		X_train = file[:int(len(file)*train_size)]
		X_test = file[int(len(file)*train_size):]

		return X_train, X_test



	scalers = []
	scaled_data = []
	if X.ndim != 1:

		for i in range(len(X[0,:])):
			scalers.append(MinMaxScaler(feature_range=(0, 1)))
		#scaler = MinMaxScaler(feature_range=(-1, 1))	#scale data
		#scaler1 = MinMaxScaler(feature_range=(-1, 1))	#scale data	
		#scale2 = MinMaxScaler(feature_range=(-1, 1))	#scale data
		#scaler3 = MinMaxScaler(feature_range=(-1, 1))	#scale data	
		#scaler4 = MinMaxScaler(feature_range=(-1, 1))	#scale data
		index_count = 0
		for i in scalers:
			scaled_data.append(i.fit_transform(X[:,index_count].reshape(-1, 1)))
			index_count +=1 


		"""	X0 = scaler.fit_transform(X[:,0].reshape(-1, 1))
		X1 = scaler1.fit_transform(X[:,1].reshape(-1, 1))
		X2 = scaler.fit_transform(X[:,2].reshape(-1, 1))
		X3 = scaler1.fit_transform(X[:,3].reshape(-1, 1))
		X4 = scaler1.fit_transform(X[:,3].reshape(-1, 1))"""

		X = np.stack((scaled_data),1)

	else:
		scalers.append(MinMaxScaler(feature_range=(0, 1)))	#scale data
		X = scalers[0].fit_transform(X.reshape(-1, 1))	
	input_dim = np.shape(X)[1] -1


	X_train, X_test = test_train_split(X,0.85) #Training data from 80% of the total data set


	X, y = GroupData(X_train,seq_length,predict_distance)

	y = np.asarray(y)

	##Convert samples and lables
	samples = torch.cuda.FloatTensor(X)

	samplesX = samples[:,:,1,:]
	samples = samples[:,:,1:,:]

	samples = torch.transpose(samples,0,1)
	samples = torch.squeeze(samples)
	samples = torch.split(samples,batch_size,dim = 1)
	samples = list(samples)
	del samples[-1]

	samplesX = torch.transpose(samplesX,0,1)
	samplesX = torch.squeeze(samplesX)
	samplesX = torch.split(samplesX,batch_size,dim = 1)
	samplesX = list(samplesX)
	del samplesX[-1]
	samples = tuple(samples)
	samplesX = tuple(samplesX)

	lables = torch.cuda.FloatTensor(y[:,0])
	#lables = torch.transpose(lables,0,1)
	lablesplot = torch.FloatTensor(y[:,0])
	lables = torch.squeeze(lables)
	lables = torch.split(lables,batch_size,dim = 0)

	lables = list(lables)
	del lables[-1]
	lables = tuple(lables)


	

######################################################################################################################
	

	modelpre = LSTMModelpre(input_dim = input_dim, hidden_dim = hiddenDimension,seq= seq_length, output_dim=1, layer_dim=numberLayers,dropout = dropout, batch_size = batch_size)
	modelpost = LSTMModel(input_dim = 2, hidden_dim = hiddenDimension,seq= seq_length, output_dim=1, layer_dim=1,dropout = dropout, batch_size = batch_size)
	if ID != None:
		



		#model.load_state_dict(torch.load(path+"/Models/"+str(ID)+".pth")) 
		#model = torch.load(path+"/Models/"+str(ID)+".pth")
		for i in model.state_dict():
			if i in ID:
				test = ID[i].size()
				if len(model.state_dict()[i].size()) == 2:


					if model.state_dict()[i].size()[0]-ID[i].size()[0] < 0:
						idx = torch.tensor([0,model.state_dict()[i].size()[0]])
						ID[i] = torch.index_select(ID[i],0,idx)
					if model.state_dict()[i].size()[1]-ID[i].size()[1] < 0:	
						idx = torch.tensor([0,model.state_dict()[i].size()[1]])
						ID[i] = torch.index_select(ID[i],1,idx)


					if model.state_dict()[i].size()[0]-ID[i].size()[0] > 0:
						paddingRows = torch.zeros(model.state_dict()[i].size()[0]-ID[i].size()[0],ID[i].size()[1])
						ID[i] = torch.cat((ID[i],paddingRows),0)

					if model.state_dict()[i].size()[1]-ID[i].size()[1] > 0:	
						paddingCols = torch.zeros(ID[i].size()[0],model.state_dict()[i].size()[1]-ID[i].size()[1])
						
						ID[i] = torch.cat((ID[i],paddingCols),1)
				
				elif len(model.state_dict()[i].size()) == 1:


					if model.state_dict()[i].size()[0]-ID[i].size()[0] < 0:
						idx = torch.tensor([0,model.state_dict()[i].size()[0]])
						ID[i] = torch.index_select(ID[i],0,idx)


					if model.state_dict()[i].size()[0]-ID[i].size()[0] > 0:	
						paddingRows = torch.zeros(model.state_dict()[i].size()[0]-ID[i].size()[0])
						
						ID[i] = torch.cat((ID[i],paddingRows),0)
				try:
					model.state_dict()[i].data.copy_(ID[i])
				except:
					print(model.state_dict()[i])
					print(ID[i])
					print(model.state_dict()[i].size())
					print(ID[i].size())
					print(number)
					print(hiddenDimension)




					

				
		
		
		

	modelpre.cuda()
	modelpost.cuda()




	loss_fn1 = torch.nn.MSELoss(reduction = "none")
	loss_fn2 = torch.nn.MSELoss(reduction = "none")
	

	
	optimiser1 = torch.optim.Adam(modelpre.parameters(), lr=lr, weight_decay=l2)

	optimiser2 = torch.optim.Adam(modelpost.parameters(), lr=lr, weight_decay=l2)

	#res = torch.cuda.FloatTensor()
	startTime = time.time()
	modelpre.init_hidden()
	modelpost.init_hidden()
	for t in range(num_epochs):

		#for param in model.parameters():
		#	print(param.size())

		# Clear stored gradient
		modelpre.zero_grad()
		modelpre.init_hidden()
		modelpost.zero_grad()
		modelpost.init_hidden()

		for X,X2,y in zip(samples,samplesX,lables):
			y_pred = modelpre(X)
			#res = torch.cat((res, y_pred),0)
			loss1 = loss_fn1(y_pred[-1,:], y)

			# Backward pass
			loss1.sum().backward(retain_graph=True)

			# Update parameters
			optimiser1.step()
			modelpre.zero_grad()
			# Zero out gradient, else they will accumulate between epochs


			y_pred = modelpost(torch.cat((y_pred,X2),0))
			#res = torch.cat((res, y_pred),0)

			loss2 = loss_fn2(y_pred, y)

			# Backward pass
			loss2.sum().backward()

			# Update parameters
			optimiser2.step()
			modelpost.zero_grad()
			# Zero out gradient, else they will accumulate between epochs
			
		#torch.save(model,path+"/Models/"+str(lr)+".pth")
		if t % 5 ==  0:
			print("Epoch",t,"completed in:",time.time()-startTime,"seconds")



	##Debugging
	#for i in res:
	#	for c in i:
	#			out.append(c)
	"""
	res = list(res.cpu())
	plt.plot(res, label="Preds")
	plt.plot(lablesplot, label=("Data days ahead"))
	plt.legend()
	plt.savefig(("Graphs/"+str(HyperParameters.keys())+".png"))
	plt.clf()
	"""






	X, y = GroupData(X_test,seq_length,predict_distance)

	##Convert samples and lables
	y = np.asarray(y)

	##Convert samples and lables
	samples = torch.cuda.FloatTensor(X)

	samplesX = samples[:,:,1,:]
	samples = samples[:,:,1:,:]

	samples = torch.transpose(samples,0,1)
	samples = torch.squeeze(samples)
	samples = torch.split(samples,batch_size,dim = 1)
	samples = list(samples)
	del samples[-1]

	samplesX = torch.transpose(samplesX,0,1)
	samplesX = torch.squeeze(samplesX)
	samplesX = torch.split(samplesX,batch_size,dim = 1)
	samplesX = list(samplesX)
	del samplesX[-1]
	samples = tuple(samples)
	samplesX = tuple(samplesX)


	lables = torch.cuda.FloatTensor(y[:,0])
	#lables = torch.transpose(lables,0,1)
	lablesplot = torch.FloatTensor(y[:,0])
	lables = torch.squeeze(lables)
	lables = torch.split(lables,batch_size,dim = 0)
	lablesplot = lablesplot[0:-len(lables[-1])]
	lables = list(lables)
	del lables[-1]
	lables = tuple(lables)



	loss_fn = torch.nn.MSELoss(reduction = "mean")
	
	#model.init_hidden()
	results = torch.cuda.FloatTensor()
	for X,X2,y in zip(samples,samplesX,lables):

		X3 = modelpre(X)
		y_pred = modelpost(torch.cat((X3,X2),0))
		results = torch.cat((results, y_pred),0)

	##test_lost_score =  list(test_lost_score.cpu())

	lables = lablesplot
	results = results.cpu().detach()
	lables = lables.cpu().detach()
	results = np.asarray(results)
	x = torch.FloatTensor()
	for i in samples:
		i= i.cpu()
		x = torch.cat((x,i[-1,:]),0)
	plt.plot(x)
	plt.plot(results)
	plt.show()
	print(np.shape(results))
	results = scalers[0].inverse_transform(results.reshape(-1,1))

	test_lost_score = 0
	test_lost_score = np.asarray(test_lost_score)
	test_lost_score = scalers[0].inverse_transform(test_lost_score.reshape(-1,1))

	lables = np.asarray(lables)
	lables = scalers[0].inverse_transform(lables.reshape(-1,1))




	##Directional Symmetry

	t_old = results[0]
	tprime_old = lables[0]
	d = 0
	for t,tprime in zip(results[1:],lables[1:]):



		if (t - t_old)*(tprime - tprime_old) >-0:
			d += 1

		t_old = t
		tprime_old = tprime 

	DS = d*(100/(len(results)-1))
	DS = 100 - DS



	##Mean Squared Error
	results = torch.cuda.FloatTensor(results)
	lables = torch.cuda.FloatTensor(lables)
	loss = loss_fn(results, lables)



	##Mean absolute percentage error (MAPE)

	total_error = 0
	for i,c in zip(results,lables):
		test_lost_score = abs(c - i)
		test_lost_score = test_lost_score/c
		total_error += test_lost_score

	total_error = total_error/ len(lables)

	error = total_error*100

	results = results.cpu()#.detach()
	lables = lables.cpu()#.detach()
	results = np.asarray(results)
	lables = np.asarray(lables)



	plt.figure(figsize = [19.20,10.80])
	plt.plot(results, label="Preds")
	plt.plot(lables, label=("Data days ahead"))
	plt.legend()
	plt.savefig(("Graphs/"+str(float(loss))+".pdf"),dpi=1200)
	plt.clf()

	#MAPE
	print("Total training and validation time for model ",number,": ",time.time() - startTime," \n lr:",lr ," ;hiddenDimension:",hiddenDimension," ;numberLayers:",numberLayers," ;seq_length:",seq_length," ;Batch Size:",batch_size,"l2:",l2," -- MAPE:",float(error)," -- MSE:",float(loss)," -- DS:",float(DS))
	

	
	#RMSE
	#print("lr:",lr ," ;hiddenDimension:",hiddenDimension," ;numberLayers:",numberLayers," ;seq_length:",seq_length, " -- RMSE:",float(error))
	
	#print(model.state_dict())



	#torch.save(model,path+"/Models/"+str(ID)+".pth")



	ReturnDict = {

	"EvalScore": float(loss),

	}

	return ReturnDict








##Inputs
"""
lr
hidden dim
number layers
sequence length
"""

#Size of search
#range of values


