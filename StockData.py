



##Gets historical Stock data from yahoo finance and saves as .csv file
def StockHist(TAG,startDate,endDate):

	import yfinance as yf
	import os
	import pandas
	import numpy as np
	path = os.getcwd()
	os.chdir(path+"\\StockData")

	##Getting Data
	data = yf.Ticker(str(TAG))
	hist = data.history(start = startDate, end = endDate)

	##Writing file
	pandas.DataFrame.to_csv(hist,str(TAG+".csv"))

	#Writes to binary numpy file type
	formattedData = hist.to_records()
	formattedData.tofile(str(TAG)+".npy")


StockHist("GOOGL","2018-01-01","2018-12-31")







