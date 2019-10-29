import os
import yfinance as yf

import pandas
import numpy as np

path = os.getcwd()
os.chdir(path+"\\StockData")


##Gets historical Stock data from yahoo finance and saves as .csv file
def StockHist(TAG,startDate,endDate):

	##Getting Data
	data = yf.Ticker(str(TAG))

	hist = data.history(start = startDate, end = endDate,interval = "1d")

	##Writing file

	np.save(str(TAG)+".npy",hist)
	pandas.DataFrame.to_csv(hist,str(TAG+".csv"))








StockHist("AAPL","2018-01-01","2018-12-31")
StockHist("GOOGL","2018-01-01","2018-12-31")

