import os
import sys
sys.path.append(r'c:\program files\python38\lib\site-packages')

import yfinance as yf
from sklearn.linear_model import LinearRegression
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
	pandas.DataFrame.to_csv(hist,str(TAG+".csv"))











