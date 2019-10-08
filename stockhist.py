



##Gets historical Stock data
def stockhist(TAG):

	import yfinance as yf
	import os
	import pandas

	os.chdir("C:\\Users\\chris\\index\\Data")

	str(TAG)
	##Getting Data
	data = yf.Ticker(TAG)
	hist = data.history(start = "2018-01-01", end = "2018-12-31")
	

	##Writing file
	filename = str(TAG+".csv")
	pandas.DataFrame.to_csv(hist,filename)

stockhist("GOOGL")





