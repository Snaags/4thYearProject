import os
import pandas
import numpy as np
import random
path = os.getcwd()

OIL = pandas.read_csv(path+"/StockData/OIL.csv")
OILD = OIL["Date"]
APPLD = pandas.read_csv(path+"/StockData/AAPL.csv")
APPLD = APPLD[["Date"]]
APPLD = np.asarray(APPLD)#convert to numpy array
OILD = np.asarray(OILD)
out = []
for i in APPLD:
	print(i)
	print(OILD)
	x = np.where(OILD == i)
	print(x)
	out.append(OIL[x,:])
print(len(out))