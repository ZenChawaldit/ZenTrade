import quandl
import pandas as pd
import matplotlib as plt
import os
quandl.ApiConfig.api_key = "5hP1qvLqKeLFqAHnA9yG"
#check if the csv is in current folder
def getCSV(ticker, days):
	if checkInPath(ticker, days):
		df = pd.read_csv("Stock_Data/%s-%d.csv"%(ticker, days), sep = ",")
		length = len(df["Adj. Close"])
		#days = 90
		#print("before drop",df)
		df = (df.drop([i for i in range(days, length)]))
		#cut the csv and save
		df.to_csv("Stock_Data/%s-%d.csv"%(ticker, days))
		#print("after drop",df)
	else:
		#get data from online if data not already in folder
		#https://www.quandl.com/api/v3/datasets/OPEC/ORB.csv?start_date=2003-01-01
		quandl.ApiConfig.api_key = "5hP1qvLqKeLFqAHnA9yG"
		df = pd.read_csv('https://www.quandl.com/api/v3/datasets/WIKI/%s/data.csv?start_date=2017-11-01&api_key=5hP1qvLqKeLFqAHnA9yG' %(ticker), sep=',')
		#df = df.iloc[::-1]
		
		df.to_csv("Stock_Data/%s-%d.csv"%(ticker, days))
		#print(df.columns)
	#print(df)

#check if a file is in current folder
def checkInPath(ticker, days):
	if os.path.isfile("Stock_Data/%s-%d.csv"%(ticker, days)):
		#print("inFolder")
		return True
	else:
		#print("not in folder")
		return False
#getCSV("AAPL", 200)