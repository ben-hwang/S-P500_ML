import pandas as pd 
import os
import time
from time import mktime
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt 
from matplotlib import style 
style.use("dark_background")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import re
#=====================================================================================

data_path = "/Users/benhwang/Desktop/investing ML/investing data"

#get the debt/equity stat
def data_collection(stat):
	statspath = data_path + '/_KeyStats'

	#names of each stock (a, aa, aapl)
	stock_list = [x[0] for x in os.walk(statspath)]
	stock_list.sort()
	df = pd.DataFrame(columns = ['Date', 
								 'Unix', 
								 'Ticker', 
								 'D/E Ratio', 
								 'Stock Price', 
								 'Stock Price Change %',
								 'SP500',
								 'SP500 Change %',
								 'Difference',
								 'Status'])

	sp500_df = pd.DataFrame.from_csv("s&p500.csv")

	tickerlist = []

	#each directory is each file (each stock) in _KeyStats
	for directory in stock_list[1:]:

		#eachfile is array of all files in directory
		eachfile = os.listdir(directory)
		eachfile.sort()

		#gets each company ticker
		ticker = directory.split("/")[-1]
		tickerlist.append(ticker)

		start_stock_value = False
		start_sp500_value = False

		if len(eachfile) > 0:
			#each singular file
			for file in eachfile:

				#time of each stock
				date_time = datetime.strptime(file, '%Y%m%d%H%M%S.html')
				unix_time = time.mktime(date_time.timetuple())

				#get path of file so we can open it
				fullpath = directory + '/' + file
				content = open(fullpath, 'r').read()

				try:
					try:
						value = float(content.split(stat + ':</td><td class="yfnc_tabledata1">')[1].split('</td>')[0])
					except:
						try:
							value = float(content.split(stat + ':</td>\n<td class="yfnc_tabledata1">')[1].split('</td>')[0])
						except:
							pass

					try:
						stock_price = float(content.split('</small><big><b>')[1].split('</b></big>')[0])
					except:

						try:
							stock_price = content.split('</small><big><b>')[1].split('</b></big>')[0]
							stock_price = re.search(r' (\d{1,8}\.\d{1,8})', stock_price)
							stock_price = float(stock_price.group[1])
						except:
							stock_price = content.split('<span class = "time_rtq_ticker">')[1].split('</span>')[0]
							stock_price = re.search(r' (\d{1,8}\.\d{1,8})', stock_price)
							stock_price = float(stock_price.group[1])

					try:
						sp500_date = datetime.fromtimestamp(unix_time).strftime('%Y-%m-%d')
						row = sp500_df[(sp500_df.index == sp500_date)]
						sp500_value = float(row["Adj Close"])
					except:
						sp500_date = datetime.fromtimestamp(unix_time-259200).strftime('%Y-%m-%d')
						row = sp500_df[(sp500_df.index == sp500_date)]
						sp500_value = float(row["Adj Close"])

					if (not start_stock_value):
						start_stock_value = stock_price

					if (not start_sp500_value):
						start_sp500_value = sp500_value

					stock_price_change = ((stock_price - start_stock_value)/start_stock_value) * 100
					sp500_price_change = ((sp500_value - start_sp500_value)/start_sp500_value) * 100

					difference = stock_price_change - sp500_price_change

					if (difference > 0):
						status = "Outperform"
					else:
						status = "Underperform"

					df = df.append({'Date': date_time, 
									'Unix': unix_time, 
									'Ticker': ticker,
									'D/E Ratio': value,
									'Stock Price': stock_price,
									'SP500': sp500_value,
									'Stock Price Change %': stock_price_change,
									'SP500 Change %': sp500_price_change,
									'Difference': difference,
									'Status': status}, ignore_index = True)

				except:
					pass


	outfile = stat.replace(' ', '').replace(')', '').replace('(', '').replace('/', '') + ('.csv')
	df.to_csv(outfile)

	feature_names = ['Unix', 
				 'D/E Ratio', 
				 'SP500', 
				 'Stock Price Change %', 
				 'SP500 Change %',
				 'Difference'] 

	y = df['Stock Price Change %']
	X = df[list(feature_names)]

	X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2)
	rf = RandomForestRegressor()
	rf.fit(X_train, Y_train)
	y_pred =rf.predict(X_test)
	r2 = r2_score(Y_test, y_pred)
	mse = mean_squared_error(Y_test,y_pred)

	print(y_pred)

	print(r2, mse)

data_collection('Total Debt/Equity (mrq)')





