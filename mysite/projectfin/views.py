from django.views.generic import TemplateView
from django.shortcuts import render_to_response
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, Http404
from django.template import loader
from projectfin import tasks
from projectfin.forms import HomeForm
from projectfin.forms import SP500Form
from projectfin.forms import CryptoForm
import io
import os
import requests
import time
import quandl
from math import sqrt
from scipy.cluster.vq import kmeans,vq
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import statsmodels.api as sm
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
import datetime
import pickle
from scipy.stats import norm
from sklearn.covariance import GraphicalLassoCV
from sklearn import cluster, covariance, manifold
from sklearn.preprocessing import StandardScaler

os.environ['KMP_DUPLICATE_LIB_OK']='True' ## MUST CHECK MULTI THREADS ERROR

#NEED TO OPTIMIZE THESE PARAMETERS
DESIRED_LAGS = 5
FORCE_INDEX_N = [5,10,15]
EoM_N = [5,10,15]
CCI_N = [5,10,15]
STDDEV_N = [5,10,15]
ACCDIST_N = [5,10,15]
MOM_N = [5,10,15]

def GetHistoricalData(ticker):
    startdate = "2001-12-31"
    enddate =  "2018-12-31"
    quandl.ApiConfig.api_key = "tWu2uwwXw4TPWtumfozm"
    mydata = quandl.get(f"WIKI/{ticker}", start_date=startdate, end_date=enddate)
    mydata = mydata.drop(columns=['Ex-Dividend','Split Ratio','Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume'])
    return mydata

def GetHistorySP500(symbols):
	df=pd.DataFrame()
	for name in symbols:
		startdate = "2001-12-31"
		enddate =  "2018-12-31"
		quandl.ApiConfig.api_key = "tWu2uwwXw4TPWtumfozm"
		mydata = quandl.get(f"WIKI/{name}", start_date=startdate, end_date=enddate)
		df[f"{name}"]=mydata['Close']
	df.dropna()
	return df

def CalculatePredictors(df3):
	#Calculating lags
	for i in range(0,DESIRED_LAGS+1):
		df3[f"lag{i}"] = (df3['Close'].shift(i)-df3['Close'].shift(i+1))/df3['Close'].shift(i+1)
	df3=df3.dropna()
	
	#Linear Regression Models based on lags, open, close and volume
	model = sm.OLS.from_formula('lag0 ~ ' + '+'.join(df3.columns.difference(['lag0','Date'])), df3)
	result = model.fit()
	df3['Regression Predictions'] = result.predict(df3)
	for m in STDDEV_N:
	    df3[f"StdDev_{m}"] = pd.Series(df3['Close'].rolling(m).std())  

	#Short/Medium/Long Rolling Moving Average
	df3['short_rolling'] = df3['Close'].rolling(window=7).mean()
	df3['long_rolling'] = df3['Close'].rolling(window=28).mean()
	df3['med_rolling'] = df3['Close'].rolling(window=14).mean()
	df3=df3.dropna()

	#RSI
	df3['Price Difference'] = df3['Close'].diff()
	df3 = df3[1:]
	up = df3['Price Difference'].copy()
	down = df3['Price Difference'].copy()
	up[up < 0] = 0
	down[down > 0] = 0
	roll_up1 = up.ewm(alpha=0.7).mean()
	roll_down1 = up.ewm(alpha=0.7).mean()
	RSI = roll_up1 / roll_down1
	RSI1 = 100.0 - (100.0 / (1.0 + RSI))
	df3['RSI'] = RSI1
	df3=df3.dropna()

	#Stochastic Oscillator %K
	df3['SOk'] = pd.Series((df3['Close'] - df3['Low']) / (df3['High'] - df3['Low']))
	df3['SOd'] = pd.Series(df3['SOk'].ewm(span = 20, min_periods = 19).mean())
	df3=df3.dropna()

	#Chaikin Oscillator
	ad = (2 * df3['Close'] - df3['High'] - df3['Low']) / (df3['High'] - df3['Low']) * df3['Volume']  
	df3['Chaikin'] = pd.Series(ad.ewm(span = 3, min_periods = 2).mean() - ad.ewm(span = 10, min_periods = 9).mean())  
	    
	#Momentum  
	for q in MOM_N: 
	    df3[f"MOM_{q}"] = pd.Series(df3['Close'].diff(q))
	df3=df3.dropna()
	    
	#Commodity Channel Index
	PP = (df3['High'] + df3['Low'] + df3['Close']) / 3
	for k in CCI_N:
	    df3[f"CCI_{k}"] = pd.Series((PP - PP.rolling(k).mean()) / PP.rolling(k).std())

	#Mass Index
	Range = df3['High'] - df3['Low']  
	EX1 = Range.ewm(span = 9, min_periods = 8).mean() 
	EX2 = EX1.ewm(span = 9, min_periods = 8).mean()
	Mass = EX1 / EX2
	df3['MassI'] = pd.Series(Mass.rolling(25).sum())
	df3=df3.dropna()

	#Accumulation/Distribution  
	#for h in ACCDIST_N:  
	#    ad1 = (2 * df3['Close'] - df3['High'] - df3['Low']) / (df3['High'] - df3['Low']) * df3['Volume']  
	#    df3[f"Acc/Dist_ROC_{h}"] = pd.Series(ad1.diff(h - 1) / ad1.shift(h - 1))
	#df3=df3.dropna()

	#Ease of Movement
	EoM = (df3['High'].diff(1) + df3['Low'].diff(1)) * (df3['High'] - df3['Low']) / (2 * df3['Volume'])  
	for j in EoM_N:
	    df3[f"Eom_ma_{j}"] = pd.Series(EoM.rolling(j).mean()) 


	#Force Index
	for n in FORCE_INDEX_N:
	    df3[f"force_{n}"] = pd.Series(df3['Close'].diff(n) * df3['Volume'].diff(n))
	df3=df3.dropna()

	return df3


def KNNClassifier(df3):
	from sklearn import neighbors
	#Setting up Direction and Train/Test sets
	df3['Direction']='-'
	df3['Direction'][df3['lag0']>0] = 'UP'
	df3['Direction'][df3['lag0']<0] = 'DOWN'

	X = df3.drop(columns=['Direction','lag0'])
	Y = df3['Direction']
	X_train = X[:int(0.8*len(X))]
	Y_train = Y[:int(0.8*len(Y))]
	X_test = X[int(0.8*len(X)):]
	Y_test = Y[int(0.8*len(Y)):]

	knn = neighbors.KNeighborsClassifier(n_neighbors=6)
	knn.fit(X_train, Y_train)
	y_pred_knn = knn.predict(X_test)
	cm = confusion_matrix(Y_test, y_pred_knn)
	accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
	accuracy_test = accuracy_score(Y_test, y_pred_knn)
	pred = y_pred_knn[0]
	return (accuracy_train, accuracy_test, pred)

def ANNClassifier(df3):
	import keras
	from keras.models import Sequential
	from keras.layers import Dense

	#Setting up Direction and Train/Test sets
	df3['Direction']='UP'
	df3['Direction'][df3['lag0']>0] = 'UP'
	df3['Direction'][df3['lag0']<0] = 'DOWN'

	X = df3.drop(columns=['Direction','lag0'])
	labelencoder_Y = LabelEncoder()
	df3['Direction'] = labelencoder_Y.fit_transform(df3['Direction'])
	Y = df3['Direction']

	#Splitting Train/Test set and normalizing
	X_train = X[:int(0.8*len(X))]
	Y_train = Y[:int(0.8*len(Y))]
	X_test = X[int(0.8*len(X)):]
	Y_test = Y[int(0.8*len(Y)):]

	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)

	#Building ANN
	classifier = Sequential()
	classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu', input_dim = 35))
	classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

	classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
	classifier.fit(X_train, Y_train, batch_size = 10, epochs = 10)

	# Predicting the Test set results
	y_pred = classifier.predict(X_test)
	y_pred = (y_pred > 0.5)

	# Making the Confusion Matrix
	cm = confusion_matrix(Y_test, y_pred)
	#accuracy_train = accuracy_score(Y_train, classifier.predict(X_train))
	accuracy_test = accuracy_score(Y_test, y_pred)

	return (accuracy_test, y_pred[0])


def XGBClassifier(df3):
	import xgboost as xgb
	from xgboost import XGBClassifier
	#Setting up Direction and Train/Test sets
	df3['Direction']="UP"
	df3['Direction'][df3['lag0']<0] = "DOWN"

	X = df3.drop(columns=['Direction','lag0'])
	Y = df3['Direction']
	X_train = X[:int(0.8*len(X))]
	Y_train = Y[:int(0.8*len(Y))]
	X_test = X[int(0.8*len(X)):]
	Y_test = Y[int(0.8*len(Y)):]

	classifier = XGBClassifier()
	classifier.fit(X_train, Y_train)

	# Predicting the Test set results
	y_pred = classifier.predict(X_test)
	cm = confusion_matrix(Y_test, y_pred)
	accuracy_train = accuracy_score(Y_train, classifier.predict(X_train))
	accuracy_test = accuracy_score(Y_test, y_pred)

	return (accuracy_train, accuracy_test, y_pred[0])


def plot(request):
	fileObject = open("asset_choice", "rb")
	ticker = pickle.load(fileObject)

	df = GetHistoricalData(ticker)
	data = [
        go.Scatter(
            x=df['Close'].index,
            y=df['Close'].values,
        )
    ]
	layout = go.Layout(
          title=f"Close price for {ticker}",
        )
	figure = go.Figure(data=data, layout=layout)
	offline.plot(figure, auto_open=False)
	os.rename("temp-plot.html", "polls/templates/temp-plot.html")
	return HttpResponseRedirect('/projectfin/viewplot/')


def cplot(request):
	fileObject = open("asset_choice", "rb")
	ticker = pickle.load(fileObject)

	df = getCryptosData(ticker)
	data = [
        go.Scatter(
            x=df['Close'].index,
            y=df['Close'].values,
        )
    ]
	layout = go.Layout(
          title=f"Close price for {ticker}",
        )
	figure = go.Figure(data=data, layout=layout)
	offline.plot(figure, auto_open=False)
	os.rename("temp-plot.html", "polls/templates/temp-plot.html")
	return HttpResponseRedirect('/projectfin/viewplot/')

def clusterplot(request):
	df = pd.read_csv('sp500_close.csv')
	df1 = pd.read_csv('sp500_open.csv')
	df = df.dropna(axis=1)
	df = df.drop(columns='Date')
	df1 = df1.dropna(axis=1)
	df1 = df1.drop(columns='Date')
	indx = df.columns

	#Calculate average annual percentage return and volatilities over a theoretical one year period
	returns = df.pct_change().mean() * 252
	returns = pd.DataFrame(returns)
	returns.columns = ['Returns']
	returns['Volatility'] = df.pct_change().std() * sqrt(252)

	#format the data as a numpy array to feed into the K-Means algorithm
	data = np.asarray([np.asarray(returns['Returns']),np.asarray(returns['Volatility'])]).T

	X = data
	X=X[~np.isnan(X).any(axis=1)]
	K=5

	# computing K-Means with K = 5 (5 clusters)
	centroids,_ = kmeans(X,K)
	# assign each sample to a cluster
	idx,_ = vq(X,centroids)
	traces=[]
	j=0
	for i in range(0, K):
		traces.append(go.Scatter(
			x=X[idx==i,0],
			y=X[idx==i,1],
			mode = 'markers',
			text= indx,
			))
		j=j+1
	traces.append(go.Scatter(
		x=centroids.T[0],
		y=centroids.T[1],
		mode = 'markers',
		text = [f"centroid{k}" for k in range (0,K)],
		marker = dict(
			color = 'rgba(152, 0, 0, .8)',
			)
		))

	layout = go.Layout(
		title=f"Clustering SP500 Stocks with KMeans Algorithm with K={K}",
		)
	mode = [traces]
	figure = go.Figure(data=traces, layout=layout)
	offline.plot(figure, auto_open=False)
	try:
		os.rename("temp-plot.html", "polls/templates/temp-plot.html")
	except:
		pass
	return HttpResponseRedirect('/projectfin/viewplot/')

def montecplot(request):
	fileObject = open("asset_choice", "rb")
	ticker = pickle.load(fileObject)
	data={}
	dataset=pd.DataFrame()
	data1=pd.DataFrame()
	try:
		startdate = "2001-12-31"
		enddate =  "2018-12-31"
		quandl.ApiConfig.api_key = "tWu2uwwXw4TPWtumfozm"
		mydata = quandl.get(f"WIKI/{ticker}", start_date=startdate, end_date=enddate)
		dataset["Close"]=mydata['Close']
		dataset["Open"]=mydata['Open']
		dataset.dropna()
		data[f"{ticker}"] = dataset["Close"]
	except:
		url = f"https://min-api.cryptocompare.com/data/histoday?fsym={ticker}&tsym=USD&allData=true"
		page = requests.get(url)
		data = page.json()['Data']
		dataset = pd.DataFrame(data)
		dataset = dataset.drop(columns=['volumefrom','time','high','low','volumeto','open'])
		dataset = dataset.rename(columns={'close':'Close'})
		dataset.dropna()
		data1[f"{ticker}"] = dataset["Close"]
		data=data1

	log_returns = np.log(1 + data[f"{ticker}"].pct_change())
	u = log_returns.mean()
	var = log_returns.var()
	drift = u - (0.5 * var)
	drift = np.array(drift)
	stdev = log_returns.std()
	norm.ppf(0.95)
	x = np.random.rand(10, 2)
	norm.ppf(x)
	Z = norm.ppf(np.random.rand(10,2))

	t_intervals = 1000
	iterations = 10
	daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(t_intervals, iterations)))

	price_list = np.zeros_like(daily_returns)
	price_list[0] = data[f"{ticker}"].iloc[-1]

	for t in range(1, t_intervals):
		price_list[t] = price_list[t - 1] * daily_returns[t]

	data = []
	for h in range(0, len(price_list.T)):
		data.append(go.Scatter(
			x=list(range(0,len(price_list))),
			y=price_list.T[h],
			mode='lines',
			))
	
	layout = go.Layout(
          title=f"Monte Carlo Simulation for {ticker}",
        )
	figure = go.Figure(data=data, layout=layout)
	offline.plot(figure, auto_open=False)
	os.rename("temp-plot.html", "polls/templates/temp-plot.html")
	return HttpResponseRedirect('/projectfin/viewplot/')
def viewplot(TemplateView):
    return render_to_response('temp-plot.html')

def getCryptosData(symbols):
	data={}
	d = pd.DataFrame()
	url = f"https://min-api.cryptocompare.com/data/histoday?fsym={symbols}&tsym=USD&allData=true"
	page = requests.get(url)
	data = page.json()['Data']
	d = pd.DataFrame(data)
	d = d.rename(columns={'close':'Close','high':'High','low':'Low','open':'Open','volumeto':'Volume'})
	d = d.drop(columns=['volumefrom','time'])
	return d

def GetHistoryCryptos(symbols):
	d={}
	for name in symbols:
	    url = f"https://min-api.cryptocompare.com/data/histoday?fsym={name}&tsym=USD&allData=true"
	    page = requests.get(url)
	    data = page.json()['Data']
	    d[name] = pd.DataFrame(data)
	    d[name] = d[name].drop(columns = d[name].columns.difference(['close']))
	df = pd.concat([d[f"{symbol}"] for symbol in symbols],keys=symbols, axis=1, join='inner')
	return df

def cfrontierplot(request):
	symbols=['BTC','ETH','XRP','EOS','ADA']
	# calculate daily and annual returns of the stocks
	result = GetHistoryCryptos(symbols)
	returns_daily = result.pct_change()
	returns_annual = returns_daily.mean() * 250

	# get daily and covariance of returns of the stock
	cov_daily = returns_daily.cov()
	cov_annual = cov_daily * 250

	# empty lists to store returns, volatility and weights of imiginary portfolios
	port_returns = []
	port_volatility = []
	sharpe_ratio = []
	stock_weights = []

	# set the number of combinations for imaginary portfolios
	num_assets = len(symbols)
	num_portfolios = 50000

	# populate the empty lists with each portfolios returns,risk and weights
	for single_portfolio in range(num_portfolios):
	    weights = np.random.random(num_assets)
	    weights /= np.sum(weights)
	    returns = np.dot(weights, returns_annual)
	    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
	    sharpe = returns / volatility
	    sharpe_ratio.append(sharpe)
	    port_returns.append(returns)
	    port_volatility.append(volatility)
	    stock_weights.append(weights)

	# a dictionary for Returns and Risk values of each portfolio
	portfolio = {'Returns': port_returns,
	             'Volatility': port_volatility,
	             'Sharpe Ratio': sharpe_ratio}

	# extend original dictionary to accomodate each ticker and weight in the portfolio
	for counter,symbol in enumerate(symbols):
	    portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]

	# make a nice dataframe of the extended dictionary
	df = pd.DataFrame(portfolio)

	# get better labels for desired arrangement of columns
	column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in symbols]

	# find min Volatility & max sharpe values in the dataframe (df)
	min_volatility = df['Volatility'].min()
	max_sharpe = df['Sharpe Ratio'].max()
	max_return = df['Returns'].max()

	# use the min, max values to locate and create the two special portfolios
	sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
	min_variance_port = df.loc[df['Volatility'] == min_volatility]
	
	# plot frontier, max sharpe & min Volatility values with a scatterplot
	data = [
    	go.Scatter(
    		x=df['Volatility'],
    		y=df['Returns'],
    		text = ['Volatility','Returns'],
    		mode = 'markers+text'
    	)
    ]
	layout = go.Layout(
    	title='Efficient Frontier for Cryptocurrencies',
    	)
	mode = [data]
	figure = go.Figure(data=data, layout=layout)
	offline.plot(figure, auto_open=False)
	html_file= open("temp-plot.html","a+")
	html_file.write(f"<p><b>For minimum variance point:</b> \n {min_variance_port.T} \n <b>For maximum Sharpe Ratio:</b> \n {sharpe_portfolio.T}</p>")
	html_file.close()
	os.rename("temp-plot.html", "polls/templates/temp-plot.html")
	return HttpResponseRedirect('/projectfin/viewplot/')

def frontierplot(request):
	symbols=['AAPL','SPG','MMM','BLK','ACN']
	# calculate daily and annual returns of the stocks
	result = GetHistorySP500(symbols)
	returns_daily = result.pct_change()
	returns_annual = returns_daily.mean() * 250

	# get daily and covariance of returns of the stock
	cov_daily = returns_daily.cov()
	cov_annual = cov_daily * 250

	# empty lists to store returns, volatility and weights of imiginary portfolios
	port_returns = []
	port_volatility = []
	sharpe_ratio = []
	stock_weights = []

	# set the number of combinations for imaginary portfolios
	num_assets = len(symbols)
	num_portfolios = 50000

	# populate the empty lists with each portfolios returns,risk and weights
	for single_portfolio in range(num_portfolios):
	    weights = np.random.random(num_assets)
	    weights /= np.sum(weights)
	    returns = np.dot(weights, returns_annual)
	    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
	    sharpe = returns / volatility
	    sharpe_ratio.append(sharpe)
	    port_returns.append(returns)
	    port_volatility.append(volatility)
	    stock_weights.append(weights)

	# a dictionary for Returns and Risk values of each portfolio
	portfolio = {'Returns': port_returns,
	             'Volatility': port_volatility,
	             'Sharpe Ratio': sharpe_ratio}

	# extend original dictionary to accomodate each ticker and weight in the portfolio
	for counter,symbol in enumerate(symbols):
	    portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]

	# make a nice dataframe of the extended dictionary
	df = pd.DataFrame(portfolio)

	# get better labels for desired arrangement of columns
	column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in symbols]

	# find min Volatility & max sharpe values in the dataframe (df)
	min_volatility = df['Volatility'].min()
	max_sharpe = df['Sharpe Ratio'].max()
	max_return = df['Returns'].max()

	# use the min, max values to locate and create the two special portfolios
	sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
	min_variance_port = df.loc[df['Volatility'] == min_volatility]
	
	# plot frontier, max sharpe & min Volatility values with a scatterplot
	data = [
    	go.Scatter(
    		x=df['Volatility'],
    		y=df['Returns'],
    		text = ['Volatility','Returns'],
    		mode = 'markers+text'
    	)
    ]
	layout = go.Layout(
    	title='Efficient Frontier for SP500 Stocks',
    	)
	mode = [data]
	figure = go.Figure(data=data, layout=layout)
	offline.plot(figure, auto_open=False)
	html_file= open("temp-plot.html","a+")
	html_file.write(f"<p><b>For minimum variance point:</b> \n {min_variance_port.T} \n <b>For maximum Sharpe Ratio:</b> \n {sharpe_portfolio.T}</p>")
	html_file.close()
	os.rename("temp-plot.html", "polls/templates/temp-plot.html")
	return HttpResponseRedirect('/projectfin/viewplot/')

class goToSPMarket(TemplateView):
    template_name = os.path.join('templates/','form3.html')
    
    def get(request):
	    if request.method == 'GET':
	    	form = SP500Form()

	    	args = {'form': form}
	    	return render(request, 'projectfin/form3.html', args)

	    else:
	    	form = SP500Form(request.POST)
	    	form = form.data
	    	if form != 0: #if form.is_valid() was not working (FALSE so jumped the loop)
	    		asset = form['asset']
	    	else:
	    		raise Http404

	    	fileObject = open("asset_choice", "wb")
	    	pickle.dump(asset, fileObject)
	    	fileObject.close()
	    return HttpResponse('<button><a href="/projectfin/plot">Plot</a></button> <p>&nbsp;</p> <button><a href="/projectfin/pred">Predict</a></button> <p>&nbsp;</p> <button><a href="/projectfin/montecplot">Monte Carlo Simulation</a></button>')

class goToCryptoMarket(TemplateView):
	template_name = os.path.join('templates/','form2.html')

	def get(request):
	    if request.method == 'GET':
	    	form = CryptoForm()

	    	args = {'form': form}
	    	return render(request, 'projectfin/form2.html', args)

	    else:
        	form = CryptoForm(request.POST)
        	form = form.data
        	if form != 0: #if form.is_valid() was not working (FALSE so jumped the loop)
        		asset = form['asset']
        	else:
        		raise Http404

        	fileObject = open("asset_choice", "wb")
        	pickle.dump(asset, fileObject)
        	fileObject.close()
        	return HttpResponse('<button><a href="/projectfin/cplot">Plot</a></button> <p>&nbsp;</p> <button><a href="/projectfin/cpred">Predict</a></button> <p>&nbsp;</p> <button><a href="/projectfin/montecplot">Monte Carlo Simulation</a></button>')

def cpred(request):
	fileObject = open("asset_choice", "rb")
	ticker = pickle.load(fileObject)

	df3 = getCryptosData(ticker)
	df3 = CalculatePredictors(df3)
	knn_acc_tr, knn_acc_ts, knn_pred = KNNClassifier(df3)
	ann_acc_ts, ann_pred = ANNClassifier(df3)
	xgb_acc_tr, xgb_acc_ts, xgb_pred = XGBClassifier(df3)

	return HttpResponse(f'Predicting: KNN ({knn_pred})   ANN({ann_pred})   XGB({xgb_pred}) ')


def pred(request):
	fileObject = open("asset_choice", "rb")
	ticker = pickle.load(fileObject)

	df3 = GetHistoricalData(ticker)
	df3 = CalculatePredictors(df3)
	knn_acc_tr, knn_acc_ts, knn_pred = KNNClassifier(df3)
	ann_acc_ts, ann_pred = ANNClassifier(df3)
	xgb_acc_tr, xgb_acc_ts, xgb_pred = XGBClassifier(df3)

	return HttpResponse(f'Predicting: KNN ({knn_pred})   ANN({ann_pred})   XGB({xgb_pred}) ')

class HomeView(TemplateView):
    template_name = os.path.join('templates/','form1.html')

    def get(request):
        if request.method == 'GET':
            form = HomeForm()

            args = {'form': form}
            return render(request, 'projectfin/form1.html', args)

        else:
        	form = HomeForm(request.POST)
        	form = form.data
        	if form != 0:
        		market = form['market']
        	else:
        		raise Http404
        	if market=="1":
        		return HttpResponseRedirect('/projectfin/sp500')
        	else:
        		return HttpResponseRedirect('/projectfin/crypto')





