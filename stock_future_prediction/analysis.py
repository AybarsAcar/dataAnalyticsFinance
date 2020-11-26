import pandas as pd
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
# SKLearn will be used to implement the machine learning algorithms
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
# machine learning algorithms
from tensorflow import keras


# concatenate the date, stock price, and the volume data in 1 DataFrame object
# name is the enum of stock tickers i.e 'AAPL' passed as a string
def individual_stock(price_df, vol_df, name):
  return pd.DataFrame({'Date': price_df.Date, 'Close': price_df[name], 'Volume': vol_df[name]})


# return the input / output (target) data for AI / ML Model
# Target stock price today will be tomorrow's price
def trading_window(data):
  n = 1
  # create a new column and shift the data backwards
  data['Target'] = data['Close'].shift(-n)
  return data


def normalise(df):
  x = df.copy()
  for i in x.columns[1:]:
    x[i] = x[i] / x[i][0]
  return x


def interactive_plot(df, title):
  fig = px.line(title=title)
  for i in df.columns[1:]:
    fig.add_scatter(x=df.Date, y=df[i], name=i)
  fig.show()


# data plotting
def show_plot(data, title):
  plt.figure(figsize=(13, 5))
  plt.plot(data, linewidth=3)
  plt.title(title)
  plt.grid()
  plt.show()


# read and sort the stock prices data
stock_prices = pd.read_csv('../data/stocks.csv')
stock_prices = stock_prices.sort_values(by=['Date'])

# read and sort the stock volumes data - its the trades volume
stock_volumes = pd.read_csv('../data/stock_volume.csv')
stock_volumes = stock_volumes.sort_values(by=['Date'])

# check for null values
if stock_prices.isnull().sum().any():
  raise RuntimeError('Missing Data in Stock Prices')
if stock_volumes.isnull().sum().any():
  raise RuntimeError('Missing Data in Stock Volumes');

# get the stock price info
# print(stock_prices.info())
# print(stock_prices.describe())

# get the stock volume info
# print(stock_volumes.info())
# print(stock_volumes.describe())


# average trading volume for AAPL stock
# avg_apple_vol = stock_volumes['AAPL'].mean()

# max trading volume for sp500
# max_sp500_vol = stock_volumes['sp500'].max()

# which security is traded the most
# print(stock_volumes.max())

# avg stock price of the sp500 over the specified time period
# avg_sp500 = stock_prices['sp500'].mean()

# max price of TSLA stock
# max_TSLA_price = stock_prices['TSLA'].max()

# interactive_plot(stock_prices, 'Stock Prices')
# interactive_plot(normalise(stock_prices), 'Normalised Stock Prices')
# interactive_plot(stock_volumes, 'Stock Volumes')
# interactive_plot(normalise(stock_volumes), 'Normalised Stock Volumes')

# create for AAPL
price_volume_df = individual_stock(stock_prices, stock_volumes, 'AAPL')

# our expected output is the prices shifter by 1 day
# so the closing price of day[i] is the target price of day[i-1]
price_volume_target_df = trading_window(price_volume_df)

# remove the last row
price_volume_target_df = price_volume_target_df[:-1]

# scale the data so it will be  in between 0-1
# i want to normalise all of it except the date
sc = MinMaxScaler(feature_range=(0, 1))
price_volume_target_scaled_df = sc.fit_transform(price_volume_target_df.drop(columns=['Date']))
# print(price_volume_target_scaled_df)

# Create Feature and Target
# input x
# output y
X = price_volume_target_scaled_df[:, :2]  # all rows and first 2 columns excluding the target column
Y = price_volume_target_scaled_df[:, 2:]  # grab only the target column
print(X.shape)
print(Y.shape)

# Split the data
# training -> 65%
# testing -> 35%
training_data_volume = int(0.65 * len(X))  # 1402 samples

# training data
X_train = X[:training_data_volume]
Y_train = Y[:training_data_volume]

# testing data
X_test = X[training_data_volume:]
Y_test = X[training_data_volume:]

show_plot(X_train, 'Training Data')
show_plot(X_test, 'Testing Data')