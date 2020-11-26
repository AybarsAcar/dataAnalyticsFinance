import pandas as pd
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go


def plot_for_all(df):
  for i in df.columns:
    if i != 'Date' and i != 'sp500':
      fig = px.scatter(df, x='sp500', y=i, title=i)
      [b, a] = np.polyfit(df['sp500'], df[i], 1)
      fig.add_scatter(x=df['sp500'], y=b * df['sp500'] + a)
      fig.show()


# security -> string i.e 'AAPL'
# 252 -> number of trading days in a year
# CAPM model is applied
def expected_return_security(df, security):
  [beta, alpha] = np.polyfit(df['sp500'], df[security], 1)
  # return of the market
  rm = df['sp500'].mean() * 252
  # risk free rate
  rf = 0
  return rf + beta * (rm - rf)


# we will return the betas and alphas in a hashtable
def calc_betas(df):
  betas = {}
  alphas = {}
  for i in df.columns:
    if i != 'Date' and i != 'sp500':
      [b, a] = np.polyfit(df['sp500'], df[i], 1)
      betas[i] = b
      alphas[i] = a
  return [betas, alphas]


# plots a line graph
def interactive_plot(df, title):
  fig = px.line(title=title)
  for i in df.columns[1:]:
    fig.add_scatter(x=df.Date, y=df[i], name=i)
  fig.show()


# helper normalise function
def normalise(df):
  x = df.copy()
  for i in x.columns[1:]:
    x[i] = x[i] / x[i][0]
  return x


# calculate the daily % returns
def calc_daily_return(df):
  df_daily_return = df.copy()
  for i in df.columns[1:]:
    for j in range(1, len(df)):
      df_daily_return[i][j] = ((df[i][j] - df[i][j - 1]) / df[i][j - 1]) * 100
    df_daily_return[i][0] = 0;
  return df_daily_return


# returns the beta for the stock
def stock_beta(df, stock):
  df = calc_daily_return(df)
  stock_daily_returns = df[stock];
  market_daily_return = df['sp500']


# read the data into a DataFrame object
stocks_df = pd.read_csv('../data/stocks.csv')
stocks_df = stocks_df.sort_values(by=['Date'])

# interactive_plot(stocks_df, "Stocks Graph")
# interactive_plot(normalise(stocks_df), "Normalised Stocks Graph")

daily_returns = calc_daily_return(stocks_df)
# interactive_plot(daily_returns, "Daily Returns")

# beta -> measure of volatility compared to the market
# alpha -> excess return on top of the market
[beta, alpha] = np.polyfit(daily_returns['sp500'], daily_returns['AAPL'], 1)
print("Beta for {} stock: {}".format("AAPL", beta))
# print("Alpha for {} stock: {}".format("AAPL", alpha))

# this will give us the fit line
# the slope of this polynomial fit will be the beta for AAPL against the market
# daily_returns.plot(kind='scatter', x='sp500', y='AAPL')
# plt.plot(daily_returns['sp500'], beta * daily_returns['sp500'] + alpha, color='r')
# plt.show()

[beta_tesla, alpha_tesla] = np.polyfit(daily_returns['sp500'], daily_returns['TSLA'], 1)
print("Beta for {} stock: {}".format("TSLA", beta_tesla))
print("Alpha for {} stock: {}".format("TSLA", alpha_tesla))

print("------------------\n")

ER_AAPL = expected_return_security(daily_returns, 'AAPL')
print("Expected return for AAPL = {}".format(ER_AAPL))
print("Expected return for TSLA = {}".format(expected_return_security(daily_returns, 'TSLA')))

print("------------------\n")

[betas, alphas] = calc_betas(daily_returns)
print(betas)
print(alphas)

print("------------------\n")

# Expected return of the whole portfolio
keys = list(betas.keys())
ER = {}
rf = 0
rm = daily_returns['sp500'].mean() * 252
for key in keys:
  ER[key] = rf + (betas[key] * (rm - rf))

print(ER)

# ER of portfolio assumed equal weights
sum = 0
for stock_return in ER.values():
  sum += stock_return
print("Expected return of the portfolio with equal weights = {}".format(sum / len(ER)))

ER_2 = (ER['AAPL'] + ER['AMZN']) / 2
print(ER_2)
