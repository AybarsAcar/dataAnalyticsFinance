import pandas as pd
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go


# create a table where each stock starts at their initial $value -> weight * total investment
# and an additional column to store the sum of all $ values in the portfolio
# df -> portfolio: DataFrame object
# weights -> numpy array where length == len(df)
# initial_investment -> double
def create_portfolio_table(df_initial, weights_array, initial_investment):
  df = df_initial.copy()
  df = normalise(df)
  for counter, stock in enumerate(df.columns[1:]):
    df[stock] = df[stock] * weights_array[counter]
    df[stock] = df[stock] * initial_investment

  # add a daily total worth column
  df['portfolio daily worth in $'] = df[df != 'Date'].sum(axis=1)

  # add a % change in value column to the portfolio
  df['portfolio daily % return'] = 0.0000
  for i in range(1, len(stocks_df)):
    df['portfolio daily % return'][i] = (
                                            (df['portfolio daily worth in $'][i] -
                                             df['portfolio daily worth in $'][i - 1]) /
                                            df['portfolio daily worth in $'][i - 1]) * 100
  return df


# helper normalise function
def normalise(df):
  x = df.copy()
  for i in x.columns[1:]:
    x[i] = x[i] / x[i][0]
  return x


# generate random weights array to analyse the portfolio
def generate_weights_array(number_of_stocks):
  np.random.seed()
  weights = np.array(np.random.random(number_of_stocks))
  return weights / np.sum(weights)


def interactive_plot(df, title):
  fig = px.line(title=title)
  for i in df.columns[1:]:
    fig.add_scatter(x=df['Date'], y=df[i], name=i)
  fig.show()


def get_cummulative_return(df):
  return ((df['portfolio daily worth in $'][-1:] - df['portfolio daily worth in $'][0]) / \
          df['portfolio daily worth in $'][0]).values[0]


# 252 is the number of trading days in a yer
def get_sharpe_ratio(df):
  return df_portfolio['portfolio daily % return'].mean() / df_portfolio['portfolio daily % return'].std() * np.sqrt(252)


# read the data into a DataFrame object
stocks_df = pd.read_csv('../data/stocks.csv')
stocks_df = stocks_df.sort_values(by=['Date'])

# Portfolio Allocation
# generate the random weights for the assets
np.random.seed(101)  # passing in a bit value ensures we get the same random pseudo values

# create random weights
weights = np.array(np.random.random(9))

# Ensure the sum of weights == 1 -> normalise them
weights = weights / np.sum(weights)

# update the stocks df
df_portfolio = create_portfolio_table(stocks_df, weights, 1000000)

# fig = px.line(x=df_portfolio.Date, y=df_portfolio['portfolio daily % return'], title="Daily Return")
# fig.show()

# interactive_plot(df_portfolio.drop(['portfolio daily worth in $', 'portfolio daily % return'], axis=1),
#                  "Portfolio individual stock")

# fig = px.histogram(df_portfolio, x='portfolio daily % return')
# fig.show()
#
# fig = px.line(x = df_portfolio.Date, y=df_portfolio['portfolio daily worth in $'], title="Portfolio value")
# fig.show()

cummReturn = get_cummulative_return(df_portfolio)
print(cummReturn)

print("Standard Deviation of Portfolio: {}".format(df_portfolio['portfolio daily % return'].std()))
print("Average Daily Return of Portfolio: {}".format(df_portfolio['portfolio daily % return'].mean()))
print("Sharpe Ratio of Portfolio: {}".format(get_sharpe_ratio(df_portfolio)))
