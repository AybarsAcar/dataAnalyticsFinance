import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff


# function to plot the dataframe that performs data visualisation
# @input df -> DataFrame, fig_title -> title of the graph
def show_plot(df, fig_title):
  df.plot(x='Date', figsize=(15, 7), linewidth=3, title=fig_title)
  plt.grid()
  plt.show()


def plot_heatmap(matrix, boolean):
  plt.figure(figsize=(10, 10))
  sns.heatmap(matrix, annot=boolean)
  plt.show()


def plot_interactive_histogram(df):
  df_hist = df.copy()
  df_hist = df_hist.drop(columns=["Date"])

  data = []
  for i in df_hist.columns:
    data.append(df[i].values)

  fig = ff.create_distplot(data, df_hist.columns)
  fig.show()


# calculate the daily % returns
def calc_daily_return(df_stock):
  df_daily_return = df_stock.copy()
  for i in range(1, len(df_stock)):
    # calculate the % change -> daily return
    df_daily_return[i] = ((df_stock[i] - df_stock[i - 1]) / df_stock[i - 1]) * 100
  df_daily_return[0] = 0
  return df_daily_return


# calculate multiple stocks daily returns
# df has more than 1 stock i.e [[1,2,3],[2,5,4],[1,14,5]]
def calc_daily_returns(df):
  df_daily_return = df.copy()

  # loop stocks
  for i in df.columns[1:]:
    # loop the returns in each stock
    for j in range(1, len(df)):
      df_daily_return[i][j] = ((df[i][j] - df[i][j - 1]) / df[i][j - 1]) * 100

    df_daily_return[i][0] = 0

  return df_daily_return


# read the stocks data
stocks_df = pd.read_csv("../data/stocks.csv")
# sort by date
stocks_df = stocks_df.sort_values(by=['Date'])

# daily return on the S&P500
df_sp500 = stocks_df['sp500']

daily_return_amazon = calc_daily_return(stocks_df['AMZN'])
daily_return_sp500 = calc_daily_return(stocks_df['sp500'])
# print(daily_return_sp500)


daily_return_1 = calc_daily_returns(stocks_df)
# print(daily_return_1.head())
# show_plot(daily_return_1, "Daily Returns")

# Correlations between daily returns
cm = daily_return_1.drop(columns=['Date']).corr()
# print(cm)
# plot_heatmap(cm, True)

# histogram
# daily_return_1.hist(figsize=(10,10), bins=40)
# plt.show()

# plot_interactive_histogram(daily_return_1)
