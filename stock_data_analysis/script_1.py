import pandas as pd
import matplotlib.pyplot as plt

# read the stocks data
stocks_df = pd.read_csv("../data/stocks.csv")
print(stocks_df.head())

# sort the stocks by date
stocks_df = stocks_df.sort_values(by=['Date'])

# skip the first column
print("Total Number of Stocks: {}".format(len(stocks_df.columns[1:])))

# the stocks - enhanced for loop
for stock in stocks_df.columns[1:]:
  print(stock)

# average return of the S&P500
mean_sp500 = stocks_df["sp500"].mean()

# which stock has the min dispertion from the mean in dollar value?
min_dispertion = stocks_df.std()
print(min_dispertion)

# what is the maximum price for AMZN stock over the specified time period?
# describe the data
stocks_df.describe()

# check if the data has any missing values
print(stocks_df.isnull().sum())  # we dont have any missing elements

# getting dataframe info
stocks_df.info()


# function to plot the dataframe that performs data visualisation
# @input df -> DataFrame, fig_title -> title of the graph
def show_plot(df, fig_title):
  df.plot(x='Date', figsize=(15, 7), linewidth=3, title=fig_title)
  plt.grid()
  plt.show()


show_plot(stocks_df, "RAW STOCK PRICES (WITHOUT NORMALISATION)")


# normalised stock prices // scaled stock prices
# to normalise the data
# @input df - DataFrame
def normalise(df):
  x = df.copy()
  for i in x.columns[1:]:
    x[i] = x[i] / x[i][0]
  return x


normalised_stocks = normalise(stocks_df)
show_plot(normalised_stocks, "Normalised Stock Prices")
