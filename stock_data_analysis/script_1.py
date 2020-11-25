import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from copy import copy
# import plotly.express as px
# import plotly.figure_factory as ff
# import plotly.graph_objects as go

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
