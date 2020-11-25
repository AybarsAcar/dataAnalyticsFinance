import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# read the data into a dataframe class
stock_df = pd.read_csv("../data/stocks.csv")
print(stock_df)

# multiple graphs in one single line graph
stock_df.plot(x='Date', y=['AAPL', 'sp500'], linewidth=3)
plt.label('Price')
plt.title('Stock Prices')
plt.grid()
