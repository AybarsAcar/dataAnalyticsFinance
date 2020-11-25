import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# read the data into a dataframe class
stock_df = pd.read_csv("../data/stocks.csv")
print(stock_df)

# plot the stock dataframe
stock_df.plot(x='Date', y='AAPL', label='AAPL Stock Prices',
              color='b', linewidth=3)
plt.ylabel('Price')
plt.legend(loc="upper left")
plt.title("Apple Stock over Time")
