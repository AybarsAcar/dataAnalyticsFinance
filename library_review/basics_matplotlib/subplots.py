import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

stock_df = pd.read_csv("../../data/stocks.csv")

plt.figure(figsize=(8, 5))

# print out the stocks side by side
# 1 row, 2 figures next to each other and pring the first element
plt.subplot(1, 2, 1)
plt.plot(stock_df["AAPL", 'r--'])

# now print out the second graph
plt.subplot(1, 2, 2)
plt.plot(stock_df["sp500", 'b.'])

plt.grid()

# print them on top of each other
plt.subplot(2, 1, 1)
plt.plot(stock_df["AAPL", 'r--'])

plt.subplot(2, 1, 2)
plt.plot(stock_df["sp500", 'b.'])
