import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(6, 6))

# create a grid 1 by 1
ax = fig.add_subplot(111, projection='3d')

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [5, 6, 2, 3, 13, 4, 1, 2, 4, 8]
z = [2, 3, 3, 5, 7, 9, 11, 9, 10, 12]

ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X label')
ax.set_ylabel('Y label')
ax.set_zlabel('Z label')

# 3D plot for sp500, GOOG, AAPL
daily_returns_df = pd.read_csv("../data/daily_returns.csv")

apple = daily_returns_df['AAPL'].tolist()
google = daily_returns_df['GOOG'].tolist()
sp = daily_returns_df['sp500'].tolist()

ax.scatter(apple, google, sp, c='r', marker='o')

ax.set_xlabel('AAPL')
ax.set_ylabel('GOOG')
ax.set_zlabel('S&P500')
