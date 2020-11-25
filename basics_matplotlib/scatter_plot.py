import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

daily_returns_df = pd.read_csv("../data/daily_returns.csv")
print(daily_returns_df)

# scatter plot b/w APPL and s&p500
X = daily_returns_df["APPL"]
Y = daily_returns_df["sp500"]

# when apple stock changes what happens to the sp?
plt.scatter(X, Y)
# label it
plt.xlabel('Apple daily returns')
plt.ylabel('S&P500 daily returns')
