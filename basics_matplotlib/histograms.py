import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

daily_returns_df = pd.read_csv("../data/daily_returns.csv")
print(daily_returns_df)

# pick APPL and plot its daily return histogrmas
# i want the mean and the std of the stock
mu = daily_returns_df["AAPL"].mean()
sigma = daily_returns_df["AAPL"].std()

print("Apple mean: " + str(mu))
print("Apple standard deviation: " + str(sigma))

# print out the histogram
plt.figure(figsize=(7, 5))
num_bins = 40
plt.hist(daily_returns_df["AAPL"], num_bins)

# add a grid
plt.grid()

plt.title("Histogram: mu= " + str(mu) + ", sigma= " + str(sigma))
