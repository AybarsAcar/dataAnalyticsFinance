import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# generate 4 random datasets
np.random.seed(20)

# args to normal funcion -> mean, std, data.size
# normal function generates a normally distributed data
data_1 = np.random.normal(200, 20, 20000)
data_2 = np.random.normal(60, 30, 20000)
data_3 = np.random.normal(70, 20, 20000)
data_4 = np.random.normal(40, 5, 20000)

data = [data_1, data_2, data_3, data_4]

fig = plt.figure(figsize=(10, 7))

# 111 -> 1 grid, 1 image
axes = fig.add_subplot(111)

# plot the boxplot
boxplot = axes.boxplot(data)
