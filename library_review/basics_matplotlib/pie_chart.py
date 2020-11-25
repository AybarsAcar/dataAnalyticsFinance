import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

values = [20, 55, 5, 17, 3]
colors = ['g', 'r', 'y', 'b', 'm']
labels = ["AAPL", "GOOG", "T", "TSLA", "AMZN"]

# explode - putting an emphasis
# this shifts the "GOOG" stock a little away from the pie
explode = [0, 0.2, 0, 0, 0]


plt.figure(figsize=(7, 7))
plt.pie(values, colors=colors, labels=labels)

plt.title("STOCK PORTFOLIO")
