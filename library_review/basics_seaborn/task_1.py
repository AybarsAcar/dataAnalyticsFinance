import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import the cancer dataset from the sklearn library
x1 = np.array([1, 2, 3])
print(x1.shape)

x2 = np.array([4, 5, 6])

# pairs every elemnt
z = np.c_[x1, x2]
print(z)  # prints [[1, 4], [2, 5], [3, 6]] -> int[][]
print(z.shape)  # prints (3, 2)
