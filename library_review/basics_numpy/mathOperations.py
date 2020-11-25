import numpy as np

x = np.arange(1, 10)
y = np.arange(1, 10)

# add 2 numpy arrays together
# loops in C and adds the elements at the same index
sum = x + y

squared = x**2

sqrt = np.sqrt(squared)  # goes back to your original array x

# get the exponential of the elements in the array
z = np.exp(y)

# get the Eucledian distance between 2 given arrays
a = np.array([3, 20, 30])
b = np.array([4, 6, 7])

eucledian_distance = np.sqrt(a**2 + b**2)
