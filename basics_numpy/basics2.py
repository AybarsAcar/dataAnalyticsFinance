import numpy as np

# define a one dimensional array
my_list = [-30, 4, 50, 60, 29, 15, 22, 90]
my_numpy_array = np.array(my_list)

# lenght
print(len(my_numpy_array))

# shape
print(my_numpy_array.shape)

# data type
print(my_numpy_array.dtype)

# reshape the array into a 2D matrix into a 2 by 4
my_matrix = my_numpy_array.reshape(2, 4)

# maximum and minimum values
print(my_numpy_array.min)
print(my_numpy_array.max)

# getting the index of the min and max element
my_numpy_array.argmax()
my_numpy_array.argmin()

# random 20*20 numpy array of rand values between -1000 and 1000
x = np.random.randint(-1000, 1000, (20, 20))
max_value = x.max
min_value = x.min
mean_value = x.mean
