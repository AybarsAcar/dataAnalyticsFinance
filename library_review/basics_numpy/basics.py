import numpy as np

# Defining one dimentional array
my_list = [10, 20, 50, 60, 70]
print("my list: " + str(my_list))

# Create a numpy array from the list
# Numpy array backed by arrays implemented in C
my_array = np.array(my_list)
print(my_array)

print(type(my_array))

# Multidimentional (Matrix) using numpy
matrix = np.array([[5, 6], [2, 4]])
print("Matrix: " + str(matrix))

# Built in methods in numpy ####################

# rand() gets a uniform dist between 0 and 1
# this will return an array of 15 random floats in an array
my_rand = np.random.rand(15)

# create a 5*5 matrix with random values
x = np.random.rand(5, 5)

# to get a normal distributed
normal_dist = np.random.randn(10)

# randint - is used to generate random integers betwen upper and lower bounds
my_random_int = np.random.randint(1, 10)

# randint to generate multiple random integers
# 15 random integers b/w [1,100] returned in an array
my_rand_ints = np.random.randint(1, 100, 15)

# numpy array from [1, 50)
my_array_1_50 = (np.arange(1, 50))
with_steps = np.arange(1, 50, 5)  # skips 5

# create a matrix wiht 1's in diagonals and 0's anywhere else
# 15*15 matrix wiht 1's in the diagonal
a_matrix = np.eye(15)

# Array of ones with a length of 10
x_1s = np.ones(10)

# Matrices of ones 15 by 15
matrix_of_ones = np.ones(15, 15)
