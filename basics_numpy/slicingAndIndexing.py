import numpy as np

a = np.array([3, 20, 30, 40, 50])

# accessing elements with their index
print(a[0])

# optain more than 1 element
print(a[0:3])  # returns the elements at index 0, 1, 2

# changing more than 1 elements to same value (broadcasting)
a[0:2] = 10

# 2D numpy array - 5by5 matrix with 1-10 random integers
matrix = np.random.randint(1, 10, (5, 5))

# get a row from a matrix
matrix[2]  # returns the 3rd row (row at index 2)

# get the specific element
matrix[2][3]  # returns the element at index 3 of the array at index 2

# get a matrix out of the matrix
sub_matrix = matrix[:3]  # returns the first 3 rows of the matrix

print("Original Matrix: \n")
print(matrix)

# [1:3] 3 is exlusive, think of it as the length
print("Sub Matrix - Slice from index 1 to 3 and 1 to 3 at each array inside the array: \n")
sub_matrix_2 = matrix[1:3, 1:3]
print(sub_matrix_2)

print("Only first 2 columns: \n")
print(matrix[:, 0:2])


#  replace the last row with -1's
matrix[-1] = -1
print("Updated matrix")
print(matrix)
