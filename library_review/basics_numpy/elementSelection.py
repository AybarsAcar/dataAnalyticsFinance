import numpy as np

matrix = np.random.randint(1, 10, (5, 5))

# only elements greater than 3
# and returns an array
matrix_2 = matrix[matrix > 3]

# return the even numbers in an array
matrix_3 = matrix[matrix % 2 == 0]


# replace the negative elements with 0 and odd elements wiht 25
x = np.random.randint(-50, 50, (5, 5))
print("Before: ")
print(x)
print()

x[x < 0] = 0
x[x % 2 != 0] = 25

print("After: ")
print(x)
