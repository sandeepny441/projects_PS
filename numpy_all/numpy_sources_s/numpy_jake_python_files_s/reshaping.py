import numpy as np 
np.random.seed(0)  # seed for reproducibility
grid = np.arange(1, 10)
print(grid)

grid = np.arange(1, 10).reshape((3, 3))
print(grid)

#reshape
#newaxis 

x = np.array([1, 2, 3, 4, 5, 6])
print(x)
print(x.shape)
print("------------------------")
# row vector via reshape

x = x.reshape((2,3))
print(x)
print(x.shape)
print("------------------------")

# row vector via newaxis
# np.newaxis converts one dimensional arrays to two dimensional arrays
x = np.array([1, 2, 3, 4, 5, 2,2,2,26])
print(x)
y = x[np.newaxis, :]
print(x.shape)
print(y.shape)
print("------------------------")

x = np.array([1, 2, 3, 4, 5, 2,2,2,26])
print(x)
y = x[:np.newaxis]
print(x.shape)
print(y.shape)
print("------------------------")