import numpy as np 
np.random.seed(0)  # seed for reproducibility


x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
z = np.concatenate([x, y], axis = 0)
a = np.concatenate([x, y, z], axis = 0)

print(x)
print(y)
print(z)
print(a)
print("==============================")

grid = np.array([[1, 2, 3],
                 [4, 5, 6]])
print(grid)
print("-------------------------------")
# concatenate along the first axis
grid_c_0 = np.concatenate([grid, grid], axis = 0)
print(grid_c_0)
print("==============================")


# concatenate along the second axis 
print(grid)
print("-------------------------------")
grid_c_1 =  np.concatenate([grid, grid], axis=1)
print(grid_c_1)
print("==============================")