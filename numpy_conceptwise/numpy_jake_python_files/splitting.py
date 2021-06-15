import numpy as np 
np.random.seed(0)  # seed for reproducibility

print('----------------------------------')
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [2, 5])
print(x1, x2, x3)
print("====================================")

grid = np.arange(16).reshape((4, 4))
print(grid)
print('----------------------------------')

upper, lower = np.vsplit(grid, [2])
print(upper)
print(lower)
print('----------------------------------')

upper, lower = np.vsplit(grid, [3])
print(upper)
print(lower)
print("====================================")
left, right = np.hsplit(grid, [2])
print(left)
print(right)
print('----------------------------------')
left, right = np.hsplit(grid, [3])
print(left)
print(right)
print("====================================")

#splitting in the d-axis -- depth axis
grid_new = np.array([[9, 8, 7],
                 [6, 5, 4]])
grid_new_3 = np.dstack([grid_new, grid_new])
print(grid_new_3)
print('----------------------------------')
grid_new_3 = np.dsplit(grid_new_3, [2])
print(grid_new_3)
print("====================================")
