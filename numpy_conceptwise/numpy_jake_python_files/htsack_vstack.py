import numpy as np 
np.random.seed(0)  # seed for reproducibility

print('----------------------------------')
x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])

# vertically stack the arrays
grid_new = np.vstack([x, grid])

print(x)
print(grid)
print(grid_new)

print('----------------------------------')
# vertically stack the arrays
y = np.array([[10000],[2000]])
grid_new = np.hstack([y, grid])

print(x)
print(grid)
print(grid_new)
print("====================================")

#stack in the d-axis -- depth axis
grid_new_3 = np.dstack([grid_new, grid_new])
print(grid_new_3)
print("====================================")