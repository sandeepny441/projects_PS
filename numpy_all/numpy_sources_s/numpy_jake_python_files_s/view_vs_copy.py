import numpy as np 
np.random.seed(0)  # seed for reproducibility

x1 = np.random.randint(10, size=6)  # One-dimensional array
x2 = np.random.randint(10, size=(3, 4))  # Two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5))  # Three-dimensional array

# In Numpy, array slicing returns views not copies 
# In Python, list slicing returns copies, not views

print(x1)
print("==================================") 
x1_modified = x1[:3]
print(x1_modified)
print("==================================") 
x1_modified[0] = 10000
print(x1_modified)

print(x1) 
print("==================================") 
# x changed since x1_modified is changed, which means x1_modified is not a copy
# but a view, which means we are having the same pointer to the actual data

# if we want to keep the original array intact, we can use .copy() method 
x1_sub_copy = x1[:2].copy()
print(x1_sub_copy)

x1_sub_copy[0] = 4200
print(x1_sub_copy)

print(x1)
print("==================================") 