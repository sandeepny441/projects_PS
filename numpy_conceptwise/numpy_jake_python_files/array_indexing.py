import numpy as np 
import numpy as np
np.random.seed(0)  # seed for reproducibility

x1 = np.random.randint(10, size=6)  # One-dimensional array
x2 = np.random.randint(10, size=(3, 4))  # Two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5))  # Three-dimensional array


print(x1)
print(x2)
print(x3)

print("x3 ndim: ", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size: ", x3.size)

print("dtype:", x3.dtype)

print("itemsize:", x3.itemsize, "bytes")
#itemsize refers to size of each item

print("nbytes:", x3.nbytes, "bytes")
#nbytes refers to the total memory

print("=====================================")

print(x1)
print(x1[2])

print(x2)
print(x2[1,0])

print(x3)
print(x3[1, 1, 2]) #length, breadth and width 

print("=====================================")

x3[1,1,1] = 2000.0 
#since the array is int by default, and numpy 
# is fixed type, this float value will be truncated to int 
print(x3)
print(x3.dtype)


