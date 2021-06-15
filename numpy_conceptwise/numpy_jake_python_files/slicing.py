import numpy as np 
import numpy as np
np.random.seed(0)  # seed for reproducibility

x1 = np.random.randint(10, size=6)  # One-dimensional array
x2 = np.random.randint(10, size=(3, 4))  # Two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5))  # Three-dimensional array


print(x1)
print(x1[0:4])
print("==================================")


print(x2)
print(x2[0:4, :2])
print("==================================")


print(x3)
print(x3[0:2, :1, :1])
print("==================================")


# ONE-D slicing 
x = np.arange(10)
print(x)
print(x[:5])  # first five elements)
print(x[5:])
print(x[4:7])
print(x[::2])
print(x[1::2])
print(x[5::-2])
print("==================================")

print(x2)
print(x2[:2, :3])
print(x2[:3, ::2])
print(x2[::-1, ::-1])
print("==================================")


print(x2[:, 0]) 
print(x2[0, :]) 
print(x2[0])
print("==================================") 

# Reshaping
grid = np.arange(1, 10).reshape((3, 3))
print(grid)



