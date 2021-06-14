import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt 
import array
import sys 

np.random.seed(10)
this_array = np.random.rand(10, 2)
print(this_array)
print(this_array.mean())
print("---------------------------------")

this_array = np.random.random((10, 2))
print(this_array)
print(this_array.mean())
print("---------------------------------")

this_array = np.random.normal(1, 0.2, (10, 3))
print(this_array)
print("---------------------------------")

this_array = np.random.randint((10, 20))
print(this_array)
print("---------------------------------")

this_array = np.random.randint(10, 20, (13,3))
print(this_array)
print("---------------------------------")


# new in latest numpy
# rng = np.random.default_rng() 
# a = rng.random(size=10)
# print(a)

plt.plot(this_array)
plt.show()