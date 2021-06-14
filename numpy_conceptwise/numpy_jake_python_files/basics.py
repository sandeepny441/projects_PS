import numpy as np  
import array
import sys 

this_list = list(range(1,101))
this_array = array.array('i', this_list)
this_numpy_array = np.array(this_list)


print(sys.getsizeof(this_list))
print(sys.getsizeof(this_array))
print(sys.getsizeof(this_numpy_array))
print("---------------------------------")

print(np.ones((3,4)))
print("---------------------------------")

print(np.zeros((3,4)))
print("---------------------------------")

print(np.empty((3,4)))
print("---------------------------------")

print(np.eye((5)))
print("---------------------------------")

print(np.full((3,5), 4))
print("---------------------------------")

print(np.arange(1, 20, 3))
print("---------------------------------")

print(np.linspace(1, 20, 6))
print("---------------------------------")
print("--------------END----------------")