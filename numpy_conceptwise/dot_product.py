import numpy as np 
import sys 

this_list = [1,2,3,4]

this_array = np.array([10, 20, 30, 40])

#multiplication
this_array_1 = this_array * 2 
print(this_array_1)

#Vector_addition
this_array_2  = this_array + this_array_1
print(this_array_2)

#Broadcasting
this_array_3  = this_array + np.array(1000)
print(this_array_3)

#Memory Size
print(sys.getsizeof(this_array))
#112 bytes

#squared
this_array_4 = this_array**2
print(this_array_4)

#squared
this_array_4 = np.sqrt(this_array)
print(this_array_4)

#logging 
this_array_4 = np.log(this_array)
print(this_array_4)










