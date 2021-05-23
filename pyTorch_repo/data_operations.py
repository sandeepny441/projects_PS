import torch 

# Arithmetic operations
x = torch.rand([2,4])
y = torch.rand([2,4])

print(x)
print(y)
print('-------------------------')

#Methods with trailing undersdocre are in-place operations in pytorch
x.add_(y)
print(x)
z = torch.add(x, y)
print(z)
print('-------------------------')

x.sub_(y)
print(x)
z = torch.sub(x, y)
print(z)
print('-------------------------')

x.mul_(y)
print(x)
z = torch.mul(x, y)
print(z)
print('-------------------------')

#Slicing
x = torch.tensor([(2,13, 21), (1,23, 10)])
print(x)
print(x.size())
print(x.shape)
print(x.ndim)

print(x)
print(x[2:3, 2:3])
print('-------------------------')

x= torch.rand(3,4)
print(x)
print(x.ndim)
print(x.shape)
print(x.ndim)

print(x)
print(x[2:3, 3:4])
print('-------------------------')

#View
y = x.view(-1, 3)
print(y)
print(y.size())
print(y.shape)
print(y.ndim)
print('-------------------------')


#torch and  numpy || tensor --> numpy arrays
import numpy as np 

a = torch.ones(5)
print(a)
print(type(a))

b = a.numpy()
print(b)
print(type(b))

a.add_(1)

print(a)
print(b)

print('-------------------------')
#torch and  numpy || numpy arrays --> tensors

this_numpy_array = np.random.rand(1,2)
print(this_numpy_array)
print(this_numpy_array.shape, this_numpy_array.size, this_numpy_array.ndim)

print('-------------------------')

this_tensor_array = torch.from_numpy(this_numpy_array)
print(this_tensor_array)
print(this_tensor_array.shape, this_tensor_array.size(), this_tensor_array.ndim)


#using CPU as device
x = torch.ones(5)
y = torch.ones(5)
z = torch.zeros(5)

print(x, y, z)
print(x.device, y.device, z.device)
print('-------------------------')

a = x 
print(a)

a.add_(10)
print(a)
print(x)
print('-------------------------')


#using GPU as device
if torch.cuda.is_available():
	device = torch.device("cuda")
x = torch.ones(5, device = device)
y = torch.ones(5, device = device)
z = torch.zeros(5, device = device)

print(x, y, z)
print(x.device, y.device, z.device)
print('------------------------------')

a = x 
print(a)

a.add_(10)
print(a)
print(x)
print('------------------------------')

