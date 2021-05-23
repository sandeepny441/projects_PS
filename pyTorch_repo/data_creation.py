import torch 

x = torch.empty(1)
print(x)
print("Shape: ", x.shape)
print("Number of dimensions: ", x.ndim)
print('--------------')

x = torch.empty(2,4)
print(x)
print("Shape: ", x.shape)
print("Number of dimensions: ", x.ndim)
print('--------------')

x = torch.empty(2,4,3)
print(x)
print("Shape: ", x.shape)
print("Number of dimensions: ", x.ndim)
print('--------------')

x = torch.rand(10, 2)
print(x)
print("Shape: ", x.shape)
print("Number of dimensions: ", x.ndim)
print(x.mean(), x.median(), x.std(), x.var())
print('--------------')

x = torch.zeros(2, 2)
print(x)
print('--------------')

x = torch.zeros(2, 2, dtype = torch.float16)
print(x)
print(x.size())
print("Shape: ", x.shape)
print("Number of dimensions: ", x.ndim)
print('--------------')

x = torch.zeros([1,2,3,4], dtype = torch.float16)
print(x)
print(x.size())
print("Shape: ", x.shape)
print("Number of dimensions: ", x.ndim)
print('--------------')

x = torch.ones(2, 4)
print(x)
print('--------------')

x = torch.ones([3,2], dtype = torch.float16)
print(x)
print(x.size())
print("Shape: ", x.shape)
print("Number of dimensions: ", x.ndim)
print('--------------')

x = torch.tensor([3,2], dtype = torch.float16)
print(x)
print(x.size())
print("Shape: ", x.shape)
print("Number of dimensions: ", x.ndim)
print('--------------')





