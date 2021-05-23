import torch 

# Arithmetic operations
x = torch.rand([2,4])
y = torch.rand([2,4])

print(x)
print(y)
print('-------------------')

x.add_(y)
print(x)
z = torch.add(x, y)
print(z)
print('-------------------')

x.sub_(y)
print(x)
z = torch.sub(x, y)
print(z)
print('-------------------')

x.mul_(y)
print(x)
z = torch.mul(x, y)
print(z)
print('-------------------')

#Slicing
x = torch.tensor([(2,13), (1,10)])
print(x)
print(x.size())
# print(x.view(2,2))
print(x.shape)
print(x.ndim)

y = x.view(size =(1,1))
print(y)

