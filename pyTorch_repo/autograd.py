import torch 

# requires_grad = True will be needed inorder to calculate gradients
x = torch.randn(3, requires_grad = True)
print(x)
print('---------------------')

y = x+2
print(x)
print(y)
print('---------------------')

y = x-2
print(x)
print(y)
print('---------------------')

y = x*2
print(x)
print(y)
print('---------------------')

y = x/2
print(x)
print(y)
print('---------------------')

y = x.mean()
print(x)
print(y)
print('---------------------')

y = x.median()
print(x)
print(y)
print('---------------------')

y = x
y.backward(x)

# we need to give a vector/tensor as an argument 
# to the backward function when we apply it on a scalar value
print(x.grad)
print('---------------------')

y = x.mean()
y.backward() 

# we need not give arguments to the backward function 
# when we apply it on a scalar value
print(x.grad)
print('---------------------')

