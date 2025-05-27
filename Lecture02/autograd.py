import torch

x = torch. randn(3, requires_grad=True)
print(x) # output: tensor([1.2709, 1.9115, 1.1068], requires_grad=True)

y = x + 2
print(y) # output: tensor([3.2709, 3.9115, 3.1068], grad_fn=<AddBackward0>) 

z = y * y * 3
print(z)  #output: tensor([32.0971, 45.8998, 28.9572], grad_fn=<MulBackward0>)

z = z.mean()
print(z) # output: tensor(35.3180, grad_fn=<MeanBackward0>)

# dz/dx
z.backward()
print(x.grad) # output: tensor([6.0000, 6.0000, 6.0000])

x = torch.randn(3, requires_grad=True)
x.requires_grad_(False)
print(x) # output: tensor([-0.1952,  0.1984, -0.1235]

y = x.detach()
print(y) # output: tensor([-0.1952,  0.1984, -0.1235])

with torch.no_grad():
    y = x + 2
    print(y) # output: tensor([1.2709, 1.9115, 1.1068])

# gradient accumulation
weight = torch.ones(4, requires_grad=True)

for epoch in range(1):
    model_output = (weight * 3).sum()
    model_output.backward()
    print(weight.grad) # output: tensor([3., 3., 3., 3.])
    weight.grad.zero_() # zero the gradient

# optimizer
optimizer = torch.optim.SGD([weight], lr=0.01)
optimizer.step() # update the weight
optimizer.zero_grad() # zero the gradient