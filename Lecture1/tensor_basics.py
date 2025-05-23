import torch
"""
# scalar
x = torch.empty(1)
print(x) # output: tensor([9.7390e+32])

# vector
x = torch.empty(2)
print(x) # output: tensor([0., 0.])

# matrix
x = torch.empty(2, 3)
print(x) #output: tensor([[9.8370e+32, 1.4672e-42, 0.0000e+00],[0.0000e+00, 0.0000e+00, 0.0000e+00]])

# 3D tensor
x = torch.empty(2, 3, 4)
print(x) #output: tensor([[[9.8370e+32, 1.4672e-42, 0.0000e+00, 0.0000e+00],

# 4D tensor
x = torch.empty(2, 3, 4, 5)
print(x) # output: tensor([[[9.8372e+32, 1.4672e-42, 0.0000e+00, 0.0000e+00],
         # [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]],

        # [[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]]])

# create a tensor with random values
x = torch.rand(5, 3)
print(x) # output: tensor([[0.6266, 0.1465, 0.2197],
                        # [0.7002, 0.6457, 0.4120],
                        # [0.5983, 0.8381, 0.9512],
                        # [0.7446, 0.8282, 0.0909],
                        # [0.4880, 0.0803, 0.1262]])

# fill with zeros
x = torch.zeros(5, 3)
print(x) # output: tensor([[0., 0., 0.],
                        # [0., 0., 0.],
                        # [0., 0., 0.],
                        # [0., 0., 0.],
                        # [0., 0., 0.]])

# fill with ones
x = torch.ones(5, 3)
print(x) # output: tensor([[1., 1., 1.],
                        # [1., 1., 1.],
                        # [1., 1., 1.],
                        # [1., 1., 1.],
                        # [1., 1., 1.]])

# fill with random integers
x = torch.randint(0, 10, (5, 3))
print(x) # output: tensor([[2, 4, 1],
                        # [5, 7, 8],
                        # [0, 3, 6],
                        # [9, 2, 1],
                        # [4, 5, 8]])

# check size
print(x.size()) # output: torch.Size([5, 3])

# check shape
print(x.shape) # output: torch.Size([5, 3])

# check data type
print(x.dtype) # output: torch.int64

# check device
print(x.device) # output: cpu

# check if GPU is available
print(torch.cuda.is_available()) # output: False

# construct from data
x = torch.tensor([5.5, 3])
print(x) # output: tensor([5.5000, 3.0000])
print(x.size()) # output: torch.Size([2]) 
"""
# optimization
x = torch.tensor([5.5, 3], requires_grad=True)

# operations
y = torch.rand(2, 2)
x = torch.rand(2, 2)
# addition
z = x+y
print(z) # output: tensor([[0.8066, 0.8606],
                        # [0.7312, 1.2324]])
z = torch.add(x, y) # also works
# subtraction
z = x-y
print(z) # output: tensor([[ 0.4762, -0.7372],
                        # [ 0.3331,  0.3696]])
z = torch.sub(x, y) # also works
# multiplication
z = x*y
print(z) # output: ([[0.6814, 0.2471],
                   # [0.0020, 0.2671]])
z = torch.mul(x, y) # also works
# division
z = x/y
print(z) # output: tensor([[1.0783e+00, 1.1972e+00],
                         # [2.1343e-03, 2.3414e+00]])
z = torch.div(x, y) # also works

# slicing
x = torch.rand(5, 3)
print(x) # output: tensor([[0.6266, 0.1465, 0.2197],
                        # [0.7002, 0.6457, 0.4120],
                        # [0.5983, 0.8381, 0.9512],
                        # [0.7446, 0.8282, 0.0909],
                        # [0.4880, 0.0803, 0.1262]])
# ll rows of column 0
print(x[:, 0]) # output: tensor([0.6266, 0.7002, 0.5983, 0.7446, 0.4880])
# all columns of row 0
print(x[0, :]) # output: tensor([0.6266, 0.1465, 0.2197])
# single element
print(x[0, 0]) # output: tensor(0.6266)
# original value
print(x[0, 0].item()) # output: 0.6266

# reshape
x = torch.rand(4, 4)
print(x) # output: tensor([[0.6266, 0.1465, 0.2197, 0.7002],
                        # [0.6457, 0.4120, 0.5983, 0.8381],
                        # [0.9512, 0.7446, 0.8282, 0.0909],
                        # [0.4880, 0.0803, 0.1262, 0.0000]])
y = x.view(16)
print(y.size()) # output: torch.Size([16])
z = x.view(-1, 8) # -1 means infer the dimension
print(z.size()) # output: torch.Size([2, 8])

# Numpy

x = torch.ones(5)
print(x) # output: tensor([1., 1., 1., 1., 1.])
y = x.numpy()
print(y) # output: [1. 1. 1. 1. 1.]

x.add_(1)
print(x) # output: tensor([2., 2., 2., 2., 2.])
print(y) # output: [2. 2. 2. 2. 2.]

import numpy as np
x = np.ones(5)
y = torch.from_numpy(x)
print(y) # output: tensor([1., 1., 1., 1., 1.])
x += 1
print(x) # output: [2. 2. 2. 2. 2.]
print(y) # output: tensor([2., 2., 2., 2., 2.])

