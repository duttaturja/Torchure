import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print("Softmax outputs:", outputs)

x = torch.tensor([2.0, 1.0, 0.1])
softmax = nn.Softmax(dim=0)
outputs = softmax(x)
print("Softmax outputs (PyTorch):", outputs)

loss = nn.CrossEntropyLoss()

Y = torch.tensor([2, 0, 1])  # nsample x 1 = 3 x 1
# nsample x nclasses = 1 x 3
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
Y_pred_bad = torch.tensor([[2.0, 1.0, 2.1], [0.1, 1.0, 0.1], [0.1, 3.0, 0.1]])


l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print("CrossEntropy Loss (good prediction):", l1.item())
print("CrossEntropy Loss (bad prediction):", l2.item())

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)

print("Predictions (good):", predictions1.tolist())
print("Predictions (bad):", predictions2.tolist())
