import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# Forward pass and compute loss
y_hat = w * x
loss = (y_hat - y) ** 2

print(f"Loss: {loss.item()}")


# Backward pass
loss.backward()
print(f"Gradient: {w.grad.item()}")

# Update weights
## next forward and backward pass
learning_rate = 0.1

with torch.no_grad():
    w -= learning_rate * w.grad
    print(f"Updated weight: {w.item()}")
    w.grad.zero_()  # Reset gradients for the next iteration

# Forward pass again
y_hat = w * x
# Compute new loss
loss = (y_hat - y) ** 2
print(f"New Loss: {loss.item()}")
# Backward pass again
loss.backward()
print(f"New Gradient: {w.grad.item()}")