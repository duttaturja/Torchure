import torch
import torch.nn as nn

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w * x

print(f"Prediction before training: f(5) = {forward(5):.3f}")

# Training loop
learning_rate = 0.01

loss = nn.MSELoss()  # Mean Squared Error Loss

optimizer = torch.optim.SGD([w], lr=learning_rate)  # Stochastic Gradient Descent

for epoch in range(100):
    # Forward pass
    y_pred = forward(X)
    
    # Compute loss
    l = loss(Y, y_pred)
    
    # Compute gradients
    l.backward()  # d loss / d w
    
    # Update weights
    optimizer.step()  # Update weights using the optimizer
    optimizer.zero_grad()  # zero gradients

    if epoch % 10 == 0:  # Print every 10 epochs   
        print(f"Epoch {epoch+1}, Loss: {l:.4f}, Weight: {w:.4f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")

# Final prediction after training
"""
______________________________________________________________
Prediction before training: f(5) = 0.000
Epoch 1, Loss: 30.0000, Weight: 0.3000
Epoch 11, Loss: 1.1628, Weight: 1.6653
Epoch 21, Loss: 0.0451, Weight: 1.9341
Epoch 31, Loss: 0.0017, Weight: 1.9870
Epoch 41, Loss: 0.0001, Weight: 1.9974
Epoch 51, Loss: 0.0000, Weight: 1.9995
Epoch 61, Loss: 0.0000, Weight: 1.9999
Epoch 71, Loss: 0.0000, Weight: 2.0000  
Epoch 81, Loss: 0.0000, Weight: 2.0000  
Epoch 91, Loss: 0.0000, Weight: 2.0000  
Prediction after training: f(5) = 10.000
______________________________________________________________
"""