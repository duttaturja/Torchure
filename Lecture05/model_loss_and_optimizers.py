import torch
import torch.nn as nn

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([[5]], dtype=torch.float32)  # Test input for prediction
n_samples, n_features = X.shape  # Get the number of samples and features
print(f"Number of samples: {n_samples}, Number of features: {n_features}")

# Model prediction
input_size = n_features  # Number of features in the input
output_size = 1  # Single output for linear regression
model = nn.Linear(input_size, output_size)  # Simple linear model with one input and one output

# wrapper
class LineaRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LineaRegression, self).__init__()
        # define layers
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
model = LineaRegression(input_size, output_size)
print(f"Prediction before training: f(5) = {model(X_test).item():.3f}")

# Training loop
learning_rate = 0.01

loss = nn.MSELoss()  # Mean Squared Error Loss

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # Stochastic Gradient Descent

for epoch in range(100):
    # Forward pass
    y_pred = model(X)
    
    # Compute loss
    l = loss(Y, y_pred)
    
    # Compute gradients
    l.backward()  # d loss / d w
    
    # Update weights
    optimizer.step()  # Update weights using the optimizer
    optimizer.zero_grad()  # zero gradients

    if epoch % 10 == 0:  # Print every 10 epochs
        w, b = model.parameters()   
        print(f"Epoch {epoch+1}, Loss: {l:.4f}, Weight: {w[0][0].item():.4f}")

print(f"Prediction after training: f(5) = {model(X_test).item():.3f}")

# Final prediction after training
"""
______________________________________________________________
Number of samples: 4, Number of features: 1       
Prediction before training: f(5) = 2.237
Epoch 1, Loss: 18.6022, Weight: 0.7166
Epoch 11, Loss: 0.4965, Weight: 1.7071
Epoch 21, Loss: 0.0271, Weight: 1.8690
Epoch 41, Loss: 0.0130, Weight: 1.9045
Epoch 51, Loss: 0.0123, Weight: 1.9079
Epoch 61, Loss: 0.0116, Weight: 1.9108
Epoch 71, Loss: 0.0109, Weight: 1.9134
Epoch 81, Loss: 0.0103, Weight: 1.9160
Epoch 91, Loss: 0.0097, Weight: 1.9185
Prediction after training: f(5) = 9.836
______________________________________________________________
After adding a wrapper class, the model is more structured and can be easily extended or modified. The training process remains the same, but now the model can be reused or adapted for different tasks without changing the core logic.
_______________________________________________________________
Number of samples: 4, Number of features: 1
Prediction before training: f(5) = -3.495
Epoch 1, Loss: 54.8905, Weight: -0.2839
Epoch 11, Loss: 1.5128, Weight: 1.4190
Epoch 21, Loss: 0.1264, Weight: 1.6992
Epoch 31, Loss: 0.0854, Weight: 1.7504
Epoch 41, Loss: 0.0796, Weight: 1.7646
Epoch 51, Loss: 0.0749, Weight: 1.7727
Epoch 61, Loss: 0.0706, Weight: 1.7795
Epoch 71, Loss: 0.0665, Weight: 1.7861
Epoch 81, Loss: 0.0626, Weight: 1.7924
Epoch 91, Loss: 0.0589, Weight: 1.7985
Prediction after training: f(5) = 9.596
______________________________________________________________
"""