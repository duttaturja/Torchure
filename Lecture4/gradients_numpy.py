import numpy as np

# f = w * x

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0
w = np.float32(w)  # Ensure w is a float32 for consistency

# model prediction
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y) ** 2).mean()

# gradient 
# MSE = 1/N * (w*x - y)**2
# d3/dw = 2/N * (w*x - y) * x
def gradient(x, y, y_predicted):
    return np.dot(2 * x, (y_predicted - y)).mean()

print(f"Prediction before training: f(5) = {forward(5):.3f}")

# Training loop
learning_rate = 0.01

for epoch in range(20):
    # Forward pass
    y_pred = forward(X)
    
    # Compute loss
    l = loss(Y, y_pred)
    
    # Compute gradients
    dw = gradient(X, Y, y_pred)
    
    # Update weights
    w= np.float32(w)  # Ensure w is a float32 for consistency
    w -= learning_rate * dw
    
    
    print(f"Epoch {epoch+1}, Loss: {l:.4f}, Weight: {w:.4f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")


# Final prediction after training
"""
______________________________________________________________

Prediction before training: f(5) = 0.000
PS C:\Users\Turja Dutta\OneDrive\Desktop\New folder\Torchure> uv run Lecture4/gradients_numpy.py
Prediction before training: f(5) = 0.000
Epoch 1, Loss: 30.0000, Weight: 1.2000
Epoch 2, Loss: 4.8000, Weight: 1.6800 
Epoch 3, Loss: 0.7680, Weight: 1.8720 
Epoch 4, Loss: 0.1229, Weight: 1.9488 
Epoch 5, Loss: 0.0197, Weight: 1.9795 
Epoch 6, Loss: 0.0031, Weight: 1.9918 
Epoch 7, Loss: 0.0005, Weight: 1.9967 
Epoch 8, Loss: 0.0001, Weight: 1.9987 
Epoch 9, Loss: 0.0000, Weight: 1.9995 
Epoch 10, Loss: 0.0000, Weight: 1.9998
Epoch 11, Loss: 0.0000, Weight: 1.9999
Epoch 12, Loss: 0.0000, Weight: 2.0000
Epoch 13, Loss: 0.0000, Weight: 2.0000
Epoch 14, Loss: 0.0000, Weight: 2.0000
Epoch 15, Loss: 0.0000, Weight: 2.0000
Epoch 16, Loss: 0.0000, Weight: 2.0000
Epoch 17, Loss: 0.0000, Weight: 2.0000
Epoch 18, Loss: 0.0000, Weight: 2.0000
Epoch 19, Loss: 0.0000, Weight: 2.0000
Epoch 20, Loss: 0.0000, Weight: 2.0000
Prediction after training: f(5) = 10.000
______________________________________________________________
"""