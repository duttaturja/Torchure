import torch

# f = w * x

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y) ** 2).mean()

print(f"Prediction before training: f(5) = {forward(5):.3f}")

# Training loop
learning_rate = 0.01

for epoch in range(100):
    # Forward pass
    y_pred = forward(X)
    
    # Compute loss
    l = loss(Y, y_pred)
    
    # Compute gradients
    l.backward()  # d loss / d w
    
    # Update weights
    with torch.no_grad():
        w -= learning_rate * w.grad

        # zero gradients
        w.grad.zero_()  # Reset gradients for the next iteration
    
    if epoch % 10 == 0:  # Print every 10 epochs   
        print(f"Epoch {epoch+1}, Loss: {l:.4f}, Weight: {w:.4f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")

# Final prediction after training
"""
______________________________________________________________

Prediction before training: f(5) = 0.000            
Epoch 1, Loss: 30.0000, Weight: 0.3000
Epoch 2, Loss: 21.6750, Weight: 0.5550
Epoch 4, Loss: 11.3145, Weight: 0.9560
Epoch 5, Loss: 8.1747, Weight: 1.1126
Epoch 6, Loss: 5.9062, Weight: 1.2457
Epoch 7, Loss: 4.2673, Weight: 1.3588
Epoch 8, Loss: 3.0831, Weight: 1.4550
Epoch 9, Loss: 2.2275, Weight: 1.5368
Epoch 10, Loss: 1.6094, Weight: 1.6063
Epoch 11, Loss: 1.1628, Weight: 1.6653
Epoch 12, Loss: 0.8401, Weight: 1.7155
Epoch 13, Loss: 0.6070, Weight: 1.7582
Epoch 14, Loss: 0.4385, Weight: 1.7945
Epoch 15, Loss: 0.3168, Weight: 1.8253
Epoch 16, Loss: 0.2289, Weight: 1.8515
Epoch 17, Loss: 0.1654, Weight: 1.8738
Epoch 18, Loss: 0.1195, Weight: 1.8927
Epoch 19, Loss: 0.0863, Weight: 1.9088
Epoch 20, Loss: 0.0624, Weight: 1.9225
Prediction after training: f(5) = 9.612
______________________________________________________________
the value of w is 2.0000, which is the expected value for the linear relationship y = 2x. But here the value of w is 1.9225, which is close to the expected value. The difference is due to the learning rate and the number of epochs. The model has learned to approximate the relationship between X and Y, and the loss has decreased significantly over the epochs.

This is after 100 epochs, the model has learned to approximate the relationship between X and Y, and the loss has decreased significantly over the epochs.
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