import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# prepare the dataset
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20.0, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

# reshape y to be a column vector
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)
 
# loss and optimizer

criterion = nn.MSELoss()

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(100):
    # forward pass
    y_predicted = model(X)

    # compute loss
    loss = criterion(y_predicted, y)

    # backward pass    
    loss.backward()

    # update weights
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')


# plot the results
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro', label='Original data')
plt.plot(X_numpy, predicted, 'b', label='Fitted line')
plt.show()