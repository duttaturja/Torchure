import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# prepare the dataset
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape
# print(f'Number of samples: {n_samples}, Number of features: {n_features}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32)).view(y_train.shape[0], 1)
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32)).view(y_test.shape[0], 1)

# model
input_size = n_features

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)


    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x)) # returns value between 0 and 1
        return y_predicted
    
model = LogisticRegression(input_size)

# loss and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(100):
    # forward pass
    y_predicted = model(X_train)

    # compute loss
    loss = criterion(y_predicted, y_train)

    # backward pass    
    loss.backward()

    # update weights
    optimizer.step()

    # zero the gradients
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# evaluate the model
with torch.no_grad():
    y_test_predicted = model(X_test)
    y_test_predicted_cls = y_test_predicted.round()  # convert probabilities to 0 or 1

    acc = y_test_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    acc *= 100  # convert to percentage
    print(f'Accuracy: {acc:.4f}%')