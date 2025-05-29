import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__ (self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
model = Model(n_input_features=6)
# train your model...
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for param in model.parameters():
    print(param)

FILE = "./Model/test.pth"
torch.save(model, FILE)  # Save the entire mode

torch.save(model.state_dict(), FILE)

model = torch.load(FILE)

model.eval()  # Set the model to evaluation mode

loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()  # Set the loaded model to evaluation mode

for param in loaded_model.parameters():
    print(param)

print(model.state_dict())

checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict(),
}

torch.save(checkpoint, "./Dataset/checkpoint.pth")


# Load the checkpoint
loaded_checkpoint = torch.load("./Dataset/checkpoint.pth")
epoch = loaded_checkpoint["epoch"]

model = model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

model.load_state_dict(loaded_checkpoint["model_state"])
optimizer.load_state_dict(loaded_checkpoint["optim_state"])

print(optimizer.state_dict())

# Saving and Loading Models in PyTorch
PATH = "./Dataset/test.pth"
# Save on GPU, Load on CPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

device = torch.device('cpu')
model = Model(6)
model.load_state_dict(torch.load(PATH, map_location=device))

# Save on GPU, Load on GPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

model = Model(6)
model.load_state_dict(torch.load(PATH))
model.to(device)

# Note: Be sure to use the .to(torch.device('cuda')) function 
# on all model inputs, too!

# Save on CPU, Load on GPU
torch.save(model.state_dict(), PATH)

device = torch.device("cuda")
model = Model(6)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)