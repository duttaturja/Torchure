import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

############## TENSORBOARD ########################
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('Runs/mnist1')
###################################################

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper-parameters
input_size = 28 * 28  # 28x28 images
hidden_size = 100 # number of hidden units
num_classes = 10  # 10 classes for MNIST
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./Dataset',
                                           train=True,
                                           transform=transforms.ToTensor())

test_dataset = torchvision.datasets.MNIST(root='./Dataset',
                                          train=False,
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)

for  i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(samples[i][0], cmap='gray')
    
# plt.show()

############## TENSORBOARD ########################
img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist_images', img_grid)
#writer.close()
#sys.exit()
###################################################

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = x.view(-1, input_size)  # flatten the input
        out = self.l1(out)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
# model
model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

############## TENSORBOARD ########################
writer.add_graph(model, samples.reshape(-1, 28*28).to(device))
#writer.close()
#sys.exit()
###################################################

# train the model
running_loss = 0.0
running_correct = 0
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 100, 1, 28, 28
        # 100, 784

        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
            ############## TENSORBOARD ########################
            writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
            running_accuracy = running_correct / 100 / predicted.size(0)
            writer.add_scalar('accuracy', running_accuracy, epoch * n_total_steps + i)
            running_correct = 0
            running_loss = 0.0
            ###################################################



def main():
    class_labels = []
    class_preds = []
    
    # test the model
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            class_prob_batch = [F.softmax(outputs, dim=1) for output in outputs]

            class_preds.append(class_prob_batch)
            class_labels.append(labels)


        class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
        class_labels = torch.cat(class_labels)

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the model on the test images: {acc:.2f}%')
        ############## TENSORBOARD ########################
        classes = range(10)
        for i in classes:
            labels_i = class_labels == i
            preds_i = class_preds[:, i]
            # Convert to numpy arrays for tensorboard
            labels_i = labels_i.cpu().detach().numpy().reshape(-1).astype(np.float32)
            preds_i = preds_i.cpu().detach().numpy().reshape(-1).astype(np.float32)
            # Add to tensorboard
            writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
            writer.close()
        ###################################################


if __name__ == '__main__':
    model.to(device)
    main()

    path = './Models/ff.pth'
    
    # # save the model checkpoint
    # torch.save(model.state_dict(), 'path')
    # print('Model saved to ff.pth')