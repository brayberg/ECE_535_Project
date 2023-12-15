import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn


#import models as YourModel  # Import your model class or module
#__all__ = ["cnn"]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        #output = F.log_softmax(x, dim=1)
        return x

def cnn(args):
    dataset = args.dataset

    if "cifar" in dataset or dataset == "mnist" or dataset == "fmnist":
        return Net()
    else:
        raise NotImplementedError(f"not supported yet.")
        
# Define the transformations for the test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-10 normalization
])

# Load the CIFAR-10 test dataset
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Initialize your model (make sure it has the same architecture as the saved model)
model = Net()

# Load the pre-trained model state_dict
model_path = 'fedavg.pth'
checkpoint = torch.load(model_path)

# Set the model to evaluation mode
model.eval()

# Define variables to keep track of accuracy
correct = 0
total = 0

# Evaluate the model on the test dataset
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate the accuracy
print(correct,total)
accuracy = correct / total
print(f'Test Accuracy: {100 * accuracy:.2f}%')
