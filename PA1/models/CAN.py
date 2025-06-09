import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
class CAN(nn.Module):
    def __init__(self, feature_channels=32):
        super(CAN, self).__init__()
        self.features = nn.Sequential(
            # dilation=1 (receptive field 3×3)
            nn.Conv2d(1, feature_channels, kernel_size=3, padding=1, dilation=1),
            nn.LeakyReLU(inplace=True),
    
            # dilation=2 (receptive field 7×7)
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(inplace=True),
    
            # 3×3 conv with dilation=4 (receptive field 15×15)
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=4, dilation=4),
            nn.LeakyReLU(inplace=True),
           
            # dilation=8 (receptive field 31×31)
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=8, dilation=8),
            nn.LeakyReLU(inplace=True),
    
            # Final conv to produce the 10 class channels
            nn.Conv2d(feature_channels, 10, kernel_size=3, padding=1, dilation=1),
            nn.LeakyReLU(inplace=True),
    
            # Average pooling layer
            nn.AvgPool2d(28)
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

# Prepare the MNIST dataset and dataloaders:
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a function to train and evaluate the model for given feature channels:
def train_and_evaluate(feature_channels, num_epochs=20):
    model = CAN(feature_channels=feature_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[Channel {feature_channels} | Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
    print('Finished Training')
    # Evaluate the model on the test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Run experiments with different numbers of feature channels:
feature_channels_list = [8,16, 32, 64, 128]
accuracies = []

for feature_channels in feature_channels_list:
    print(f'\nRunning experiment with {feature_channels} feature channels:')
    acc = train_and_evaluate(feature_channels, num_epochs=20)
    accuracies.append(acc)
    print(f'Accuracy with {feature_channels} feature channels: {acc:.2f}%')

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(feature_channels_list, accuracies, marker='o', linestyle='-')
plt.title('Test Accuracy vs. Number of Feature Channels')
plt.xlabel('Number of Feature Channels')
plt.ylabel('Test Accuracy (%)')
plt.grid(True)
plt.savefig("CAN_Accuracy with num of feature_channels")
plt.show()
