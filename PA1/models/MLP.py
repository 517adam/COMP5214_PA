import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Download and prepare the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# check available GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class MLP(nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_input, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_output)
        )

    def forward(self, x):
        return self.net(x)



def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_model(model, train_loader, test_loader, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        accuracy = evaluate(model, test_loader)
        test_accuracies.append(accuracy)

        if (epoch + 1) % 5 == 0: 
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

    return train_losses, test_accuracies

batch_size = 64
epochs = 20
train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True  
)

test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True
)
# model = MLP(input_size, hidden_size, output_size).to(device)
# losses, accuracies = train_model(model, train_loader, test_loader, epochs)
neurons = [4, 8, 16, 32, 64, 128, 256]
accuracies = []
for n in neurons:
    model = MLP(784, n, 10).to(device)
    losses, accuracy = train_model(model, train_loader, test_loader, epochs)
    accuracies.append(accuracy[-1])
    print(f"Neurons: {n}, Accuracy: {accuracy[-1]:.2f}%")
plt.figure(figsize=(10, 6))
plt.plot(neurons, accuracies, 'o-')
plt.xscale('log')
plt.xticks(neurons, labels=neurons)
plt.xlabel('Number of Neurons per Hidden Layer')
plt.ylabel('Test Accuracy (%)')
plt.title('MLP Accuracy vs. Hidden Layer Neurons on MNIST')
plt.grid(True)
plt.savefig("MLP Num of Neurons versus accuracy")
plt.show()