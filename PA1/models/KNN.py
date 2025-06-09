import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms

import torch
from torchvision import datasets, transforms
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Download and load the training data
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Extract X_train, y_train from trainloader
X_train = []
y_train = []
for data, target in trainloader:
    X_train.append(data)
    y_train.append(target)

X_train = torch.cat(X_train, dim=0)
y_train = torch.cat(y_train, dim=0)

# Extract X_test, y_test from testloader
X_test = []
y_test = []
for data, target in testloader:
    X_test.append(data)
    y_test.append(target)

X_test = torch.cat(X_test, dim=0)
y_test = torch.cat(y_test, dim=0)

# Convert to numpy arrays if needed
X_train_np = X_train.cpu().detach().numpy()
y_train_np = y_train.cpu().detach().numpy()
X_test_np = X_test.cpu().detach().numpy()
y_test_np = y_test.cpu().detach().numpy()
X_train = X_train_np.squeeze().reshape(60000,-1)
X_test = X_test_np.squeeze().reshape(10000,-1)
y_train = y_train_np
y_test = y_test_np

class CustomKNN(KNeighborsClassifier):
    # rewrite the predict method to handle ties
    def predict(self, X):
        distances, indices = self.kneighbors(X)
        y_pred = []
        for i in range(len(X)):
            neighbor_labels = self._y[indices[i]]
            counts = np.bincount(neighbor_labels)
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            if len(candidates) > 1:     # handle ties
                sad = {}
                for cls in candidates:
                    mask = (neighbor_labels == cls)
                    sad[cls] = np.sum(distances[i][mask])
                y_pred.append(min(sad, key=sad.get))
            else:
                y_pred.append(np.argmax(counts))
        return np.array(y_pred)

k_values = list(range(1, 11))
accuracies = []

for k in k_values:
    knn = CustomKNN(n_neighbors=k, metric='manhattan')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"K={k}, accuracy: {accuracy:.4f}")


plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, 'bo-', linewidth=2, markersize=8)
plt.title('K value versus accuracy')
plt.xlabel('K value')
plt.ylabel('accuracy')
plt.grid(True)
plt.xticks(k_values)


best_k_idx = np.argmax(accuracies)
best_k = k_values[best_k_idx]
best_acc = accuracies[best_k_idx]
plt.annotate(f'Best Accuracy: {best_acc:.4f}\nK={best_k}',
             xy=(best_k, best_acc),
             xytext=(best_k+0.5, best_acc-0.02),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.savefig("k value versus accuracy")
plt.show()

print(f"\n Best k is{best_k}, Accuracy: {best_acc:.4f}")
