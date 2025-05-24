# Databricks notebook source
# MAGIC %md
# MAGIC Step 1: Work with Real-World Datasets — MNIST
# MAGIC MNIST is a classic beginner-friendly dataset of handwritten digits (0-9). It’s a great dataset to learn image classification with PyTorch.

# COMMAND ----------

# MAGIC %md
# MAGIC | Concept         | Explanation                                                               |
# MAGIC | --------------- | ------------------------------------------------------------------------- |
# MAGIC | `batch_size=64` | You don’t feed the model one image at a time, but **64 images per step**  |
# MAGIC | `shuffle=True`  | Randomizes order of images each epoch, avoids learning bias from sequence |
# MAGIC | `DataLoader`    | Efficiently yields `images, labels` pairs for each batch                  |
# MAGIC

# COMMAND ----------

# Code to download and load MNIST:

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformation: convert images to tensors and normalize
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors (values between 0 and 1)
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std of MNIST
])

# Download and load the training dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Download and load the test dataset
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Create data loaders for batching and shuffling
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")


# COMMAND ----------

# MAGIC %md
# MAGIC Step 1.2: Visualize a Batch of MNIST Images

# COMMAND ----------

import matplotlib.pyplot as plt

# Get one batch of training data
images, labels = next(iter(train_loader))

# Plot first 8 images and labels
fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for i in range(8):
    axes[i].imshow(images[i].squeeze(), cmap='gray')
    axes[i].set_title(f"Label: {labels[i].item()}")
    axes[i].axis('off')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ✅ Step 2: Build a Neural Network for MNIST
# MAGIC
# MAGIC Each MNIST image is:
# MAGIC
# MAGIC Size 28x28 pixels (grayscale)
# MAGIC
# MAGIC Flattened to a 784-dimensional vector (28×28)
# MAGIC
# MAGIC Belongs to one of 10 classes (digits 0 through 9)

# COMMAND ----------

# MAGIC %md
# MAGIC Step 2.1: Define the Neural Network
# MAGIC
# MAGIC Let’s start simple: one hidden layer with 128 neurons.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Why this is needed:
# MAGIC Your input image is shaped [batch_size, 1, 28, 28] (channels = 1 for grayscale)
# MAGIC
# MAGIC Fully connected layers like nn.Linear() expect flat vectors
# MAGIC
# MAGIC So we reshape it from 4D → 2D:
# MAGIC
# MAGIC | Shape         | Meaning                                |
# MAGIC | ------------- | -------------------------------------- |
# MAGIC | `[-1, 28*28]` | “-1” tells PyTorch to infer batch size |
# MAGIC | `28*28 = 784` | flatten image into vector              |
# MAGIC

# COMMAND ----------

import torch.nn as nn
import torch.nn.functional as F

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # input layer
        self.fc2 = nn.Linear(128, 10)       # output layer for 10 digits

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # flatten the image
        x = F.relu(self.fc1(x))
        x = self.fc2(x)          # no softmax here — CrossEntropyLoss handles that
        return x

model = MNISTClassifier()
print(model)


# COMMAND ----------

# MAGIC %md
# MAGIC Step 2.2: Define Loss Function and Optimizer

# COMMAND ----------

import torch.optim as optim

criterion = nn.CrossEntropyLoss()  # expects raw logits and class indices
optimizer = optim.SGD(model.parameters(), lr=0.01)


# COMMAND ----------

# MAGIC %md
# MAGIC Step 2.3: Train the Model

# COMMAND ----------

for epoch in range(5):  # 5 epochs for now
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")


# COMMAND ----------

# MAGIC %md
# MAGIC Step 2.4: Evaluate on Test Set

# COMMAND ----------

model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

print(f"Test Accuracy: {100 * test_correct / test_total:.2f}%")


# COMMAND ----------

images, labels = next(iter(train_loader))
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

# Show one image
import matplotlib.pyplot as plt
plt.imshow(images[0].squeeze(), cmap='gray')
plt.title(f"Label: {labels[0].item()}")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ✅ Step 4: Build Deeper Neural Networks with Regularization
# MAGIC
# MAGIC We'll cover:
# MAGIC
# MAGIC Adding more hidden layers
# MAGIC
# MAGIC Using better activation functions
# MAGIC
# MAGIC Introducing Dropout (regularization)
# MAGIC
# MAGIC Switching to a better optimizer like Adam

# COMMAND ----------

# MAGIC %md
# MAGIC 🧠 Step 4.1: Build a Deeper Network
# MAGIC
# MAGIC From  784 → 128 → 10 to 784 → 256 → 128 → 64 → 10
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC We'll use ReLU and Dropout between layers.
# MAGIC
# MAGIC 🧱 Model Definition:

# COMMAND ----------

class DeepMNISTClassifier(nn.Module):
    def __init__(self):
        super(DeepMNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.1)  # lower dropout

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

model = DeepMNISTClassifier()
print(model)


# COMMAND ----------

# MAGIC %md
# MAGIC ⚙️ Step 4.2: Switch to Adam Optimizer
# MAGIC
# MAGIC Adam adapts learning rates automatically per parameter
# MAGIC
# MAGIC Often leads to faster and more stable training

# COMMAND ----------

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# COMMAND ----------

# MAGIC %md
# MAGIC 🔁 Step 4.3: Train the Deeper Model
# MAGIC
# MAGIC Reuse your earlier training loop — no changes needed except replacing the model and optimizer.

# COMMAND ----------

model.train()  # 🔁 This enables Dropout and BatchNorm in training mode

for epoch in range(5):  # 5 epochs for now
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")


# COMMAND ----------

# MAGIC %md
# MAGIC ✅ Step 5: Model Evaluation on Test Data
# MAGIC
# MAGIC Now that your model has learned from training data, let’s test how well it generalizes to unseen data (your test_loader).
# MAGIC
# MAGIC 🔍 Why Evaluate on Test Set?
# MAGIC Training accuracy ≠ Real-world performance
# MAGIC
# MAGIC We check overfitting and generalization
# MAGIC
# MAGIC Dropout and batch norm are disabled using model.eval()

# COMMAND ----------

# MAGIC %md
# MAGIC 🧪 Evaluation Code:
# MAGIC
# MAGIC 📌 Why it's important:
# MAGIC During training, we want:
# MAGIC
# MAGIC Dropout: ON (randomly drop some neurons)
# MAGIC
# MAGIC BatchNorm: uses batch stats
# MAGIC
# MAGIC During evaluation, we want:
# MAGIC
# MAGIC Dropout: OFF
# MAGIC
# MAGIC BatchNorm: uses learned population stats

# COMMAND ----------

model.eval()  # Turn off dropout and batchnorm

test_loss = 0
correct = 0
total = 0

with torch.no_grad():  # no gradients needed for evaluation
    for images, labels in test_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")


# COMMAND ----------

# MAGIC %md
# MAGIC | Scenario                       | Interpretation                      |
# MAGIC | ------------------------------ | ----------------------------------- |
# MAGIC | Test accuracy ≈ Train accuracy | Good generalization                 |
# MAGIC | Test accuracy ≪ Train accuracy | Overfitting                         |
# MAGIC | Test accuracy ≫ Train accuracy | Possible data leakage or randomness |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC 📈 Next Logical Step: Visualize Predictions
# MAGIC Let’s make things more intuitive by visualizing:
# MAGIC
# MAGIC A batch of correct predictions
# MAGIC
# MAGIC A few wrong predictions
# MAGIC
# MAGIC Optional: Confusion matrix to see class-wise performance
# MAGIC
# MAGIC ✅ Code to Visualize Predictions
# MAGIC Add this after evaluation:
# MAGIC

# COMMAND ----------

import matplotlib.pyplot as plt

# Get one batch from test loader
images, labels = next(iter(test_loader))
outputs = model(images)
_, preds = torch.max(outputs, 1)

# Plot 10 test images with predictions
fig = plt.figure(figsize=(12, 6))
for idx in range(10):
    ax = fig.add_subplot(2, 5, idx+1)
    ax.imshow(images[idx].squeeze(), cmap='gray')
    ax.set_title(f"Pred: {preds[idx].item()}\nActual: {labels[idx].item()}")
    ax.axis('off')
plt.tight_layout()
plt.show()
