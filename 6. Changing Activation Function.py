# Databricks notebook source
# MAGIC %md
# MAGIC 🔁 Experiment 1: Change Activation Function
# MAGIC Right now, you're likely using ReLU, which is common. Let's try swapping it with:
# MAGIC
# MAGIC 🔹 Option A: LeakyReLU
# MAGIC Allows a small, non-zero gradient when input < 0
# MAGIC
# MAGIC Helps avoid the “dying ReLU” problem
# MAGIC
# MAGIC 🔹 Option B: GELU
# MAGIC Smooth, differentiable
# MAGIC
# MAGIC Used in Transformer models (e.g., BERT)

# COMMAND ----------

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)



# COMMAND ----------


# Define the model with LeakyReLU
class LeakyReLUModel(nn.Module):
    def __init__(self):
        super(LeakyReLUModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.act = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss, and optimizer
model = LeakyReLUModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# COMMAND ----------


# Training loop
model.train()
for epoch in range(5):
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")


# COMMAND ----------


# Evaluation
model.eval()
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_acc = 100 * correct / total
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC 🧠 Interpretation
# MAGIC LeakyReLU didn’t outperform ReLU in this setup — but it still generalizes well.
# MAGIC
# MAGIC This teaches you that activation functions impact training behavior and convergence, but results can vary based on:
# MAGIC
# MAGIC Layer widths
# MAGIC
# MAGIC Learning rate
# MAGIC
# MAGIC Dataset
# MAGIC
# MAGIC Number of epochs