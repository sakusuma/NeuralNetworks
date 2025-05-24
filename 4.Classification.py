# Databricks notebook source
# MAGIC %md
# MAGIC Most real-world PyTorch applications are classification-based — e.g., image recognition, spam detection, medical diagnosis — not just regression.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ✅ You'll Learn:
# MAGIC How classification differs from regression
# MAGIC
# MAGIC How to use nn.Softmax, nn.CrossEntropyLoss
# MAGIC
# MAGIC How to one-hot encode labels (or use class indices)
# MAGIC
# MAGIC How to evaluate accuracy
# MAGIC
# MAGIC How to interpret logits and class probabilities
# MAGIC
# MAGIC
# MAGIC 🧪 We'll build a simple classifier:
# MAGIC Input: 2D features (just like before)
# MAGIC
# MAGIC Output: 3 classes (e.g., Class 0, 1, or 2)
# MAGIC
# MAGIC Goal: Learn to predict which class a sample belongs to
# MAGIC
# MAGIC 👇 Here's what the next steps could look like:
# MAGIC 📊 Create a toy dataset (2D points in 3 classes)
# MAGIC
# MAGIC 🧠 Define a classification model with output = 3 units
# MAGIC
# MAGIC 💥 Use CrossEntropyLoss (which combines LogSoftmax + NLLLoss)
# MAGIC
# MAGIC 🔁 Train the classifier using a loop similar to regression
# MAGIC
# MAGIC 📈 Measure accuracy, not just loss
# MAGIC
# MAGIC 📉 Visualize decision boundaries (optional but awesome)
# MAGIC
# MAGIC 🎯 Problem Statement
# MAGIC We’ll classify 2D points into 3 different classes. Each point belongs to either Class 0, Class 1, or Class 2.

# COMMAND ----------

# MAGIC %md
# MAGIC 🧪 Step 1: Generate a Toy Dataset

# COMMAND ----------

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Create synthetic 2D classification data with 3 classes
X, y = make_classification(n_samples=300, n_features=2, n_classes=3, 
                           n_clusters_per_class=1, n_informative=2, 
                           n_redundant=0, random_state=0)

# Convert to torch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)  # Note: CrossEntropyLoss expects class indices, not one-hot


# COMMAND ----------

# MAGIC %md
# MAGIC 👀 Visualize the Data

# COMMAND ----------

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title("3-Class Classification Problem")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 🧠 Step 2: Define a Simple Classifier
# MAGIC We’ll use a small neural net with:
# MAGIC
# MAGIC 2 input features
# MAGIC
# MAGIC 1 hidden layer with 5 neurons
# MAGIC
# MAGIC 3 output classes

# COMMAND ----------

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(2, 5)     # input → hidden
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 3)     # hidden → 3 class outputs

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # Raw logits (no softmax)


# COMMAND ----------

model = Classifier()


# COMMAND ----------

# MAGIC %md
# MAGIC ⚖️ Step 3: Define Loss and Optimizer

# COMMAND ----------

criterion = nn.CrossEntropyLoss()  # applies Softmax + NLLLoss internally
optimizer = optim.SGD(model.parameters(), lr=0.05)


# COMMAND ----------

# MAGIC %md
# MAGIC 🔁 Step 4: Train the Classifier

# COMMAND ----------

for epoch in range(1000):
    outputs = model(X_tensor)         # Forward pass
    loss = criterion(outputs, y_tensor)

    optimizer.zero_grad()             # Clear gradients
    loss.backward()                   # Backprop
    optimizer.step()                  # Update weights

    if (epoch + 1) % 100 == 0:
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_tensor).float().mean()
        print(f"Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}, Accuracy: {accuracy.item()*100:.2f}%")


# COMMAND ----------

# MAGIC %md
# MAGIC  Step 5: Test and Visualize Decision Boundary

# COMMAND ----------

# MAGIC %md
# MAGIC 🧩 Decision Boundary Plotting Code:

# COMMAND ----------

import numpy as np

# Create a mesh grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

# Flatten and convert to tensor
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

# Model predictions for each point on grid
with torch.no_grad():
    Z = model(grid)
    _, predicted = torch.max(Z, 1)

# Reshape to match mesh grid
Z = predicted.numpy().reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='viridis')
plt.title("Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 🔎 What This Shows
# MAGIC Each region in the plot is a predicted class.
# MAGIC
# MAGIC Data points are overlaid.
# MAGIC
# MAGIC You’ll see how well your model separates the 3 classes in feature space.