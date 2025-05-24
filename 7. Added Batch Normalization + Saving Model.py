# Databricks notebook source
# MAGIC %md
# MAGIC What Is Batch Normalization?
# MAGIC Batch Normalization is a technique to improve the training of deep neural networks by normalizing the input of each layer so that the distribution of inputs remains consistent.
# MAGIC
# MAGIC It was introduced by Ioffe and Szegedy in 2015 to solve the problem of internal covariate shift — the change in the distribution of layer inputs during training.

# COMMAND ----------

# MAGIC %md
# MAGIC 🔬 How It Works (Step-by-Step)
# MAGIC For each mini-batch and for each feature in the batch:
# MAGIC
# MAGIC 1. Compute Mean and Variance
# MAGIC Let 
# MAGIC 𝑥
# MAGIC 1
# MAGIC ,
# MAGIC 𝑥
# MAGIC 2
# MAGIC ,
# MAGIC .
# MAGIC .
# MAGIC .
# MAGIC ,
# MAGIC 𝑥
# MAGIC 𝑚
# MAGIC x 
# MAGIC 1
# MAGIC ​
# MAGIC  ,x 
# MAGIC 2
# MAGIC ​
# MAGIC  ,...,x 
# MAGIC m
# MAGIC ​
# MAGIC   be inputs for a mini-batch of size 
# MAGIC 𝑚
# MAGIC m:
# MAGIC
# MAGIC 𝜇
# MAGIC =
# MAGIC 1
# MAGIC 𝑚
# MAGIC ∑
# MAGIC 𝑖
# MAGIC =
# MAGIC 1
# MAGIC 𝑚
# MAGIC 𝑥
# MAGIC 𝑖
# MAGIC (batch mean)
# MAGIC μ= 
# MAGIC m
# MAGIC 1
# MAGIC ​
# MAGIC   
# MAGIC i=1
# MAGIC ∑
# MAGIC m
# MAGIC ​
# MAGIC  x 
# MAGIC i
# MAGIC ​
# MAGIC  (batch mean)
# MAGIC 𝜎
# MAGIC 2
# MAGIC =
# MAGIC 1
# MAGIC 𝑚
# MAGIC ∑
# MAGIC 𝑖
# MAGIC =
# MAGIC 1
# MAGIC 𝑚
# MAGIC (
# MAGIC 𝑥
# MAGIC 𝑖
# MAGIC −
# MAGIC 𝜇
# MAGIC )
# MAGIC 2
# MAGIC (batch variance)
# MAGIC σ 
# MAGIC 2
# MAGIC  = 
# MAGIC m
# MAGIC 1
# MAGIC ​
# MAGIC   
# MAGIC i=1
# MAGIC ∑
# MAGIC m
# MAGIC ​
# MAGIC  (x 
# MAGIC i
# MAGIC ​
# MAGIC  −μ) 
# MAGIC 2
# MAGIC  (batch variance)
# MAGIC 2. Normalize
# MAGIC Center and scale the input:
# MAGIC
# MAGIC 𝑥
# MAGIC ^
# MAGIC 𝑖
# MAGIC =
# MAGIC 𝑥
# MAGIC 𝑖
# MAGIC −
# MAGIC 𝜇
# MAGIC 𝜎
# MAGIC 2
# MAGIC +
# MAGIC 𝜖
# MAGIC (unit variance and zero mean)
# MAGIC x
# MAGIC ^
# MAGIC   
# MAGIC i
# MAGIC ​
# MAGIC  = 
# MAGIC σ 
# MAGIC 2
# MAGIC  +ϵ
# MAGIC ​
# MAGIC  
# MAGIC x 
# MAGIC i
# MAGIC ​
# MAGIC  −μ
# MAGIC ​
# MAGIC  (unit variance and zero mean)
# MAGIC 𝜖
# MAGIC ϵ is a small constant to avoid division by zero.
# MAGIC
# MAGIC 3. Scale and Shift
# MAGIC Introduce two learnable parameters: 
# MAGIC 𝛾
# MAGIC γ (scale) and 
# MAGIC 𝛽
# MAGIC β (shift):
# MAGIC
# MAGIC 𝑦
# MAGIC 𝑖
# MAGIC =
# MAGIC 𝛾
# MAGIC ⋅
# MAGIC 𝑥
# MAGIC ^
# MAGIC 𝑖
# MAGIC +
# MAGIC 𝛽
# MAGIC y 
# MAGIC i
# MAGIC ​
# MAGIC  =γ⋅ 
# MAGIC x
# MAGIC ^
# MAGIC   
# MAGIC i
# MAGIC ​
# MAGIC  +β
# MAGIC These parameters allow the network to learn the optimal range of the transformed values.
# MAGIC
# MAGIC ✅ Why It Helps

# COMMAND ----------

# MAGIC %md
# MAGIC | Benefit                              | Description                                                                    |
# MAGIC | ------------------------------------ | ------------------------------------------------------------------------------ |
# MAGIC | **Faster Convergence**               | BN reduces the need for careful weight initialization and high learning rates. |
# MAGIC | **Stabilizes Training**              | Reduces oscillations by keeping activations within a stable range.             |
# MAGIC | **Reduces Internal Covariate Shift** | Keeps the distribution of inputs stable across layers.                         |
# MAGIC | **Acts Like Regularization**         | Slight noise from mini-batch statistics has a similar effect as Dropout.       |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC 🧱 Modified Model with Batch Normalization
# MAGIC
# MAGIC We'll place BatchNorm1d after the linear layer and before the activation.

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

# Define model with Batch Normalization and LeakyReLU
class BatchNormModel(nn.Module):
    def __init__(self):
        super(BatchNormModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.fc3 = nn.Linear(64, 10)
        
        self.act = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.act(self.bn1(self.fc1(x)))
        x = self.act(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

# Initialize model, loss, optimizer
model = BatchNormModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
# MAGIC | Metric                | Value      | Comments                        |
# MAGIC | --------------------- | ---------- | ------------------------------- |
# MAGIC | **Baseline Accuracy** | 97.76%     | Using ReLU                      |
# MAGIC | **LeakyReLU Only**    | 95.92%     | Slight dip                      |
# MAGIC | **LeakyReLU + BN**    | **97.68%** | On par with baseline or better  |
# MAGIC | **Train Accuracy**    | 98.53%     | Best so far                     |
# MAGIC | **Test Loss**         | 0.7162     | Lower than LeakyReLU-only model |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC 🧠 Types of Normalization in Deep Learning

# COMMAND ----------

# MAGIC %md
# MAGIC | Normalization                   | Use Case                                                | Normalizes Over                    | Common In            |
# MAGIC | ------------------------------- | ------------------------------------------------------- | ---------------------------------- | -------------------- |
# MAGIC | **Batch Normalization** (BN)    | Stabilizes training by normalizing across batch samples | Batch dimension                    | CNNs, MLPs           |
# MAGIC | **Layer Normalization** (LN)    | Good for variable-length inputs                         | All features in each sample        | Transformers, NLP    |
# MAGIC | **Instance Normalization** (IN) | Effective for style transfer and image generation       | Each individual sample/channel     | GANs, style transfer |
# MAGIC | **Group Normalization** (GN)    | Alternative to BN when batch size is small              | Groups of channels                 | Small-batch settings |
# MAGIC | **Weight Normalization**        | Reparameterizes weights for better optimization         | Directly normalizes weights        | Optimization tricks  |
# MAGIC | **Spectral Normalization**      | Stabilizes GANs by bounding weight matrix norms         | Singular values of weight matrices | GANs                 |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC 🧾 What Does .pt Stand For?
# MAGIC
# MAGIC .pt = PyTorch
# MAGIC
# MAGIC It's the standard file extension for saved PyTorch model weights or full model objects.
# MAGIC
# MAGIC You might also see .pth — both are valid and used interchangeably (pt = PyTorch, pth = PyTorch checkpoint).

# COMMAND ----------

# MAGIC %md
# MAGIC 💾 Saving a PyTorch Model

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Only the model parameters (recommended):
# MAGIC
# MAGIC Saves only the weights and biases.
# MAGIC
# MAGIC Lightweight and flexible.
# MAGIC
# MAGIC Requires re-creating the model class when loading.

# COMMAND ----------

torch.save(model.state_dict(), "model_weights.pt")

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Entire model (structure + weights) (not recommended for long-term use):
# MAGIC
# MAGIC Saves the architecture + weights.
# MAGIC
# MAGIC Can break if code changes.
# MAGIC
# MAGIC Useful for quick experiments.

# COMMAND ----------

torch.save(model, "full_model.pt")

# COMMAND ----------

# MAGIC %md
# MAGIC ♻️ Loading a Saved Model
# MAGIC
# MAGIC Step-by-Step (Recommended Way):
# MAGIC
# MAGIC ✅ 1. Save model weights

# COMMAND ----------

torch.save(model.state_dict(), "model_weights.pt")


# COMMAND ----------

# MAGIC %md
# MAGIC ✅ 2. Recreate the model architecture

# COMMAND ----------

model = BatchNormModel()


# COMMAND ----------

# MAGIC %md
# MAGIC ✅ 3. Load weights

# COMMAND ----------

model.load_state_dict(torch.load("model_weights.pt"))
model.eval()  # Set to inference mode

# COMMAND ----------

# MAGIC %md
# MAGIC 📦 Full Example Using Your Model

# COMMAND ----------

# Save the trained model weights
torch.save(model.state_dict(), "mnist_batchnorm_model.pt")

# Later... Load it back
loaded_model = BatchNormModel()
loaded_model.load_state_dict(torch.load("mnist_batchnorm_model.pt"))
loaded_model.eval()

# Move to GPU if needed
loaded_model.to(device)
