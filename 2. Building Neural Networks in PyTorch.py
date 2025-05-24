# Databricks notebook source
# MAGIC %md
# MAGIC Below is a simple feedforward neural network with two layers.
# MAGIC
# MAGIC Uses ReLU activation after the first layer.
# MAGIC
# MAGIC Final layer does not use an activation function, making it suitable for regression problems.

# COMMAND ----------

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the network class by inheriting the nn.Module , we gain parameter management and easy integration with pytorch utils
class SimpleNet(nn.Module):
    # defines the constructor method that initializes the network.
    def __init__(self):
        # Calls the constructor of nn.Module, ensuring that the model behaves correctly within PyTorch.
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 3)  # Input layer to hidden layer, Input is 2 neurons and output is 3 neurons
        self.fc2 = nn.Linear(3, 1)  # Hidden layer to output, Input is 3 neurons and output is 1 neuron
    # This function defines how the input data flows through the model.
    def forward(self, x):
        # self.fc1(x): Applies the first fully connected layer on the input x
        x = F.relu(self.fc1(x))    # Apply ReLU activation after fc1
        # self.fc2(x): Passes the transformed values to the final output layer.
        x = self.fc2(x)            # Output layer (no activation for regression)
        return x

# Instantiate the model
model = SimpleNet()
print(model)


# COMMAND ----------

# MAGIC %md
# MAGIC Test with Sample Input

# COMMAND ----------

# Sample input: batch of 2 samples, each with 2 features
input_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
# What happens when you call model(input)? : It calls the model’s forward() method internally, which performs the forward pass through the layers.
output = model(input_data)
print(output)


# COMMAND ----------

# MAGIC %md
# MAGIC 🧠 Step-by-Step Breakdown
# MAGIC We'll trace the execution through layers and tensors:
# MAGIC
# MAGIC ✅ Step 1: fc1 = nn.Linear(2, 3)
# MAGIC This layer computes:
# MAGIC
# MAGIC fc1
# MAGIC (
# MAGIC 𝑥
# MAGIC )
# MAGIC =
# MAGIC 𝑥
# MAGIC 𝑊
# MAGIC 1
# MAGIC 𝑇
# MAGIC +
# MAGIC 𝑏
# MAGIC 1
# MAGIC fc1(x)=xW 
# MAGIC 1
# MAGIC T
# MAGIC ​
# MAGIC  +b 
# MAGIC 1
# MAGIC ​
# MAGIC  
# MAGIC Where:
# MAGIC
# MAGIC x is [batch_size, input_features] → shape [2, 2]
# MAGIC
# MAGIC W1 is [3, 2] → transforms 2 inputs into 3 outputs
# MAGIC
# MAGIC Output of fc1 will be [2, 3]
# MAGIC
# MAGIC Let’s assume the layer has:
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC View in %md code 
# MAGIC """
# MAGIC fc1.weight = [
# MAGIC     [w11, w12],
# MAGIC     [w21, w22],
# MAGIC     [w31, w32]
# MAGIC ]
# MAGIC
# MAGIC fc1.bias = [b1, b2, b3]
# MAGIC """
# MAGIC Pythonic implementation is:
# MAGIC [1.0, 2.0] @ [[w11, w21, w31],
# MAGIC               [w12, w22, w32]] + [b1, b2, b3]

# COMMAND ----------

# MAGIC %md
# MAGIC See Internals by Printing Weights

# COMMAND ----------

print("fc1 weights:\n", model.fc1.weight)
print("fc1 bias:\n", model.fc1.bias)

print("fc2 weights:\n", model.fc2.weight)
print("fc2 bias:\n", model.fc2.bias)
