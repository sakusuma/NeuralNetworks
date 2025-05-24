# Databricks notebook source
# MAGIC %md
# MAGIC 🧭 What You'll Learn
# MAGIC Define a simple dataset
# MAGIC
# MAGIC Use loss function (e.g., MSE for regression)
# MAGIC
# MAGIC Use optimizer (e.g., SGD)
# MAGIC
# MAGIC Train the model (forward → loss → backward → update)
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC 🏗️ Step 1: Define a Simple Dataset
# MAGIC Let’s say we want to learn this function:
# MAGIC
# MAGIC 𝑦 =
# MAGIC 2
# MAGIC 𝑥
# MAGIC 1
# MAGIC +
# MAGIC 3
# MAGIC 𝑥
# MAGIC 2
# MAGIC y=2x 
# MAGIC 1
# MAGIC ​
# MAGIC  +3x 
# MAGIC 2
# MAGIC ​
# MAGIC  
# MAGIC Here’s how we can create some training data:

# COMMAND ----------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Set random seed for reproducibility
torch.manual_seed(0)

# Inputs: 4 samples with 2 features each
X = torch.tensor([[1.0, 2.0],
                  [2.0, 3.0],
                  [3.0, 4.0],
                  [4.0, 5.0]])

# Outputs using y = 2*x1 + 3*x2
y = torch.tensor([[8.0],
                  [13.0],
                  [18.0],
                  [23.0]])


# COMMAND ----------

# MAGIC %md
# MAGIC 🧠 Step 2: Define the Model
# MAGIC
# MAGIC Same as before:

# COMMAND ----------

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 1)  # Directly map 2 inputs to 1 output

    def forward(self, x):
        return self.fc1(x)

model = SimpleNet()


# COMMAND ----------

# MAGIC %md
# MAGIC 💔 Step 3: Define the Loss Function
# MAGIC
# MAGIC Since this is a regression problem:

# COMMAND ----------

criterion = nn.MSELoss()


# COMMAND ----------

# MAGIC %md
# MAGIC ⚙️ Step 4: Define the Optimizer
# MAGIC
# MAGIC We’ll use Stochastic Gradient Descent (SGD):

# COMMAND ----------

optimizer = optim.SGD(model.parameters(), lr=0.01)


# COMMAND ----------

# MAGIC %md
# MAGIC 🔁 Step 5: Training Loop
# MAGIC
# MAGIC We’ll train for 1000 epochs:
# MAGIC
# MAGIC Each time this loop runs, it does one epoch of training, which includes:
# MAGIC
# MAGIC ✅ 1. outputs = model(X)
# MAGIC This does a forward pass using your current model weights.
# MAGIC
# MAGIC 🔍 Example (initial epoch):
# MAGIC
# MAGIC Your model starts with random weights and bias, say w1, w2, and b.
# MAGIC
# MAGIC It computes:
# MAGIC
# MAGIC 𝑦
# MAGIC ^
# MAGIC =
# MAGIC 𝑋
# MAGIC @
# MAGIC [
# MAGIC 𝑤
# MAGIC 1
# MAGIC ,
# MAGIC 𝑤
# MAGIC 2
# MAGIC ]
# MAGIC 𝑇
# MAGIC +
# MAGIC 𝑏
# MAGIC y
# MAGIC ^
# MAGIC ​
# MAGIC  =X@[w1,w2] 
# MAGIC T
# MAGIC  +b
# MAGIC This gives a prediction outputs for all 4 training samples.
# MAGIC
# MAGIC 📉 2. loss = criterion(outputs, y)
# MAGIC This computes the loss between predicted and actual values.
# MAGIC
# MAGIC Since we use Mean Squared Error (MSE):
# MAGIC
# MAGIC loss =
# MAGIC 1
# MAGIC 𝑁
# MAGIC ∑
# MAGIC 𝑖
# MAGIC =
# MAGIC 1
# MAGIC 𝑁
# MAGIC (
# MAGIC 𝑦
# MAGIC ^
# MAGIC 𝑖
# MAGIC −
# MAGIC 𝑦
# MAGIC 𝑖
# MAGIC )
# MAGIC 2
# MAGIC loss= 
# MAGIC N
# MAGIC 1
# MAGIC ​
# MAGIC   
# MAGIC i=1
# MAGIC ∑
# MAGIC N
# MAGIC ​
# MAGIC  ( 
# MAGIC y
# MAGIC ^
# MAGIC ​
# MAGIC   
# MAGIC i
# MAGIC ​
# MAGIC  −y 
# MAGIC i
# MAGIC ​
# MAGIC  ) 
# MAGIC 2
# MAGIC  
# MAGIC So this tells us how wrong the model is right now.
# MAGIC
# MAGIC 🔄 3. optimizer.zero_grad()
# MAGIC This clears previous gradients from the last epoch.
# MAGIC If you skip this, gradients would accumulate (which we don't want in this case).
# MAGIC
# MAGIC 🔙 4. loss.backward()
# MAGIC This is the backpropagation step.
# MAGIC
# MAGIC It computes the gradient of the loss with respect to all model parameters (i.e., w1, w2, and b in our case).
# MAGIC
# MAGIC PyTorch tracks all operations during the forward pass and builds a computation graph, so when we call .backward(), it can automatically compute the gradients using the chain rule.
# MAGIC
# MAGIC 🧠 5. optimizer.step()
# MAGIC This is where the actual learning happens.
# MAGIC
# MAGIC It takes the gradients computed in the .backward() step and updates the weights in the direction that reduces the loss.
# MAGIC
# MAGIC In the case of SGD:
# MAGIC
# MAGIC 𝑤
# MAGIC :
# MAGIC =
# MAGIC 𝑤
# MAGIC −
# MAGIC lr
# MAGIC ×
# MAGIC ∂
# MAGIC loss
# MAGIC ∂
# MAGIC 𝑤
# MAGIC w:=w−lr× 
# MAGIC ∂w
# MAGIC ∂loss
# MAGIC ​
# MAGIC  
# MAGIC where lr is the learning rate (0.01 in this case).
# MAGIC
# MAGIC 🖨️ 6. Print Loss Every 100 Epochs
# MAGIC This helps you observe the loss reducing over time, like:

# COMMAND ----------

for epoch in range(1000):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()  # Clear gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')


# COMMAND ----------

# MAGIC %md
# MAGIC ✅ Step 6: Test the Trained Model
# MAGIC

# COMMAND ----------

test_input = torch.tensor([[5.0, 6.0]])
predicted = model(test_input)
print("Prediction for [5,6]:", predicted.item())


# COMMAND ----------

print("Weights:", model.fc1.weight)
print("Bias:", model.fc1.bias)