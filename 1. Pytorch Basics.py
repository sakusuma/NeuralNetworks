# Databricks notebook source
# MAGIC %md
# MAGIC Source: https://chatgpt.com/share/6831ef3b-4d44-8003-a035-23eeb55ba428

# COMMAND ----------

# MAGIC %md
# MAGIC a) Install PyTorch

# COMMAND ----------

pip install torch torchvision

# COMMAND ----------

# MAGIC %md
# MAGIC b) Tensors: PyTorch’s core data structure

# COMMAND ----------

import torch

# Create a tensor
x = torch.tensor([1.0, 2.0, 3.0])
print(x)



# COMMAND ----------


# Tensor of zeros and ones
print(torch.zeros(2, 3))
print(torch.ones(2, 3))



# COMMAND ----------

# Random tensor
print(torch.rand(2, 2))



# COMMAND ----------

# Tensor operations
y = torch.tensor([4.0, 5.0, 6.0])
print(x + y)
print(x * y)

# COMMAND ----------

# MAGIC %md
# MAGIC c) Gradients with autograd : We'll break it into operations and draw the graph. Each node represents an operation, and edges show the flow of data.
# MAGIC
# MAGIC          x = [2.0, 3.0]           ← Leaf tensor (requires_grad=True)
# MAGIC              /     \
# MAGIC         [x**2]   [3 * x]          ← Element-wise operations
# MAGIC              \     /
# MAGIC             y = x**2 + 3x         ← Element-wise addition
# MAGIC                   |
# MAGIC            z = sum(y)             ← Reduces y to scalar
# MAGIC                   |
# MAGIC            z.backward()           ← Starts gradient computation
# MAGIC
# MAGIC   
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Now let’s annotate each operation with the gradient flow
# MAGIC       x = [2.0, 3.0]
# MAGIC              |
# MAGIC        ----------------
# MAGIC        |              |
# MAGIC    x**2 (→ 2x)     3*x (→ 3)
# MAGIC        \              /
# MAGIC          \          /
# MAGIC          y = x**2 + 3x
# MAGIC                 |
# MAGIC            z = y.sum()
# MAGIC                 |
# MAGIC        ∂z/∂x = 2x + 3 = [7.0, 9.0]
# MAGIC          

# COMMAND ----------

x = torch.tensor([2.0, 3.0], requires_grad=True)
print(x)

# COMMAND ----------

x**2

# COMMAND ----------

3*x

# COMMAND ----------

y = x**2 + 3 * x
print(y)

# COMMAND ----------

z = y.sum()

# COMMAND ----------

z

# COMMAND ----------

# Backpropagation
z.backward()

# COMMAND ----------

print(x.grad)  # dz/dx