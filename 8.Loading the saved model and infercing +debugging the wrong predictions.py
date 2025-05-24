# Databricks notebook source
# MAGIC %md
# MAGIC Save your current trained model.
# MAGIC
# MAGIC Restart the notebook (simulate a new session).
# MAGIC
# MAGIC Recreate the model class.
# MAGIC
# MAGIC Load weights.
# MAGIC
# MAGIC Run inference on test data.

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

# MAGIC %md
# MAGIC

# COMMAND ----------

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

# COMMAND ----------

# Later... Load it back
loaded_model = BatchNormModel()
loaded_model.load_state_dict(torch.load("mnist_batchnorm_model.pt"))
loaded_model.eval()

# Move to GPU if needed
loaded_model.to(device)


# COMMAND ----------

# MAGIC %md
# MAGIC Define Inference Loop

# COMMAND ----------

correct = 0
total = 0

loaded_model.to(device)
loaded_model.eval()

with torch.no_grad():  # No gradients needed
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = loaded_model(images)
        _, predicted = torch.max(outputs.data, 1)  # get class with highest logit
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Inference Accuracy: {accuracy:.2f}%")


# COMMAND ----------

# MAGIC %md
# MAGIC Visualize Predictions

# COMMAND ----------

import matplotlib.pyplot as plt

classes = list(range(10))  # MNIST digits: 0–9

def imshow(img):
    img = img.squeeze()  # remove channel dim
    plt.imshow(img.cpu(), cmap='gray')
    plt.axis('off')
    plt.show()

# Show 5 sample predictions
loaded_model.eval()
with torch.no_grad():
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    outputs = loaded_model(images)
    _, preds = torch.max(outputs, 1)

    for i in range(5):
        imshow(images[i])
        print(f"Ground Truth: {labels[i].item()}, Predicted: {preds[i].item()}")


# COMMAND ----------

# MAGIC %md
# MAGIC Understanding the predictions that went wrong

# COMMAND ----------

import matplotlib.pyplot as plt

def imshow(img, title):
    img = img.squeeze().cpu()  # Remove channel and move to CPU
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Set model to evaluation mode
loaded_model.eval()

# Store incorrect predictions
wrong_images = []
wrong_labels = []
wrong_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = loaded_model(images)
        _, preds = torch.max(outputs, 1)

        # Find indices where predictions are wrong
        wrong = preds != labels
        wrong_images.extend(images[wrong])
        wrong_labels.extend(labels[wrong])
        wrong_preds.extend(preds[wrong])

# Show first 5 misclassified images
print(f"Total Misclassified Images: {len(wrong_images)}")
for i in range(min(5, len(wrong_images))):
    title = f"True: {wrong_labels[i].item()} | Pred: {wrong_preds[i].item()}"
    imshow(wrong_images[i], title)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Group Misclassifications by Digit Confusion
# MAGIC
# MAGIC Run inference over test data
# MAGIC
# MAGIC Track misclassified (true_label, predicted_label) pairs
# MAGIC
# MAGIC Count them using a dictionary
# MAGIC
# MAGIC Optionally visualize some from each category

# COMMAND ----------

from collections import defaultdict
import matplotlib.pyplot as plt

# Dictionary: key=(true, pred), value=list of images
confusion_mistakes = defaultdict(list)

loaded_model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = loaded_model(images)
        _, preds = torch.max(outputs, 1)

        mismatches = preds != labels
        for img, true, pred in zip(images[mismatches], labels[mismatches], preds[mismatches]):
            confusion_mistakes[(true.item(), pred.item())].append(img.cpu())

# Show summary
print("Confusion Summary (True ➜ Predicted):")
for (true, pred), imgs in sorted(confusion_mistakes.items()):
    print(f"{true} ➜ {pred} : {len(imgs)} times")


# COMMAND ----------

def show_confused_digits(true_digit, predicted_digit, max_imgs=5):
    key = (true_digit, predicted_digit)
    images = confusion_mistakes.get(key, [])
    print(f"Showing {len(images)} examples of {true_digit} ➜ {predicted_digit}")
    for i in range(min(max_imgs, len(images))):
        img = images[i].squeeze()
        plt.imshow(img, cmap='gray')
        plt.title(f"{true_digit} ➜ {predicted_digit}")
        plt.axis('off')
        plt.show()

# Example usage
show_confused_digits(3, 5)
