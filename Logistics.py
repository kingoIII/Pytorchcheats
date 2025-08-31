import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Create toy data for 3 tasks:
# Regression, Binary Classification, Multi-class Classification

# 1. Regression Data (y = 2x + 1)
X_reg = torch.linspace(-1, 1, 100).unsqueeze(1)
y_reg = 2 * X_reg + 1 + 0.1 * torch.randn(X_reg.size())

# 2. Binary Classification Data (threshold)
X_bin = torch.linspace(-1, 1, 100).unsqueeze(1)
y_bin = (X_bin > 0).float()

# 3. Multi-class Classification Data (3 classes)
X_multi = torch.cat([torch.randn(100, 2) + i for i in range(3)])
y_multi = torch.tensor([i for i in range(3) for _ in range(100)])

# Define simple models for each task
class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class MultiClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3)

    def forward(self, x):
        return self.linear(x)  # CrossEntropyLoss applies softmax internally

# Initialize models and loss functions
model_reg = Regressor()
loss_reg = nn.MSELoss()

model_bin = BinaryClassifier()
loss_bin = nn.BCELoss()

model_multi = MultiClassifier()
loss_multi = nn.CrossEntropyLoss()

# Simulate a forward pass with toy data
out_reg = model_reg(X_reg)
loss_r = loss_reg(out_reg, y_reg)

out_bin = model_bin(X_bin)
loss_b = loss_bin(out_bin, y_bin)

out_multi = model_multi(X_multi)
loss_m = loss_multi(out_multi, y_multi)

(loss_r.item(), loss_b.item(), loss_m.item())
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Create toy data for 3 tasks:
# Regression, Binary Classification, Multi-class Classification

# 1. Regression Data (y = 2x + 1)
X_reg = torch.linspace(-1, 1, 100).unsqueeze(1)
y_reg = 2 * X_reg + 1 + 0.1 * torch.randn(X_reg.size())

# 2. Binary Classification Data (threshold)
X_bin = torch.linspace(-1, 1, 100).unsqueeze(1)
y_bin = (X_bin > 0).float()

# 3. Multi-class Classification Data (3 classes)
X_multi = torch.cat([torch.randn(100, 2) + i for i in range(3)])
y_multi = torch.tensor([i for i in range(3) for _ in range(100)])

# Define simple models for each task
class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class MultiClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3)

    def forward(self, x):
        return self.linear(x)  # CrossEntropyLoss applies softmax internally

# Initialize models and loss functions
model_reg = Regressor()
loss_reg = nn.MSELoss()

model_bin = BinaryClassifier()
loss_bin = nn.BCELoss()

model_multi = MultiClassifier()
loss_multi = nn.CrossEntropyLoss()

# Simulate a forward pass with toy data
out_reg = model_reg(X_reg)
loss_r = loss_reg(out_reg, y_reg)

out_bin = model_bin(X_bin)
loss_b = loss_bin(out_bin, y_bin)

out_multi = model_multi(X_multi)
loss_m = loss_multi(out_multi, y_multi)

(loss_r.item(), loss_b.item(), loss_m.item())
