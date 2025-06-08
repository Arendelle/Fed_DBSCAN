import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import random

# ========== CUDA 设置 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========== CNN 模型 ==========
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 28x28 → 12x12
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # 12x12 → 4x4
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ========== 客户端本地训练函数 ==========
def local_train(model, train_loader, epochs=1, lr=0.01):
    model = copy.deepcopy(model)
    model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    return model.cpu().state_dict()  # 保存为CPU上的权重便于聚合

# ========== 模型聚合函数 ==========
def average_weights(weights_list):
    avg_weights = copy.deepcopy(weights_list[0])
    for key in avg_weights:
        for i in range(1, len(weights_list)):
            avg_weights[key] += weights_list[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(weights_list))
    return avg_weights

# ========== 测试函数 ==========
def test(model, test_loader):
    model.to(device)
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    acc = 100. * correct / len(test_loader.dataset)
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

# ========== 数据准备 ==========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, transform=transform)

# 模拟 K 个客户端
K = 5
client_data_size = len(train_dataset) // K
client_datasets = [Subset(train_dataset, list(range(i * client_data_size, (i + 1) * client_data_size))) for i in range(K)]
client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in client_datasets]
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ========== 联邦学习主循环 ==========
global_model = CNN()
rounds = 10
for r in range(rounds):
    local_weights = []
    print(f"--- Round {r + 1} ---")
    for client_idx in range(K):
        local_model_weights = local_train(global_model, client_loaders[client_idx], epochs=1)
        local_weights.append(local_model_weights)
    averaged_weights = average_weights(local_weights)
    global_model.load_state_dict(averaged_weights)
    test(global_model, test_loader)
