import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import random
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

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

# 计算模型更新（向量化参数差）
def model_diff_vector(base_state, updated_state):
    diff = []
    for key in base_state:
        # 显式将两个 state 都搬到 CPU
        base_tensor = base_state[key].cpu()
        updated_tensor = updated_state[key].cpu()
        diff_tensor = updated_tensor - base_tensor
        diff.append(diff_tensor.view(-1))
    return torch.cat(diff).numpy()

# 构造恶意客户端
def make_malicious_dataset(dataset, shuffle_labels=True):
    if shuffle_labels:
        labels = [label for _, label in dataset]
        np.random.shuffle(labels)
        shuffled_data = [(img, labels[i]) for i, (img, _) in enumerate(dataset)]
        return shuffled_data
    return dataset

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
K = 10  # 客户端总数
client_fraction = 0.5  # 每轮选择 50% 用户参加训练
num_selected = max(1, int(client_fraction * K))
client_data_size = len(train_dataset) // K
client_datasets = [Subset(train_dataset, list(range(i * client_data_size, (i + 1) * client_data_size))) for i in range(K)]
client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in client_datasets]
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 客户端初始化及恶意注入
malicious_ratio = 0.2  # 20% 客户端是恶意的
num_malicious = int(K * malicious_ratio)
malicious_clients = random.sample(range(K), num_malicious)
print(f"Malicious clients: {malicious_clients}")

client_datasets = []
for i in range(K):
    subset = Subset(train_dataset, list(range(i * client_data_size, (i + 1) * client_data_size)))
    if i in malicious_clients:
        # 使用打乱标签的数据（转换为list，打乱）
        subset = make_malicious_dataset(subset)
    client_datasets.append(subset)

client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in client_datasets]

# 联邦学习循环 + DBSCAN 检测
global_model = CNN().to(device)
rounds = 10

for r in range(rounds):
    print(f"\n--- Round {r+1} ---")
    local_weights = []
    diff_vectors = []

    base_weights = copy.deepcopy(global_model.cpu().state_dict())

    selected_clients = random.sample(range(K), num_selected)
    print("Selected clients:", selected_clients)

    for client_idx in selected_clients:
        local_model_weights = local_train(global_model, client_loaders[client_idx], epochs=1)
        local_weights.append(local_model_weights)

        vec = model_diff_vector(base_weights, local_model_weights)
        diff_vectors.append(vec)

    # 特征标准化 + DBSCAN
    X = StandardScaler().fit_transform(diff_vectors)
    clustering = DBSCAN(eps=240.0, min_samples=2).fit(X)

    # 聚类标签仅对应选中客户端
    selected_indices = [i for i, label in enumerate(clustering.labels_) if label != -1]
    print("DBSCAN labels:", clustering.labels_)
    print("Selected (benign) clients:", [selected_clients[i] for i in selected_indices])

    if not selected_indices:
        print("⚠️ 全部参与客户端被标记为异常，使用全部参与者")
        selected_indices = list(range(len(selected_clients)))

    benign_weights = [local_weights[i] for i in selected_indices]
    averaged_weights = average_weights(benign_weights)
    global_model.load_state_dict(averaged_weights)

    test(global_model, test_loader)
