import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import random
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import csv

from Nets import CNNMnist

# ========== CUDA 设置 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========== 客户端本地训练函数 ==========
def local_train(model, train_loader, epochs=1, lr=0.005):
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

# FedAvg聚合函数
def fed_avg(weights_list):
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

# 构造恶意客户端数据集
def make_malicious_dataset(dataset, shuffle_labels=True):
    if shuffle_labels:
        labels = [label for _, label in dataset]
        np.random.shuffle(labels)
        shuffled_data = [(img, labels[i]) for i, (img, _) in enumerate(dataset)]
        return shuffled_data
    return dataset
# 构造恶意客户端数据集 - 有目标攻击
def targeted_malicious_dataset(dataset, source_label=None, target_label=None):
    # 如果未指定攻击标签，则随机选择一个 source → target 映射
    if source_label is None or target_label is None:
        source_label, target_label = random.sample(range(10), 2)
    print(f"Malicious attack: {source_label} → {target_label}")

    # 将数据转换为 list，便于处理
    data_list = list(dataset)

    # 有目标攻击：将 source_label 的样本标签篡改为 target_label
    poisoned_data = []
    for img, label in data_list:
        if label == source_label:
            poisoned_data.append((img, target_label))
        else:
            poisoned_data.append((img, label))

    return poisoned_data


# loss和accuracy测试函数
def test(model, test_loader):
    model.to(device)
    model.eval()
    correct = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)  # 乘以 batch size
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

# 入口函数
def main():
    # ========== 数据准备 ==========
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST('./data', train=False, transform=transform)

    # 客户端模拟
    K = 50  # 客户端总数
    client_fraction = 0.7  # 每轮选择参加训练的用户比例
    num_selected = max(1, int(client_fraction * K))
    client_data_size = len(train_dataset) // K
    client_datasets = [Subset(train_dataset, list(range(i * client_data_size, (i + 1) * client_data_size))) for i in range(K)]
    client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 客户端及恶意客户端初始化
    malicious_ratio = 0.4  # 恶意客户端占比
    num_malicious = int(K * malicious_ratio)
    malicious_clients = random.sample(range(K), num_malicious)
    # malicious_clients = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    print(f"恶意客户端编号为： {malicious_clients}")

    client_datasets = []
    for i in range(K):
        subset = Subset(train_dataset, list(range(i * client_data_size, (i + 1) * client_data_size)))
        if i in malicious_clients:
            # 使用打乱标签的数据（转换为list，打乱）
            # subset = make_malicious_dataset(subset)
            # 有目标攻击
            subset = targeted_malicious_dataset(subset, 1, 8)
        client_datasets.append(subset)

    client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in client_datasets]

    # 输出训练配置
    print(f"客户端总数： {K}")
    print(f"恶意客户端占比： {malicious_ratio}")
    print(f"每轮参与的客户端数量： {num_selected}")
    # 带恶意客户端检测的训练
    print("开始带恶意客户端检测的训练")
    loss_history_w_detection, acc_history_w_detection = fed_loop(K, num_selected, client_loaders, test_loader, True)
    # 不带恶意客户端检测的训练
    print("开始不含恶意客户端检测的训练")
    loss_history_wo_detection, acc_history_wo_detection = fed_loop(K, num_selected, client_loaders, test_loader, False)
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history_w_detection, label="Loss with Detection")
    plt.plot(loss_history_wo_detection, label="Loss without Detection")
    plt.title("Loss w/wo Detection")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()
    # plt.savefig("./out/loss_curve_mal_{}.png".format(malicious_ratio))
    # 绘制准确度曲线
    plt.figure(figsize=(10, 5))
    plt.plot(acc_history_w_detection, label="Accuracy with Detection")
    plt.plot(acc_history_wo_detection, label="Accuracy without Detection")
    plt.title("Accuracy w/wo Detection")
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid()
    plt.show()
    # plt.savefig("./out/acc_curve_mal_{}.png".format(malicious_ratio))
    # 存储统计数据
    filename = "./out/output_acc_wo_detection_mal_" + str(malicious_ratio) + ".csv"
    with open(filename, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(acc_history_wo_detection)

    filename = "./out/output_acc_w_detection_mal_" + str(malicious_ratio) + ".csv"
    with open(filename, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(acc_history_w_detection)

# 联邦学习主循环
def fed_loop(K, num_selected, client_loaders, test_loader, detection: bool):
    # 联邦学习主循环
    global_model = CNNMnist().to(device)
    rounds = 50
    test_loss_history = []
    test_acc_history = []

    for r in range(rounds):
        print(f"\n--- Round {r+1} / {rounds} ---")
        local_weights = []
        diff_vectors = []

        base_weights = copy.deepcopy(global_model.cpu().state_dict())

        selected_clients = random.sample(range(K), num_selected)
        print("Selected candidate clients:", selected_clients)

        for client_idx in selected_clients:
            local_model_weights = local_train(global_model, client_loaders[client_idx], epochs=1)
            local_weights.append(local_model_weights)

            vec = model_diff_vector(base_weights, local_model_weights)
            diff_vectors.append(vec)

        if detection:   # 运行DBSCAN聚类检测
            # 特征标准化
            X = StandardScaler().fit_transform(diff_vectors)
            # DBSCAN聚类
            clustering = DBSCAN(eps=190.0, min_samples=2).fit(X)

            # 筛选出最多客户端的簇
            # selected_indices = [i for i, label in enumerate(clustering.labels_) if label != -1]
            dbscan_labels = clustering.labels_
            unique, counts = np.unique(dbscan_labels[dbscan_labels != -1], return_counts=True)
            largest_cluster = 0
            if len(unique) > 0:
                max_count_idx = np.argmax(counts)
                largest_cluster = unique[max_count_idx]
                # print("样本最多的簇为:", largest_cluster)

            selected_indices = [i for i, label in enumerate(dbscan_labels) if label == largest_cluster]
            print("DBSCAN labels:             ", dbscan_labels)
            print("Selected (benign) clients:", [selected_clients[i] for i in selected_indices])

            if not selected_indices:
                print("警告：全部参与客户端被标记为异常，使用全部参与者")
                selected_indices = list(range(len(selected_clients)))
        else:   # 不运行DBSCAN聚类检测
            print("接受恶意客户端参与训练")
            selected_indices = np.arange(len(selected_clients))

        benign_weights = [local_weights[i] for i in selected_indices]
        averaged_weights = fed_avg(benign_weights)
        global_model.load_state_dict(averaged_weights)

        loss, acc = test(global_model, test_loader)
        test_loss_history.append(loss)
        test_acc_history.append(acc)

    return test_loss_history, test_acc_history

if __name__ == "__main__":
    main()