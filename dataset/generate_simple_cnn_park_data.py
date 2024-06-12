# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.optim as optim
from sklearn.model_selection import ParameterSampler
import logging
import json
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR = './simple-cnn-park'
RESULTS_DIR = os.path.join(DATA_DIR, 'training_data')
os.makedirs(RESULTS_DIR, exist_ok=True)

def dynamic_print(data):
    sys.stdout.write("\r\x1b[K" + data)
    sys.stdout.flush()

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.feature_layer = nn.Sequential(
            model.conv1,
            nn.ReLU(),
            model.conv2,
            nn.ReLU(),
            model.conv3,
            nn.ReLU()
        )

    def forward(self, x):
        x = self.feature_layer(x)
        return x

class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1) 
        x = self.fc(x)
        return x


class ZooDataset(Dataset):
    def __init__(self, data_path, mode, idcs_file=None):
        super().__init__()
        data = np.load(os.path.join(data_path, "weights.npy"))
        shuffled_idcs = pd.read_csv(idcs_file, header=None).values.flatten()
        data = data[shuffled_idcs]
        metrics = pd.read_csv(os.path.join(data_path, "metrics.csv.gz"), compression='gzip')
        metrics = metrics.iloc[shuffled_idcs]
        self.layout = pd.read_csv(os.path.join(data_path, "layout.csv"))
        isfinal = metrics["step"] == 86
        metrics = metrics[isfinal]
        data = data[isfinal]
        assert np.isfinite(data).all()
        metrics.index = np.arange(0, len(metrics))
        idcs = self._split_indices_iid(data)[mode]
        data = data[idcs]
        self.metrics = metrics.iloc[idcs]
        self.weights, self.biases = [], []
        for i, row in self.layout.iterrows():
            arr = data[:, row["start_idx"]:row["end_idx"]]
            bs = arr.shape[0]
            arr = arr.reshape((bs, *eval(row["shape"])))
            if row["varname"].endswith("kernel:0"):
                if arr.ndim == 5:
                    arr = arr.transpose(0, 4, 3, 1, 2)
                elif arr.ndim == 3:
                    arr = arr.transpose(0, 2, 1)
                self.weights.append(arr)
            elif row["varname"].endswith("bias:0"):
                self.biases.append(arr)
            else:
                raise ValueError(f"varname {row['varname']} not recognized.")
    
    def _split_indices_iid(self, data):
        splits = {}
        test_split_point = int(0.5 * len(data))
        splits["test"] = list(range(test_split_point, len(data)))
        trainval_idcs = list(range(test_split_point))
        val_point = int(0.8 * len(trainval_idcs))
        rng = random.Random(0)
        rng.shuffle(trainval_idcs)
        splits["train"] = trainval_idcs[:val_point]
        splits["val"] = trainval_idcs[val_point:]
        return splits
    
    def __len__(self):
        return len(self.weights[0])
    
    def __getitem__(self, idx):
        weights = tuple(w[idx][None] for w in self.weights)
        biases = tuple(b[idx][None] for b in self.biases)
        return (weights, biases), self.metrics.iloc[idx].test_accuracy.item()


zoo_path = './dataset/simple-cnn-park/zoo'
zoo_dataset = ZooDataset(zoo_path, "train", f'{zoo_path}/cifar10_split.csv')
num_pretrained_configs = len(zoo_dataset)

param_grid = {
    'pretrained_index': range(num_pretrained_configs),
    'batch_size': [16, 32, 64, 128, 256, 512],
    'learning_rate': np.logspace(-4, -1, num=4),
    'momentum': [0.1, 0.5, 0.9],
    'weight_decay': np.logspace(-5, -1, num=5)
}

num_configs = 2000
sampler = ParameterSampler(param_grid, n_iter=num_configs, random_state=42)
configs = list(sampler)
df_configs = pd.DataFrame(configs)
df_configs.index.name = 'hp_index'
df_configs.to_csv(f'{DATA_DIR}/hyperparameters.csv')

def load_weights(model, weights, biases, device):
    logging.info("Loading weights and biases into the model.")
    model.conv1.weight.data = torch.from_numpy(weights[0].squeeze(0)).to(device)
    model.conv1.bias.data = torch.from_numpy(biases[0].squeeze(0)).to(device)
    model.conv2.weight.data = torch.from_numpy(weights[1].squeeze(0)).to(device)
    model.conv2.bias.data = torch.from_numpy(biases[1].squeeze(0)).to(device)
    model.conv3.weight.data = torch.from_numpy(weights[2].squeeze(0)).to(device)
    model.conv3.bias.data = torch.from_numpy(biases[2].squeeze(0)).to(device)
    model.fc.weight.data = torch.from_numpy(weights[3].squeeze(0)).reshape(10, 16).to(device)
    model.fc.bias.data = torch.from_numpy(biases[3].squeeze(0)).to(device)
    logging.info("Weights and biases loaded successfully.")

import time
from torchvision.transforms import ToTensor

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    train_loss = running_loss / total
    train_accuracy = correct / total
    return train_loss, train_accuracy

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    test_loss = running_loss / total
    test_accuracy = correct / total
    return test_loss, test_accuracy

def load_and_train(hp_index, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN().to(device)
    feature_extractor = FeatureExtractor(model).to(device)

    (weights, biases), _ = zoo_dataset[config['pretrained_index']]
    load_weights(model, weights, biases, device)

    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    epoch_data = []
    for epoch in range(50):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        dynamic_print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")
        
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"model_{hp_index}_epoch_{epoch}.pt"))

        features = feature_extractor(torch.randn(1, 1, 28, 28).to(device))
        torch.save(features, os.path.join(RESULTS_DIR, f"features_{hp_index}_epoch_{epoch}.pt"))

        epoch_result = {
            "hp_index": hp_index, "epoch": epoch, "train_loss": train_loss,
            "train_accuracy": train_accuracy, "test_loss": test_loss,
            "test_accuracy": test_accuracy
        }
        epoch_data.append(epoch_result)

    with open(os.path.join(RESULTS_DIR, f"results_{hp_index}.json"), 'w') as f:
        json.dump(epoch_data, f, indent=4)

if __name__ == "__main__":
    for hp_index, config in enumerate(configs):
        print(hp_index)
        load_and_train(hp_index, config)
