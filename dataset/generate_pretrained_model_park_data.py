import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import ParameterSampler
import numpy as np
import pandas as pd
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory configurations
DATA_DIR = './pretrained_model_park/training_data'
CIFAR_DIR = os.path.join(DATA_DIR, 'cifar10_training')
SVHN_DIR = os.path.join(DATA_DIR, 'svhn_training')
os.makedirs(CIFAR_DIR, exist_ok=True)
os.makedirs(SVHN_DIR, exist_ok=True)

# Import architectures
from dataset.pretrained_model_park.models.basic_cnn import make_cnn
from dataset.pretrained_model_park.models.basic_cnn_1d import make_cnn_1d
from dataset.pretrained_model_park.models.deepsets import make_deepsets
from dataset.pretrained_model_park.models.transformer import make_transformer
from dataset.pretrained_model_park.models.resnet import make_resnet

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.feature_layer = model.feature_layer if hasattr(model, 'feature_layer') else nn.Identity()

    def forward(self, x):
        return self.feature_layer(x)

def get_dataloaders(batch_size, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    datasets_dict = {
        'cifar10': datasets.CIFAR10(root='./data/cifar10', train=train, download=True, transform=transform),
        'svhn': datasets.SVHN(root='./data/svhn', split='train' if train else 'test', download=True, transform=transform)
    }
    return {k: DataLoader(v, batch_size=batch_size, shuffle=True) for k, v in datasets_dict.items()}

def train_and_evaluate(model, train_loaders, test_loaders, device, epochs, lr, weight_decay, momentum, architecture_name, hp_index):
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    for dataset_name, loader in train_loaders.items():
        test_loader = test_loaders[dataset_name]
        results = []
        for epoch in range(epochs):
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                train_correct += (outputs.argmax(1) == labels).sum().item()
                train_total += inputs.size(0)

            train_accuracy = train_correct / train_total

            # Evaluate on test data
            model.eval()
            test_loss, test_correct, test_total = 0, 0, 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item() * inputs.size(0)
                    test_correct += (outputs.argmax(1) == labels).sum().item()
                    test_total += inputs.size(0)

            test_accuracy = test_correct / test_total

            logging.info(f"Epoch {epoch+1} [{dataset_name}]: Train Loss: {train_loss/train_total:.6f}, Train Acc: {train_accuracy:.4f}, Test Loss: {test_loss/test_total:.6f}, Test Acc: {test_accuracy:.4f}")

            # Save model and results
            directory = CIFAR_DIR if dataset_name == 'cifar10' else SVHN_DIR
            checkpoint_path = os.path.join(directory, f"{architecture_name}_{dataset_name}_epoch_{epoch+1}_{hp_index}.pt")
            torch.save(model.state_dict(), checkpoint_path)

            results.append({
                'epoch': epoch+1,
                'train_loss': train_loss/train_total,
                'train_accuracy': train_accuracy,
                'test_loss': test_loss/test_total,
                'test_accuracy': test_accuracy
            })

        # Save results to JSON
        results_path = os.path.join(directory, f"{hp_index}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

def main(start_idx=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architectures = {
        'cnn': lambda: make_cnn(num_classes=10),
        'deepsets': lambda: make_deepsets(num_classes=10),
        'transformer': lambda: make_transformer(in_dim=3, hidden_dim=64, num_heads=2, out_dim=10, dropout=0.1, num_layers=2, vit=True, patch_size=4),
        'resnet': lambda: make_resnet(num_classes=10)
    }
    param_grid = {
        'architecture': list(architectures.keys()),
        'batch_size': [16, 32, 64, 128, 256, 512],
        'learning_rate': np.logspace(-4, -1, num=4),
        'momentum': [0.1, 0.5, 0.9],
        'weight_decay': np.logspace(-5, -1, num=5)
    }
    sampler = ParameterSampler(param_grid, n_iter=50, random_state=42)
    configs = list(sampler)
    for idx, config in enumerate(configs):
        config['hp_index'] = idx
    df_configs = pd.DataFrame(configs)
    df_configs.to_csv(os.path.join(DATA_DIR, 'hyperparameters.csv'), index=False)

    train_loaders = get_dataloaders(64, train=True)
    test_loaders = get_dataloaders(64, train=False)

    for config in configs:
        if config['hp_index'] >= start_idx:
            model = architectures[config['architecture']]()  # call the lambda function here
            logging.info(f"Training configuration {config['hp_index']}: {config['architecture']} with lr={config['learning_rate']}, weight_decay={config['weight_decay']}, momentum={config['momentum']}, batch_size={config['batch_size']}")
            train_and_evaluate(model, train_loaders, test_loaders, device, 20, config['learning_rate'], config['weight_decay'], config['momentum'], config['architecture'], config['hp_index'])

if __name__ == "__main__":
    main(start_idx=5)
