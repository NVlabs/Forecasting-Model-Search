# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

from dataset.pretrained_model_park.models.basic_cnn import make_cnn
from dataset.pretrained_model_park.models.basic_cnn_1d import make_cnn_1d
from dataset.pretrained_model_park.models.deepsets import make_deepsets
from dataset.pretrained_model_park.models.transformer import make_transformer
from dataset.pretrained_model_park.models.resnet import make_resnet

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PretrainedModelParkInterface:
    def __init__(self, dataset_type='cifar10'):
        self.data_directory = os.path.join(os.path.dirname(__file__), 'training_data')
        self.dataset_type = dataset_type
        self.hp_csv_path = os.path.join(self.data_directory, 'hyperparameters.csv')
        self.hp_df = pd.read_csv(self.hp_csv_path)
        self.results_directory = os.path.join(self.data_directory, f'{self.dataset_type}_training')
        os.makedirs(self.results_directory, exist_ok=True)
        self.available_hp_indices = self.get_available_hp_indices()
        self.max_budget = 19
        self.nr_features = self.hp_df.shape[1] - 2 
        self.architectures = {
            'cnn': lambda: make_cnn(num_classes=10),
            'deepsets': lambda: make_deepsets(num_classes=10),
            'transformer': lambda: make_transformer(in_dim=3, hidden_dim=64, num_heads=2, out_dim=10, dropout=0.1, num_layers=2, vit=True, patch_size=4),
            'resnet': lambda: make_resnet(num_classes=10)
        }

    def get_available_hp_indices(self):
        """ Checks and returns a list of indices for which results data is available. """
        available_indices = []
        for i in range(len(self.hp_df)):
            results_path = os.path.join(self.results_directory, f"{i}_results.json")
            if os.path.exists(results_path):
                available_indices.append(i)
            else:
                logging.warning(f"No results found for HP index {i} at {results_path}")
        return available_indices

    def get_results(self, hp_index):
        """ Retrieves the results from the specified results file if available. """
        results_path = os.path.join(self.results_directory, f"{hp_index}_results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as file:
                return json.load(file)
        return None

    def list_num_available_hp_configurations(self):
        return len(self.available_hp_indices)

    def list_num_hp_configurations(self):
        return len(self.hp_df)

    def get_hp_configs(self):
        return self.hp_df.to_dict('records')

    def get_dataloaders(self, batch_size=64):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_class = datasets.CIFAR10 if self.dataset_type == 'cifar10' else datasets.SVHN
        dataset_path = os.path.join(self.data_directory, self.dataset_type)
        dataset = dataset_class(root=dataset_path, train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def simulate(self, hp_index, epoch):
        """ Simulate the training of a model based on a hyperparameter index and epoch """
        results = self.get_results(hp_index)
        if results:
            # Filter results to find the entry for the specific epoch
            result_for_epoch = next((result for result in results if result['epoch'] == epoch), None)
            if result_for_epoch:
                return result_for_epoch
        return None  # Return None if no results are found for the specified epoch

    def get_architecture(self, hp_index):
        return self.find_config_by_hp_index(hp_index)['architecture']

    def get_pytorch_model(self, hp_index, epoch):
        config = self.hp_df.iloc[hp_index]
        model_constructor = self.architectures[config['architecture']]
        model = model_constructor()
        architecure = config['architecture']
        model_path = os.path.join(self.results_directory, f'{architecure}_{self.dataset_type}_epoch_{epoch}_{hp_index}.pt')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            return model
        else:
            raise FileNotFoundError(f"No model checkpoint found at {model_path}")

    def load_model_from_checkpoint(self, hp_index, epoch):
        """ Load a PyTorch model checkpoint for a specific configuration and epoch, then load the state dict into the appropriate model """
        return self.get_pytorch_model(hp_index, epoch)

    def get_state_dicts(self, hp_index, epoch):
        """ Retrieve state dicts for a specific model configuration and epoch """
        results = self.get_results(hp_index)
        if results:
            return [self.get_pytorch_model(hp_index, result['epoch']).state_dict() for result in results if result['epoch'] <= epoch]
        return []

    def get_best_n_configs(self, n, metric='test_accuracy', epoch=None, minimize=False):
        performance = {}
        for index in self.available_hp_indices:
            results = self.get_results(index)
            if results:
                relevant_results = [res for res in results if res['epoch'] == epoch] if epoch is not None else results
                best_result = min(relevant_results, key=lambda x: x[metric]) if minimize else max(relevant_results, key=lambda x: x[metric])
                performance[index] = (best_result[metric], best_result['epoch'])

        sorted_performance = sorted(performance.items(), key=lambda x: x[1][0], reverse=not minimize)[:n]
        return {index: {'config': self.hp_df.iloc[index].to_dict(), metric: perf[0], 'epoch': perf[1]} for index, perf in sorted_performance}

    def calculate_regret(self, best_seen_performance, metric='test_accuracy', minimize=False):
        all_configs_performance = self.get_best_n_configs(len(self.hp_df), metric, minimize=minimize)
        best_performance = min(all_configs_performance.items(), key=lambda x: x[1][metric])[1][metric] if minimize else max(all_configs_performance.items(), key=lambda x: x[1][metric])[1][metric]
        regret = best_performance - best_seen_performance
        return regret

    def get_performance_curve(self, hp_index, metric, epoch):
        """ Fetch performance curve up to a specified epoch for a given config """
        results = self.get_results(hp_index)
        if results:
            return [result[metric] for result in results if result['epoch'] <= epoch]
        return []

    def get_ground_truth_rankings(self, metric='test_accuracy', minimize=False):
        performance_data = [(index, result[metric], result['epoch']) for index, result in self.get_best_n_configs(len(self.hp_df), metric, minimize=minimize).items()]
        return sorted(performance_data, key=lambda x: x[1], reverse=not minimize)

    def find_hp_index_by_config(self, **config):
        """ Find hyperparameter index based on configuration details """
        configs = self.hp_df
        for key, value in config.items():
            configs = configs[configs[key] == value]
        if not configs.empty:
            return configs.index.tolist()
        return []

    def find_config_by_hp_index(self, hp_index):
        """ Retrieve configuration details by hyperparameter index """
        if hp_index in self.hp_df.index:
            return self.hp_df.loc[hp_index].to_dict()
        return None

    def prepare_config(self, config):
        """ Prepare a configuration for the surrogate model. dict -> np.array without hp_index """
        if config and isinstance(config, dict):
            # Create a list of values excluding the 'hp_index' and 'pretrained_index'  and 'architecture' keys
            filtered_values = [value for key, value in config.items() if key not in ('hp_index', 'pretrained_index', 'architecture')]
            return np.array(filtered_values)
        else:
            return np.array([])

    def hp_indices_to_configs(self, hp_indices):
        """
        Convert a list of hyperparameter indices to a list of configurations in np.ndarray format,
        excluding the 'hp_index' and 'pretrained_index' and 'architecture' keys from each configuration dictionary.
        
        Args:
            hp_indices: List of indices for which to fetch configurations.
        
        Returns:
            A list of NumPy arrays, each representing the configuration values excluding 'hp_index' and 'pretrained_index' and 'architecture'.
        """
        configs = [self.find_config_by_hp_index(hp_index) for hp_index in hp_indices]
        filtered_configs = [self.prepare_config(config) for config in configs]
        return filtered_configs

    def calculate_regret(self, best_seen_performance, metric='test_accuracy', epoch=None, minimize=False):
        # Get the best known configuration's performance from historical data as the ground truth
        best_configs = self.get_best_n_configs(1, metric, None, minimize)  # Get the best ever, not just current epoch
        if not best_configs:
            return None  # Handle cases where no configurations are available

        # Access the best configuration details
        best_config_key = next(iter(best_configs))  # Gets the first key in the dictionary
        best_performance = best_configs[best_config_key][metric]

        regret = best_performance - best_seen_performance
        return regret

    def get_ground_truth_rankings(self, metric='test_accuracy', epoch=None, minimize=False):
        """
        Retrieve ground truth rankings for all configurations based on a specified metric.
        If epoch is specified, it considers only that epoch; otherwise, it uses the best performance across all epochs.
        
        Args:
            metric (str): The metric to sort the configurations by, defaults to 'test_accuracy'.
            epoch (int, optional): The specific epoch to consider for rankings.
            minimize (bool): True if lower values are better (like loss), False if higher values are better (like accuracy).
        
        Returns:
            List of tuples: Each tuple contains the hyperparameter index and the metric value, sorted by the metric.
        """
        performance_data = []
        for index in self.available_hp_indices:
            results = self.get_results(index)
            if results:
                # If a specific epoch is given, filter results for that epoch only
                if epoch is not None:
                    result = next((res for res in results if res['epoch'] == epoch), None)
                    if result:
                        performance_data.append((index, result[metric]))
                else:
                    # If no specific epoch is given, find the best or worst performance across all epochs
                    if minimize:
                        best_result = min(results, key=lambda x: x[metric])
                    else:
                        best_result = max(results, key=lambda x: x[metric])
                    performance_data.append((index, best_result[metric]))

        # Sort the performance data by the metric, ascending if minimizing, descending if maximizing
        performance_data.sort(key=lambda x: x[1], reverse=not minimize)
        return [hp_index for hp_index, performance in performance_data]

if __name__ == "__main__":
    interface = PretrainedModelParkInterface('./training_data', 'cifar10')
    print(interface.list_num_available_hp_configurations())
    best_configs = interface.get_best_n_configs(5, 'test_accuracy')
    print("Best configurations:", best_configs)
    regret = interface.calculate_regret(0.95, 'test_accuracy')
    print("Regret:", regret)
    performance_curve = interface.get_performance_curve(0, 'test_accuracy')
    print("Performance curve:", performance_curve)
