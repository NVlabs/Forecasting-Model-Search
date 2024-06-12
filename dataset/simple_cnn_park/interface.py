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
from scipy.stats import kendalltau
import numpy as np

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

class SimpleCNNParkInterface:
    def __init__(self):
        self.zoo_path = os.path.join(os.path.dirname(__file__), 'zoo')
        self.data_path = os.path.join(os.path.dirname(__file__), 'training_data')
        self.hp_csv_path = os.path.join(os.path.dirname(__file__), 'hyperparameters.csv')
        self.hp_df = pd.read_csv(self.hp_csv_path)
        self.available_hp_indices = self.get_available_hp_indices()
        self.max_budget = 49
        # nr features is number of columns in the hyperparameter csv (excluding the 'hp_index' and 'pretrained_index' keys)
        self.nr_features = self.hp_df.shape[1] - 2

    def get_available_hp_indices(self):
        """ Returns a list of hyperparameter indices for which data is available """
        available_indices = []
        for i in range(self.list_num_hp_configurations()):
            results_path = os.path.join(self.data_path, f'results_{i}.json')
            if os.path.exists(results_path):
                available_indices.append(i)
        return available_indices

    def list_num_available_hp_configurations(self):
        """ Returns the number of hyperparameter configurations that are available """
        return len(self.available_hp_indices)

    def list_num_hp_configurations(self):
        """ Returns the number of hyperparameter configurations """
        return len(self.hp_df)

    def get_hp_configs(self):
        """ Returns all hyperparameter configurations """
        return self.hp_df.to_dict('records')

    def get_results(self, hp_index):
        """ Load results from JSON file for a specific hyperparameter index """
        results_path = os.path.join(self.data_path, f'results_{hp_index}.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as file:
                return json.load(file)
        return None

    def simulate(self, hp_index, epoch):
        """ Simulate the training of a model based on a hyperparameter index and epoch """
        results = self.get_results(hp_index)
        if results:
            # Filter results to find the entry for the specific epoch
            result_for_epoch = next((result for result in results if result['epoch'] == epoch), None)
            if result_for_epoch:
                return result_for_epoch
        return None  # Return None if no results are found for the specified epoch

    def get_pytorch_model(self, hp_index, epoch):
        """ Load a PyTorch model checkpoint for a specific configuration and epoch, then load the state dict into SmallCNN """
        loaded_checkpoint = self.load_checkpoint(hp_index, epoch)
        model = SmallCNN()
        model.load_state_dict(loaded_checkpoint)
        return model

    def get_best_n_configs(self, n, metric='test_accuracy', epoch=None, minimize=False):
        """ Find best N configurations at a specific or across all epochs based on a metric,
            including the epoch at which the best metric was recorded. """
        performance = {}
        for i in self.available_hp_indices:
            results = self.get_results(i)
            if results:
                if epoch is not None:
                    filtered_results = [result for result in results if result['epoch'] == epoch]
                    if filtered_results:
                        performance[i] = (filtered_results[0][metric], epoch)
                else:
                    if minimize:
                        best_result = min(results, key=lambda x: x[metric])
                    else:
                        best_result = max(results, key=lambda x: x[metric])
                    performance[i] = (best_result[metric], best_result['epoch'])

        # Sorting depending on whether we are minimizing or maximizing the metric
        best_configs = sorted(performance, key=lambda x: performance[x][0], reverse=not minimize)[:n]
        return {index: {'config': self.hp_df.iloc[index].to_dict(), metric: performance[index][0], 'epoch': performance[index][1]} for index in best_configs}

    def get_state_dicts(self, hp_index, epoch):
        """ Retrieve state dicts for a specific model configuration and epoch """
        results = self.get_results(hp_index)
        if results:
            return [self.get_pytorch_model(hp_index, result['epoch']).state_dict() for result in results if result['epoch'] <= epoch]
        return []

    def get_architecture(self, hp_index):
        return "simple_cnn" # for this one, it's always simple_cnn

    def get_performance_curve(self, hp_index, metric, epoch):
        """ Fetch performance curve up to a specified epoch for a given config """
        results = self.get_results(hp_index)
        if results:
            return [result[metric] for result in results if result['epoch'] <= epoch]
        return []

    def get_features_for_epoch(self, hp_index, epoch):
        """ Retrieve extracted features for a specific model configuration and epoch """
        feature_path = os.path.join(self.data_path, f'features_{hp_index}_epoch_{epoch}.pt')
        if os.path.exists(feature_path):
            return torch.load(feature_path)
        return None

    def load_checkpoint(self, hp_index, epoch):
        """ Load a PyTorch model checkpoint for a specific configuration and epoch """
        checkpoint_path = os.path.join(self.data_path, f'model_{hp_index}_epoch_{epoch}.pt')
        if os.path.exists(checkpoint_path):
            return torch.load(checkpoint_path)
        return None

    def load_model_from_checkpoint(self, hp_index, epoch):
        """ Load a SmallCNN model from a saved checkpoint for a specific configuration and epoch """
        model = SmallCNN()
        checkpoint_path = os.path.join(self.data_path, f'model_{hp_index}_epoch_{epoch}.pt')
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            return model
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

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
            # Create a list of values excluding the 'hp_index' and 'pretrained_index' keys
            filtered_values = [value for key, value in config.items() if key not in ('hp_index', 'pretrained_index')]
            return np.array(filtered_values)
        else:
            return np.array([])

    def hp_indices_to_configs(self, hp_indices):
        """
        Convert a list of hyperparameter indices to a list of configurations in np.ndarray format,
        excluding the 'hp_index' and 'pretrained_index' keys from each configuration dictionary.
        
        Args:
            hp_indices: List of indices for which to fetch configurations.
        
        Returns:
            A list of NumPy arrays, each representing the configuration values excluding 'hp_index' and 'pretrained_index'.
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
    park = SimpleCNNParkInterface()
    print(len(park.available_hp_indices))
    print(park.get_best_n_configs(n=10, metric='test_accuracy', epoch=None))
    print(park.get_ground_truth_rankings(metric='test_accuracy', epoch=None, minimize=False))