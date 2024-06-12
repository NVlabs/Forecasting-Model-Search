# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.ndimage import gaussian_filter1d

def load_data(results_file):
    with open(results_file, 'r') as file:
        return json.load(file)

def smooth(y, sigma=2):
    return gaussian_filter1d(y, sigma=sigma)

def plot_results(data_files, labels):
    plt.figure(figsize=(10, 5))

    for file_idx, results_file in enumerate(data_files):
        data = load_data(results_file)

        epochs = [iteration['budget_used'] for iteration in data['iterations']]
        regrets = [iteration['regret'] for iteration in data['iterations']]

        epochs.insert(0, 0)
        regrets.insert(0, 0.8828)
        
        smoothed_regrets = smooth(np.array(regrets))

        plt.plot(epochs, smoothed_regrets, label=labels[file_idx])

    plt.title('Smoothed Regret Over Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Regret')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    data_files_as_str = ','.join(data_files)
    filename = f'smoothed_regret_over_epoch_{data_files_as_str}.png'
    plt.savefig(filename)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot experiment results from multiple files.')
    parser.add_argument('results_files', type=str, nargs='+', help='Space-separated list of JSON files containing results')
    parser.add_argument('--labels', type=str, nargs='+', help='Space-separated list of labels for each file')
    args = parser.parse_args()
    if len(args.results_files) != len(args.labels):
        raise ValueError("Number of files and labels must be the same")
    plot_results(args.results_files, args.labels)
