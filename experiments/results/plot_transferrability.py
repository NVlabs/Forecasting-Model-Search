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
import matplotlib.pyplot as plt
import seaborn as sns

def load_latest_result(config, park, num_runs, weight_setting):
    pattern = f"results_{config}_{park}_{num_runs}_{weight_setting}"
    
    files = [f for f in os.listdir('.') if os.path.isfile(f) and f.startswith(pattern)]
    
    print(f"Looking for files matching pattern: {pattern}")
    print(f"Found files: {files}")

    if not files:
        raise ValueError(f"No files found matching the pattern: {pattern}")
    
    latest_file = max(files, key=os.path.getctime)
    
    print(f"Latest file: {latest_file}")

    with open(latest_file, 'r') as f:
        return json.load(f)

def plot_zero_shot():
    sns.set(style="whitegrid")

    configs = ["fms"]
    parks = ["pretrained_model_park_transfer_svhn", "pretrained_model_park_transfer_cifar10", "simple_cnn_park_transfer"]
    num_runs = 1000
    weight_setting = "cnn_weights"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, park in enumerate(parks):
        for j, config in enumerate(configs):
            try:
                data = load_latest_result(config, park, num_runs, weight_setting)

                epochs = [iteration["budget_used"] for iteration in data["iterations"]]
                regrets = [iteration["regret"] for iteration in data["iterations"]]
                kendall_taus = [iteration["kendall_tau"] for iteration in data["iterations"]]

                axes[i].plot(epochs, regrets, label=f'Regret - {config}')
                axes[i].plot(epochs, kendall_taus, label=f'Kendall Tau - {config}')
            except ValueError as e:
                print(e)

        axes[i].set_title(f'Zero-shot performance - {park}')
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel('Metric Value')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig('zero-shot-surrogate.png')
    plt.show()

if __name__ == "__main__":
    plot_zero_shot()
