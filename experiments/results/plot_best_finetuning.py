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

def load_latest_result(config, park, num_runs, cnn, weights):
    cnn_part = "cnn" if cnn else "no_cnn"
    weights_part = "weights" if weights else "no_weights"
    pattern = f"results_{config}_{park}_{num_runs}_{cnn_part}_{weights_part}_"
    
    files = [f for f in os.listdir('.') if os.path.isfile(f) and f.startswith(pattern)]
    
    print(f"Looking for files matching pattern: {pattern}")
    print(f"Found files: {files}")

    if not files:
        raise ValueError(f"No files found matching the pattern: {pattern}")
    
    latest_file = max(files, key=os.path.getctime)
    
    print(f"Latest file: {latest_file}")

    with open(latest_file, 'r') as f:
        return json.load(f)

def plot_fine_tuning():
    sns.set(style="whitegrid")

    configs = ["fms", "fms-gmn", "fms-nfn", "fms-flat", "dyhpo", "random-search"]
    parks = ["pretrained_model_park_svhn", "pretrained_model_park_cifar10", "simple_cnn_park"]
    num_runs = 1000
    cnn_settings = [True, False]  # True for cnn, False for no_cnn
    weight_settings = [True, False]  # True for weights, False for no_weights

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for i, park in enumerate(parks):
        for cnn in cnn_settings:
            for weights in weight_settings:
                # Skip invalid combinations
                if not cnn and not weights:
                    continue

                for j, config in enumerate(configs):
                    if (config == "fms-nfn" and "cnn" not in park) or \
                       (config == "dyhpo" and not weights) or \
                       (config == "random-search" and (not cnn or not weights)):
                        continue

                    try:
                        data = load_latest_result(config, park, num_runs, cnn, weights)

                        epochs = [iteration["budget_used"] for iteration in data["iterations"]]
                        regrets = [iteration["regret"] for iteration in data["iterations"]]
                        kendall_taus = [iteration["kendall_tau"] for iteration in data["iterations"]]

                        label = f"{config}, {'CNN' if cnn else 'No CNN'}, {'Weights' if weights else 'No Weights'}"
                        axes[0, i].plot(epochs, regrets, label=label)
                        axes[1, i].plot(epochs, kendall_taus, label=label)
                    except ValueError as e:
                        print(e)

        axes[0, i].set_title(f'Regret over time - {park}')
        axes[0, i].set_xlabel('Epochs')
        axes[0, i].set_ylabel('Regret')
        axes[0, i].legend()

        axes[1, i].set_title(f'Kendall Tau over time - {park}')
        axes[1, i].set_xlabel('Epochs')
        axes[1, i].set_ylabel('Kendall Tau')
        axes[1, i].legend()

    plt.tight_layout()
    plt.savefig('fms-for-finetuning.png')
    plt.show()

if __name__ == "__main__":
    plot_fine_tuning()
