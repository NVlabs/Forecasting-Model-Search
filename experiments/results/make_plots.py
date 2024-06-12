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
import argparse

def load_data(results_file):
    with open(results_file, 'r') as file:
        return json.load(file)

def plot_results(data_files, labels):
    # Create lists to hold the extracted data for each file
    times_list = []
    budget_used_list = []
    regrets_list = []
    kendalls_tau_list = []

    # Load and extract data from each file
    for results_file in data_files:
        data = load_data(results_file)
        times = [iteration['time'] for iteration in data['iterations']]
        budget_used = [iteration['budget_used'] for iteration in data['iterations']]
        regrets = [iteration['regret'] for iteration in data['iterations']]
        kendalls_tau = [iteration['kendall_tau'] for iteration in data['iterations']]
        
        times_list.append(times)
        budget_used_list.append(budget_used)
        regrets_list.append(regrets)
        kendalls_tau_list.append(kendalls_tau)

    # Plotting regrets over time
    plt.figure(figsize=(10, 5))
    for i in range(len(data_files)):
        plt.plot(times_list[i], regrets_list[i], marker='o', linestyle='-', label=labels[i])
    plt.title('Regret Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Regret')
    plt.legend()
    plt.grid(True)
    plt.savefig('regret_over_time.png')
    plt.show()

    # Plotting regrets over budgets
    plt.figure(figsize=(10, 5))
    for i in range(len(data_files)):
        plt.plot(budget_used_list[i], regrets_list[i], marker='o', linestyle='-', label=labels[i])
    plt.title('Regret Over Budget Used')
    plt.xlabel('Budget Used')
    plt.ylabel('Regret')
    plt.legend()
    plt.grid(True)
    plt.savefig('regret_over_budgets.png')
    plt.show()

    # Plotting Kendall's tau over time
    plt.figure(figsize=(10, 5))
    for i in range(len(data_files)):
        plt.plot(times_list[i], kendalls_tau_list[i], marker='o', linestyle='-', label=labels[i])
    plt.title("Kendall's Tau Over Time")
    plt.xlabel('Time (seconds)')
    plt.ylabel("Kendall's Tau")
    plt.legend()
    plt.grid(True)
    plt.savefig('kendalls_tau_over_time.png')
    plt.show()

    # Plotting Kendall's tau over budgets
    plt.figure(figsize=(10, 5))
    for i in range(len(data_files)):
        plt.plot(budget_used_list[i], kendalls_tau_list[i], marker='o', linestyle='-', label=labels[i])
    plt.title("Kendall's Tau Over Budget Used")
    plt.xlabel('Budget Used')
    plt.ylabel("Kendall's Tau")
    plt.legend()
    plt.grid(True)
    plt.savefig('kendalls_tau_over_budgets.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot experiment results from multiple files.')
    parser.add_argument('results_files', type=str, nargs='+', help='Space-separated list of JSON files containing results')
    parser.add_argument('--labels', type=str, nargs='+', help='Space-separated list of labels for each file')
    args = parser.parse_args()
    if len(args.results_files) != len(args.labels):
        raise ValueError("Number of files and labels must be the same")
    plot_results(args.results_files, args.labels)
