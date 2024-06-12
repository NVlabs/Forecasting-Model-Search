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
import time
import json
from utils import get_surrogate_model, get_data_interface
from collections import defaultdict
import random
import numpy as np
import torch
from scipy.stats import kendalltau

class ExperimentController:
    """Manages all data related to an experiment run."""

    def __init__(self, experiment_id, budget_limit, experiment_type, park, seed, use_cnn, use_weights):
        # self.seed(seed)
        self.budget_used = 0
        self.budget_limit = budget_limit

        # index -> budget we last evaluated the config at
        self.used_indices = defaultdict(int)

        self.experiment_type = experiment_type
        self.experiment_id = experiment_id
        self.park = park

        self.data_interface = get_data_interface(park)
        self.surrogate_model = get_surrogate_model(experiment_type, park, use_cnn, use_weights)

        self.no_improvement_threshold = int(self.data_interface.max_budget + 0.2 * self.data_interface.max_budget)
        self.no_improvement_patience = 0

        self.start_time = time.time()
        self.best_model_config = None
        self.results_dir = './results'

        self.use_cnn = use_cnn
        self.use_weights = use_weights

        self.results = {
            'iterations': [],
            'best_config': None,
            # adjust based on whether you are minimizing or maximizing (e.g. loss or accuracy)
            'best_performance': float('-inf'),
            'best_performance_history': [],
            'timestamps': []  # timestamps when the best changes
        }

    def suggest_config(self):
        print("ExperimentController.suggest_config")
        hp_index, additional_budget = self.surrogate_model.suggest_next_hp_index(self.used_indices)
        config_details = self.data_interface.find_config_by_hp_index(hp_index)
        print(f"Config selected to observe: {config_details} with additional budget {additional_budget} (current budget used for config {hp_index}: {self.used_indices[hp_index]})")
        return hp_index, additional_budget

    def observe(self, hp_index, additional_budget):
        print(f"ExperimentController.observe(hp_index={hp_index}, additional_budget={additional_budget})")
        if additional_budget + self.budget_used > self.budget_limit:
            print(f"Not enough budget left to observe config {hp_index} with additional budget {additional_budget}. Using the remaining budget instead.")
            additional_budget = self.budget_limit - self.budget_used    

        budget = min(self.data_interface.max_budget, self.used_indices[hp_index] + additional_budget)

        self.budget_used += additional_budget
        print(f"Budget used so far: {self.budget_used}/{self.budget_limit}")

        eval_results = self.data_interface.simulate(hp_index, budget)
        self.used_indices[hp_index] = budget
        print(f"Evaluation results for config {hp_index} at budget {budget}: {eval_results}")

        if eval_results is None:
            print(f"No results available for config {hp_index} at budget {budget}")
            return None

        # Extract the performance metric
        current_performance = eval_results['test_accuracy'] if 'test_accuracy' in eval_results else None

        # Update the current best configuration and performance if the current one is better
        if current_performance is not None and (self.results['best_performance'] == float('-inf') or current_performance > self.results['best_performance']):
            self.results['best_performance'] = current_performance
            self.results['best_config'] = hp_index
            self.results['best_config_details'] = self.data_interface.find_config_by_hp_index(hp_index)
            self.results['best_budget'] = budget
            self.results['best_performance_history'].append({
                'config': hp_index,
                'performance': current_performance,
                'time': self.get_wallclock_time(),
                'additional_budget': additional_budget,
                'budget_evaluated_at': budget,
                'budget_used': self.budget_used,
            })
            print(f"New best performance found: {current_performance} at config {hp_index} at budget {budget}")

        # Calculate regret using the best known configuration from historical data
        current_regret = self.data_interface.calculate_regret(self.results['best_performance'], metric='test_accuracy', epoch=budget)
        # predicted_rankings = self.surrogate_model.predict_rankings(self.used_indices)
        # current_kendall_tau = self.data_interface.calculate_kendall_tau(predicted_rankings, metric='test_accuracy', epoch=budget)
        ktau, p_value = self.calculate_kendall_tau()

        self.results['iterations'].append({
            'config': hp_index,
            'config_details': self.data_interface.find_config_by_hp_index(hp_index),
            'additional_budget': additional_budget,
            'budget_evaluated_at': budget,
            'budget_used': self.budget_used,
            'evaluation_results': eval_results,
            'regret': current_regret,
            'kendall_tau': ktau,
            'p_value': p_value,
            'time': self.get_wallclock_time()
        })

        return eval_results


    def update_performance(self, current_performance, config_to_observe):
        print(f"ExperimentController.update_performance(current_performance={current_performance}, config_to_observe={config_to_observe})")
        if current_performance is not None and current_performance > self.results['best_performance']:
            self.results['best_performance'] = current_performance
            self.results['best_config'] = config_to_observe
            self.results['time_stamps'].append(self.get_wallclock_time())
            self.results['best_performance_history'].append({
                'config': config_to_observe,
                'performance': current_performance,
                'time': self.get_wallclock_time(),
            })

    def update_surrogate_model(self, config_to_observe, eval_results, metric='test_accuracy'):
        """Update our surrogate, backpropagating loss computed from prediction vs. truth"""
        print(f"ExperimentController.update_surrogate_model(config_to_observe={config_to_observe}, eval_results={eval_results}, metric={metric})")
        self.surrogate_model.train(self.used_indices)
        current_performance = eval_results[metric]
        if current_performance > self.results['best_performance']:
            self.results['best_performance'] = current_performance
            self.results['best_config'] = config_to_observe
            self.results['best_performance_history'].append(current_performance)
            self.results['time_stamps'].append(self.get_wallclock_time())

    def has_converged(self):
        return self.budget_used >= self.budget_limit

    def save_results(self):
        os.makedirs(self.results_dir, exist_ok=True)
        results_path = os.path.join(self.results_dir, f"results_{self.experiment_id}.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        pass

    def get_wallclock_time(self):
        return time.time() - self.start_time

    def seed(self, seed_value=42):
        """Set seed for reproducibility."""
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)

    def calculate_kendall_tau(self):
        """
        Calculate the Kendall Tau correlation coefficient between predicted rankings
        from the surrogate model and the ground truth rankings from the data interface.

        Args:
            data_interface (SimpleCNNParkInterface): The interface to the park with the ground truth data.
            surrogate_model (SurrogateModel): The surrogate model used for predictions.
            used_indices (dict): A dictionary of indices that have been used so far.

        Returns:
            float: The Kendall Tau correlation coefficient.
        """
        predicted_rankings = self.surrogate_model.predict_rankings(self.used_indices)
        ground_truth_rankings = self.data_interface.get_ground_truth_rankings()
        print(len(predicted_rankings), len(ground_truth_rankings))
        tau, p_value = kendalltau(predicted_rankings, ground_truth_rankings)
        return tau, p_value


def get_experiment_id(experiment_type, park, budget_limit, use_cnn, use_weights):
    cnn_status = "no_cnn" if not use_cnn else "cnn"
    weights_status = "no_weights" if not use_weights else "weights"
    return f"{experiment_type}_{park}_{budget_limit}_{cnn_status}_{weights_status}_{int(time.time())}"

def fms(budget_limit=49, experiment_type="fms", park="simple_cnn_park", seed=14, use_cnn=True, use_weights=True):
    experiment_id = get_experiment_id(experiment_type, park, budget_limit, use_cnn, use_weights)
    print(
        f"[{experiment_id}] Starting {experiment_type} with {park} and a budget limit of {budget_limit} epochs."
    )

    controller = ExperimentController(
        experiment_id=experiment_id,
        budget_limit=budget_limit,
        experiment_type=experiment_type,
        park=park,
        seed=seed,
        use_cnn=use_cnn,
        use_weights=use_weights
    )
    i = 0

    while not controller.has_converged():
        print(
            f"Iteration {i}, wallclock time {controller.get_wallclock_time()}, budget used {controller.budget_used}"
        )
        try:
            config_to_observe, budget = controller.suggest_config()
        except Exception as e:
            print(f"Exception: {e}")
            break
        eval_results = controller.observe(config_to_observe, budget)
        print(f"Evaluation results: {eval_results}")
        controller.update_surrogate_model(config_to_observe, eval_results, 'test_accuracy')
        i += 1

    print("Run complete, now saving results.")
    controller.save_results()
    return controller.best_model_config

if __name__ == "__main__":
    import argparse
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--budget_limit", type=int, default=1000)
    # can be fms (permutation equivariant gmn), fms-nfn (permutation equivariant), fms-flat (not permutation equivariant), dyhpo, random-search
    argparse.add_argument("--experiment_type", type=str, default="fms")
    # can be:
    # simple_cnn_park (MNIST on one-cnn-benchmark),
    # simple_cnn_park_transfer (MNIST on one-cnn-benchmark, using a shared checkpoint for the GMN with svhn and cifar10 and GMN on MNIST),
    # pretrained_model_park_cifar10 (cifar10 on pretrained_model_park),
    # pretrained_model_park_svhn (svhn on pretrained_model_park),
    # pretrained_model_park_transfer_cifar10 (cifar10 on pretrained_model_park, using a shared checkpoint for the GMN with svhn and cifar10 and GMN on MNIST),
    # pretrained_model_park_transfer_svhn (svhn on pretrained_model_park using a shared checkpoint for the GMN with svhn and cifar10 and GMN on MNIST)
    argparse.add_argument("--park", type=str, default="simple_cnn_park")
    argparse.add_argument("--seed", type=int, default=14)
    argparse.add_argument("--ablate-cnn", action="store_true", default=False)
    argparse.add_argument("--ablate-weights", action="store_true", default=False)
    # the reason you'd want to do this is b/c the saved checkpoint gets stronger with times. in other words, future iterations will have stronger priors.
    argparse.add_argument("--num-cycles", type=int, default=1)
    args = argparse.parse_args()
    for i in range(args.num_cycles):
        fms(budget_limit=args.budget_limit, experiment_type=args.experiment_type, park=args.park, seed=args.seed, use_cnn=(not args.ablate_cnn), use_weights=(not args.ablate_weights))