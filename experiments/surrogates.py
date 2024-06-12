# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
import torch.nn as nn
import gpytorch
import torch.nn.functional as F
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
import os
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Dict, Tuple, Optional, List
import numpy as np
from dataset.pretrained_model_park.interface import PretrainedModelParkInterface
from dataset.simple_cnn_park.interface import SimpleCNNParkInterface
from nfn.layers import NPLinear, HNPPool, TupleOp
from torch.utils.data import default_collate
from nfn.common import state_dict_to_tensors, WeightSpaceFeatures, network_spec_from_wsfeat
from scipy.stats import norm
import copy
from copy import deepcopy
from nfn.helpers import make_cnn, sample_perm
from torch.nn.utils.rnn import pad_sequence
from gmn.feature_extractor_gmn import GraphPredGen
from gmn.graph_construct.net_makers import sd_to_net
from gmn.graph_construct.model_arch_graph import arch_to_graph, sequential_to_arch
from torch_geometric.data import Batch, Data

def get_data_interface(park):
    if park == "simple_cnn_park" or park == "simple_cnn_park_transfer":
        return SimpleCNNParkInterface()
    elif park == "pretrained_model_park_cifar10" or park == "pretrained_model_park_transfer_cifar10":
        return PretrainedModelParkInterface("cifar10")
    elif park == "pretrained_model_park_svhn" or park == "pretrained_model_park_transfer_svhn":
        return PretrainedModelParkInterface("svhn")
    else:
        raise ValueError(f"Unknown park: {park}")

class NfnFeatureExtractor(torch.nn.Module):
    """Feature extractor using permutation equivariant neural functionals."""

    def __init__(self, configuration):
        super(NfnFeatureExtractor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configuration = configuration
        self.nr_layers = configuration['nr_layers']
        self.act_func = nn.LeakyReLU()
        initial_features = configuration['nr_initial_features'] + 1
        self.fc1 = nn.Linear(initial_features, configuration['layer1_units'])
        self.bn1 = nn.BatchNorm1d(configuration['layer1_units'])
        for i in range(2, self.nr_layers):
            setattr(
                self,
                f'fc{i + 1}',
                nn.Linear(configuration[f'layer{i - 1}_units'], configuration[f'layer{i}_units']),
            )
            setattr(
                self,
                f'bn{i + 1}',
                nn.BatchNorm1d(configuration[f'layer{i}_units']),
            )
        setattr(
            self,
            f'fc{self.nr_layers}',
            nn.Linear(
                configuration[f'layer{self.nr_layers - 1}_units'] +
                (configuration['cnn_nr_channels'] if self.configuration['use_cnn'] else 0) + # accounting for the learning curve features
                (configuration['state_dict_nr_channels'] if self.configuration['use_weights'] else 0),  # accounting for the state dict features
                configuration[f'layer{self.nr_layers}_units']
            ),
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, kernel_size=(configuration['cnn_kernel_size'],), out_channels=4),
            nn.AdaptiveMaxPool1d(1),
        )

    def make_nfn(self, network_spec, nfn_channels=32):
        return nn.Sequential(
            # io_embed: encode the input and output dimensions of the weight space feature
            NPLinear(network_spec, 1, nfn_channels, io_embed=True),
            TupleOp(nn.ReLU()),
            NPLinear(network_spec, nfn_channels, nfn_channels, io_embed=True),
            TupleOp(nn.ReLU()),
            HNPPool(network_spec),
            nn.Flatten(start_dim=-2),
            nn.Linear(nfn_channels * HNPPool.get_num_outs(network_spec), self.configuration['state_dict_nr_channels'])
        ).to(self.device)

    def process_state_dicts(self, state_dicts, architectures):
        wts_and_bs = []
        for state_dict in state_dicts:
            wts_and_bs.append(state_dict_to_tensors(state_dict))
        # Here we manually collate weights and biases (stack into batch dim).
        # When using a dataloader, the collate is done automatically.
        wtfeat = WeightSpaceFeatures(*default_collate(wts_and_bs)).to(self.device)
        network_spec = network_spec_from_wsfeat(wtfeat)
        nfn = self.make_nfn(network_spec)
        out = nfn(wtfeat)
        return out

    def forward(self, x, budgets, learning_curves, state_dicts, architectures):
        # add an extra dimensionality for the budget
        # making it nr_rows x 1.
        budgets = torch.unsqueeze(budgets, dim=1)
        # concatenate budgets with examples
        x = torch.cat((x, budgets), dim=1)

        x = self.fc1(x)
        x = self.act_func(self.bn1(x))

        for i in range(2, self.nr_layers):
            x = self.act_func(
                getattr(self, f'bn{i}')(
                    getattr(self, f'fc{i}')(
                        x
                    )
                )
            )

        if self.configuration['use_cnn']:
            # add an extra dimensionality for the learning curve
            # making it nr_rows x 1 x lc_values.
            learning_curves = torch.unsqueeze(learning_curves, 1)
            lc_features = self.cnn(learning_curves)
            # revert the output from the cnn into nr_rows x nr_kernels.
            lc_features = torch.squeeze(lc_features, 2)

            # put learning curve features into the last layer along with the higher level features.
            x = torch.cat((x, lc_features), dim=1)
        
        if self.configuration['use_weights']:
            state_dict_features = self.process_state_dicts(state_dicts, architectures)
            x = torch.cat((x, state_dict_features), dim=1)

        x = self.act_func(getattr(self, f'fc{self.nr_layers}')(x))

        return x

class GmnFeatureExtractor(torch.nn.Module):
    def __init__(self, configuration):
        super(GmnFeatureExtractor, self).__init__()
        self.configuration = configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nr_layers = configuration['nr_layers']
        self.act_func = nn.LeakyReLU()
        initial_features = configuration['nr_initial_features'] + 1
        self.fc1 = nn.Linear(initial_features, configuration['layer1_units'])
        self.bn1 = nn.BatchNorm1d(configuration['layer1_units'])
        self.gnn = GraphPredGen(
            gnn_type='empnn_dit',
            hidden_dim=configuration['hidden_dim'],
            num_gnn_layers=configuration['num_gnn_layers'],
            out_dim=configuration['gmn_nr_channels'],
            pre_encoder=True,  # assume pre-encoder is needed; adjust as necessary
            readout_layers=2,  # the number of readout layers
            pool_type='ds'  # pooling type; adjust based on your requirements
        ).to(self.device)

        for i in range(2, self.nr_layers):
            setattr(
                self,
                f'fc{i}',
                nn.Linear(configuration[f'layer{i - 1}_units'], configuration[f'layer{i}_units']),
            )
            setattr(
                self,
                f'bn{i}',
                nn.BatchNorm1d(configuration[f'layer{i}_units']),
            )
        
        setattr(
            self,
            f'fc{self.nr_layers}',
            nn.Linear(
                configuration[f'layer{self.nr_layers - 1}_units'] +
                (configuration['cnn_nr_channels'] if self.configuration['use_cnn'] else 0) +  # From learning curve features
                (configuration['gmn_nr_channels'] if self.configuration['use_weights'] else 0),  # From GNN features
                configuration[f'layer{self.nr_layers}_units']
            ),
        )

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, kernel_size=(configuration['cnn_kernel_size'],), out_channels=4),
            nn.AdaptiveMaxPool1d(1),
        )

    def process_state_dicts(self, state_dicts, architectures):
        graph_data = self.convert_state_dicts_to_graph(state_dicts, architectures).to(self.device)
        # Ensure data is in the correct format
        graph_data.x = graph_data.x.float()  # Convert node features to float
        graph_data.edge_attr = graph_data.edge_attr.float()  # Convert edge attributes to float

        graph_features = self.gnn(
            x=graph_data.x, 
            edge_index=graph_data.edge_index, 
            edge_attr=graph_data.edge_attr, 
            batch=graph_data.batch
        )

        if graph_features.dim() == 1:  # If it's a scalar per graph, expand or replicate
            graph_features = graph_features.view(-1, 1).repeat(1, self.expected_feature_dim)  # Adjust `expected_feature_dim` as needed

        return graph_features


    def convert_state_dicts_to_graph(self, state_dicts, architectures):
        graphs = []
        for state_dict, architecture in zip(state_dicts, architectures):
            model = sd_to_net(state_dict, architecture)
            arch = sequential_to_arch(model)
            x, edge_index, edge_attr = arch_to_graph(arch)
            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            if not isinstance(graph, Data):
                raise TypeError(f"Expected torch_geometric.data.Data, got {type(graph)}")
            graphs.append(graph)
        
        return Batch.from_data_list(graphs)

    def forward(self, x, budgets, learning_curves, state_dicts, architectures):
        budgets = torch.unsqueeze(budgets, dim=1)
        x = torch.cat((x, budgets), dim=1)
        x = self.fc1(x)
        x = self.act_func(self.bn1(x))

        for i in range(2, self.nr_layers):
            x = self.act_func(
                getattr(self, f'bn{i}')(
                    getattr(self, f'fc{i}')(
                        x
                    )
                )
            )
        
        if self.configuration['use_cnn']:
            learning_curves = torch.unsqueeze(learning_curves, 1)
            lc_features = self.cnn(learning_curves)
            lc_features = torch.squeeze(lc_features, 2)
            x = torch.cat((x, lc_features), dim=1)

        if self.configuration['use_weights']:
            state_dict_features = self.process_state_dicts(state_dicts, architectures)
            x = torch.cat((x, state_dict_features), dim=1)

        x = self.act_func(getattr(self, f'fc{self.nr_layers}')(x))

        return x

class FlatFeatureExtractor(torch.nn.Module):
    def __init__(self, configuration):
        super(FlatFeatureExtractor, self).__init__()
        self.configuration = configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nr_layers = configuration['nr_layers']
        self.act_func = nn.LeakyReLU()
        initial_features = configuration['nr_initial_features'] + 1
        self.fc1 = nn.Linear(initial_features, configuration['layer1_units'])
        self.bn1 = nn.BatchNorm1d(configuration['layer1_units'])

        # Additional layers between flattened state dict features and final layers
        self.flat_params_fc = nn.Linear(configuration['flattened_params_units'], configuration['state_dict_nr_channels'])
        
        for i in range(2, self.nr_layers):
            setattr(
                self,
                f'fc{i}',
                nn.Linear(configuration[f'layer{i - 1}_units'], configuration[f'layer{i}_units']),
            )
            setattr(
                self,
                f'bn{i}',
                nn.BatchNorm1d(configuration[f'layer{i}_units']),
            )
        
        # Modify the last fully connected layer to accommodate the additional flattened state dict features
        setattr(
            self,
            f'fc{self.nr_layers}',
            nn.Linear(
                configuration[f'layer{self.nr_layers - 1}_units'] +
                (configuration['cnn_nr_channels'] if self.configuration['use_cnn'] else 0) +  # From learning curve features
                (configuration['state_dict_nr_channels'] if self.configuration['use_weights'] else 0),  # From state dict features
                configuration[f'layer{self.nr_layers}_units']
            ),
        )

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, kernel_size=(configuration['cnn_kernel_size'],), out_channels=4),
            nn.AdaptiveMaxPool1d(1),
        )

    def process_state_dicts(self, state_dicts, architectures):
        # Flatten state dicts into a single vector per dict
        flat_params = []
        for state_dict in state_dicts:
            params = []
            for param in state_dict.values():
                params.append(param.view(-1))
            flat_params.append(torch.cat(params))

        # Pad and batch the flattened parameters
        flat_params = pad_sequence(flat_params, batch_first=True)
        flat_params = flat_params.to(self.device)

        # Process flattened parameters with a simple FC layer
        flat_params = self.flat_params_fc(flat_params)
        return flat_params

    def forward(self, x, budgets, learning_curves, state_dicts, architectures):
        budgets = torch.unsqueeze(budgets, dim=1)
        x = torch.cat((x, budgets), dim=1)
        x = self.fc1(x)
        x = self.act_func(self.bn1(x))

        for i in range(2, self.nr_layers):
            x = self.act_func(
                getattr(self, f'bn{i}')(
                    getattr(self, f'fc{i}')(
                        x
                    )
                )
            )
        
        if self.configuration['use_cnn']:
            # add an extra dimensionality for the learning curve
            # making it nr_rows x 1 x lc_values.
            learning_curves = torch.unsqueeze(learning_curves, 1)
            lc_features = self.cnn(learning_curves)
            # revert the output from the cnn into nr_rows x nr_kernels.
            lc_features = torch.squeeze(lc_features, 2)

            # put learning curve and state_dict features into the last layer along with the higher level features.
            x = torch.cat((x, lc_features), dim=1)

        if self.configuration['use_weights']:
            state_dict_features = self.process_state_dicts(state_dicts, architectures)
            x = torch.cat((x, state_dict_features), dim=1)

        x = self.act_func(getattr(self, f'fc{self.nr_layers}')(x))

        return x

class DyHpoFeatureExtractor(nn.Module):
    """
    The feature extractor that is part of the deep kernel.
    """
    def __init__(self, configuration):
        super(DyHpoFeatureExtractor, self).__init__()
        self.configuration = configuration
        self.nr_layers = configuration['nr_layers']
        self.act_func = nn.LeakyReLU()
        initial_features = configuration['nr_initial_features'] + 1
        self.fc1 = nn.Linear(initial_features, configuration['layer1_units'])
        self.bn1 = nn.BatchNorm1d(configuration['layer1_units'])
        for i in range(2, self.nr_layers):
            setattr(
                self,
                f'fc{i + 1}',
                nn.Linear(configuration[f'layer{i - 1}_units'], configuration[f'layer{i}_units']),
            )
            setattr(
                self,
                f'bn{i + 1}',
                nn.BatchNorm1d(configuration[f'layer{i}_units']),
            )
        setattr(
            self,
            f'fc{self.nr_layers}',
            nn.Linear(
                configuration[f'layer{self.nr_layers - 1}_units'] +
                (configuration['cnn_nr_channels'] if self.configuration['use_cnn'] else 0),  # accounting for the learning curve features
                configuration[f'layer{self.nr_layers}_units']
            ),
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, kernel_size=(configuration['cnn_kernel_size'],), out_channels=4),
            nn.AdaptiveMaxPool1d(1),
        )

    def forward(self, x, budgets, learning_curves, state_dicts=None, architectures=None):
        # add an extra dimensionality for the budget
        # making it nr_rows x 1.
        budgets = torch.unsqueeze(budgets, dim=1)
        # concatenate budgets with examples
        x = torch.cat((x, budgets), dim=1)
        x = self.fc1(x)
        x = self.act_func(self.bn1(x))

        for i in range(2, self.nr_layers):
            x = self.act_func(
                getattr(self, f'bn{i}')(
                    getattr(self, f'fc{i}')(
                        x
                    )
                )
            )

        if self.configuration['use_cnn']:
            # add an extra dimensionality for the learning curve
            # making it nr_rows x 1 x lc_values.
            learning_curves = torch.unsqueeze(learning_curves, 1)
            lc_features = self.cnn(learning_curves)
            # revert the output from the cnn into nr_rows x nr_kernels.
            lc_features = torch.squeeze(lc_features, 2)

            # put learning curve features into the last layer along with the higher level features.
            x = torch.cat((x, lc_features), dim=1)

        x = self.act_func(getattr(self, f'fc{self.nr_layers}')(x))
        return x

class GPRegressionModel(gpytorch.models.ExactGP):
    """A simple GP model."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ):
        """
        Args:
            train_x: The initial train examples for the GP.
            train_y: The initial train labels for the GP.
            likelihood: The likelihood to be used.
        """
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def get_feature_extractor(experiment_type):
    if experiment_type == "fms":
        return GmnFeatureExtractor
    elif experiment_type == "fms-nfn":
        return NfnFeatureExtractor
    elif experiment_type == "fms-flat":
        return FlatFeatureExtractor
    elif experiment_type == "dyhpo":
        return DyHpoFeatureExtractor
    elif experiment_type == "random-search":
        return RandomSearchSurrogateModel

class FmsSurrogateModel():
    """The GP + Feature Extractor model."""
    def __init__(self, experiment_type, park, use_cnn, use_weights):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_type = experiment_type
        self.park = park
        self.num_permutations = 50

        self.data_interface = get_data_interface(park)
        self.nr_features = self.data_interface.nr_features
        self.surrogate_config = {
            'nr_layers': 2,
            'nr_initial_features': self.nr_features,
            'layer1_units': 64,
            'layer2_units': 128,
            'cnn_nr_channels': 4,
            'flattened_params_units': 4970,
            'state_dict_nr_channels': 128,
            'gmn_nr_channels': 32, # out_dim of GMN
            'cnn_kernel_size': 3,
            'batch_size': 64,
            'nr_epochs': 1000,
            'nr_patience_epochs': 10,
            'learning_rate': 0.001,
            'use_cnn': use_cnn,
            'use_weights': use_weights,
            'hidden_dim': 32,
            'num_gnn_layers': 4,
        }

        self.batch_size = self.surrogate_config['batch_size']
        self.nr_epochs = self.surrogate_config['nr_epochs']
        self.early_stopping_patience = self.surrogate_config['nr_patience_epochs']
        self.refine_epochs = 50

        self.feature_extractor = get_feature_extractor(experiment_type)(self.surrogate_config)
        self.gp, self.likelihood, self.mll = self.init_gp(self.surrogate_config[f'layer{self.feature_extractor.nr_layers}_units'])

        self.gp.to(self.device)
        self.likelihood.to(self.device)
        self.feature_extractor.to(self.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.gp.parameters(), 'lr': self.surrogate_config['learning_rate']},
            {'params': self.feature_extractor.parameters(), 'lr': self.surrogate_config['learning_rate']}],
        )

        # the number of initial points for which we will retrain fully from scratch
        # This is basically equal to the dimensionality of the search space + 1.
        self.initial_nr_points = 10
        # keeping track of the total hpo iterations. It will be used during the optimization
        # process to switch from fully training the model, to refining.
        self.iterations = 0

        self.num_initial_configurations = 20
        self.initial_random_configurations = [random.choice(self.data_interface.available_hp_indices) for _ in range(self.num_initial_configurations)]
        self.num_random_configruations_sampled = 0
        self.fantasize_step = 1

        self.results_dir = ''
        self.use_cnn = use_cnn
        self.use_weights = use_weights

        self.checkpoint_path = self.get_checkpoint_path()

        # flag for when the optimization of the model should start from scratch.
        self.restart = False if self.if_checkpoint_exists() else True
        if self.if_checkpoint_exists():
            self.load_checkpoint()

    def permute_tensor(self, tensor):
        """Randomly permute the elements of a tensor."""
        if tensor.ndimension() == 0:
            return tensor # no permutation needed
        elif tensor.ndimension() == 1:
            return tensor[torch.randperm(tensor.size(0))]
        else:
            # Recursively permute along each dimension
            permuted_tensor = tensor.clone()
            for i in range(tensor.size(0)):
                permuted_tensor[i] = self.permute_tensor(tensor[i])
            return permuted_tensor[torch.randperm(tensor.size(0))]

    def permute_state_dict(self, state_dict):
        """Randomly permute the parameters within each tensor of the state dict."""
        permuted_state_dict = {}
        for key, value in state_dict.items():
            permuted_state_dict[key] = self.permute_tensor(value)
        return permuted_state_dict

    def generate_permuted_state_dicts(self, state_dict, num_permutations):
        """Generate permuted state dicts."""
        permuted_state_dicts = []
        for _ in range(num_permutations):
            permuted_state_dict = self.permute_state_dict(state_dict)
            permuted_state_dicts.append(permuted_state_dict)
        return permuted_state_dicts

    def get_checkpoint_path(self):
        if 'transfer' in self.park:
            return os.path.join(self.results_dir, 'checkpoints', f'{self.experiment_type}_transfer')
        else:
            base_path = os.path.join(self.results_dir, 'checkpoints', f'{self.experiment_type}_{self.park}')

        def append_checkpoint_path():
            if not self.use_cnn and not self.use_weights:
                return "_no_cnn_no_weights"
            if not self.use_cnn:
                return "_no_cnn"
            if not self.use_weights:
                return "_no_weights"
            return ""
            
        final_path = base_path + append_checkpoint_path()
        os.makedirs(final_path, exist_ok=True)
        checkpoint_file = os.path.join(
            final_path,
            'checkpoint.pth'
        )
        return checkpoint_file
        

    def init_gp(
        self,
        train_size: int,
    ) -> Tuple[GPRegressionModel, gpytorch.likelihoods.GaussianLikelihood, gpytorch.mlls.ExactMarginalLogLikelihood]:
        """
        Called when the surrogate is first initialized or restarted.

        Args:
            train_size: The size of the current training set.

        Returns:
            model, likelihood, mll - The GP model, the likelihood and
                the marginal likelihood.
        """
        train_x = torch.ones(train_size, train_size).to(self.device)
        train_y = torch.ones(train_size).to(self.device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        model = GPRegressionModel(train_x=train_x, train_y=train_y, likelihood=likelihood).to(self.device)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(self.device)
        return model, likelihood, mll

    def restart_optimization(self):
        """
        Restart the surrogate model from scratch.
        """
        self.feature_extractor = get_feature_extractor(self.experiment_type)(self.surrogate_config).to(self.device)
        self.gp, self.likelihood, self.mll = self.init_gp(self.surrogate_config[f'layer{self.feature_extractor.nr_layers}_units'])

        self.optimizer = torch.optim.Adam([
            {'params': self.gp.parameters(), 'lr': self.surrogate_config['learning_rate']},
            {'params': self.feature_extractor.parameters(), 'lr': self.surrogate_config['learning_rate']}],
        )

    def suggest_next_hp_index(self, used_indices):
        if self.num_random_configruations_sampled < self.num_initial_configurations:
            print("Not enough random configurations sampled yet, returning a random one.")
            next_hp_index = self.initial_random_configurations[self.num_random_configruations_sampled]
            self.num_random_configruations_sampled += 1
            return next_hp_index, 1
        else:
            mean_predictions, std_predictions, hp_indices, budgets = self.predict(used_indices)
            best_prediction_index = self.find_suggested_config(
                mean_predictions,
                std_predictions,
                budgets,
                used_indices
            )
            return hp_indices[best_prediction_index], used_indices[hp_indices[best_prediction_index]] + self.fantasize_step

    def if_checkpoint_exists(self):
        """Check if a checkpoint exists for the current model configuration."""
        checkpoint_path = self.get_checkpoint_path()
        return os.path.exists(checkpoint_path)
        
    def train(self, used_indices):
        """Train the surrogate model."""
        data = self._prepare_dataset_and_budgets(used_indices)
        print(f'Started training the model')

        self.model_train(
            data,
            load_checkpoint=self.if_checkpoint_exists(),
        )
        torch.cuda.empty_cache()


    def predict(self, used_indices) -> Tuple[np.ndarray, np.ndarray, List, List]:
        """
        Predict the performances of the hyperparameter configurations
        as well as the standard deviations based on the surrogate model.

        Returns:
            mean_predictions, std_predictions, hp_indices, non_scaled_budgets:
                The mean predictions and the standard deviations over
                all model predictions for the given hyperparameter
                configurations with their associated indices, scaled and
                non-scaled budgets.
        """
        print(f"Number of used_indices: {len(used_indices)}")
        configurations, hp_indices, budgets, learning_curves, state_dicts, architectures = self.generate_candidate_configurations(used_indices)
        budgets = np.array(budgets, dtype=np.single)
        non_scaled_budgets = copy.deepcopy(budgets)
        # scale budgets to [0, 1]
        budgets = budgets / self.data_interface.max_budget

        configurations = np.array(configurations, dtype=np.single)
        configurations = torch.tensor(configurations)
        configurations = configurations.to(device=self.device)

        budgets = torch.tensor(budgets)
        budgets = budgets.to(device=self.device)

        learning_curves = self.patch_curves_to_same_length(learning_curves)
        learning_curves = np.array(learning_curves, dtype=np.single)
        learning_curves = torch.tensor(learning_curves)
        learning_curves = learning_curves.to(device=self.device)

        train_data = self._prepare_dataset_and_budgets(used_indices)
        test_data = {
            'X_test': configurations,
            'test_budgets': budgets,
            'test_curves': learning_curves,
            'state_dicts': state_dicts,
            'architectures': architectures
        }

        mean_predictions, std_predictions = self.model_predict(train_data, test_data)
        return mean_predictions, std_predictions, hp_indices, non_scaled_budgets


    def predict_rankings(self, used_indices):
        """
        Predict and rank all hyperparameter configurations based on the GP model's estimated performance,
        including simulations for configurations that have reached the maximum budget or are incomplete.

        Args:
            used_indices: Dictionary mapping hyperparameter indices to the latest budget they were used at.

        Returns:
            List of hyperparameter indices sorted by predicted or simulated performance.
        """
        # Predict rankings for existing data
        mean_predictions, std_predictions, hp_indices, non_scaled_budgets = self.predict(used_indices)

        print(len(mean_predictions), len(hp_indices))

        # Get all HP indices and find missing ones
        all_hp_indices = set(range(len(self.data_interface.available_hp_indices)))
        predicted_indices = set(hp_indices)
        missing_indices = all_hp_indices - predicted_indices

        # Simulate results for missing configurations
        simulated_results = []
        for index in missing_indices:
            print("missing:")
            print(index)
            eval_reuslts = self.data_interface.simulate(index, self.data_interface.max_budget)
            simulated_results.append((index, eval_reuslts['test_accuracy']))

        print(len(simulated_results), len(hp_indices))
        print(simulated_results)

        # Combine predictions and simulations
        combined_results = list(zip(hp_indices, mean_predictions)) + simulated_results

        # Sort all configurations by their performance, highest first
        combined_results.sort(key=lambda x: x[1], reverse=True)

        return [idx for idx, _ in combined_results]

    def model_predict(
        self,
        train_data: Dict[str, torch.Tensor],
        test_data: Dict[str, torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            train_data: A dictionary that has the training
                examples, features, budgets and learning curves.
            test_data: Same as for the training data, but it is
                for the testing part and it does not feature labels.

        Returns:
            means, stds: The means of the predictions for the
                testing points and the standard deviations.
        """
        self.gp.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad(): # gpytorch.settings.fast_pred_var():
            projected_train_x = self.feature_extractor(
                train_data['X_train'],
                train_data['train_budgets'],
                train_data['train_curves'],
                train_data['state_dicts'],
                train_data['architectures']
            )
            self.gp.set_train_data(inputs=projected_train_x, targets=train_data['y_train'], strict=False)
            projected_test_x = self.feature_extractor(
                test_data['X_test'],
                test_data['test_budgets'],
                test_data['test_curves'],
                test_data['state_dicts'],
                test_data['architectures']
            )
            preds = self.likelihood(self.gp(projected_test_x))

        means = preds.mean.detach().to('cpu').numpy().reshape(-1, )
        stds = preds.stddev.detach().to('cpu').numpy().reshape(-1, )

        return means, stds

    def model_train(self, data: Dict[str, torch.Tensor], load_checkpoint: bool = False):
        """
        Train the surrogate model.

        Args:
            data: A dictionary which has the training examples, training features,
                training budgets and in the end the training curves.
            load_checkpoint: A flag whether to load the state from a previous checkpoint,
                or whether to start from scratch.
        """
        self.iterations += 1
        print(f'Starting iteration: {self.iterations}')
        # whether the state has been changed. Basically, if a better loss was found during
        # this optimization iteration then the state (weights) were changed.
        weights_changed = False

        if load_checkpoint:
            try:
                self.load_checkpoint()
            except FileNotFoundError:
                print(f'No checkpoint file found at: {self.checkpoint_path}'
                                  f'Training the GP from the beginning')

        self.gp.train()
        self.likelihood.train()
        self.feature_extractor.train()

        self.optimizer = torch.optim.Adam([
            {'params': self.gp.parameters(), 'lr': self.surrogate_config['learning_rate']},
            {'params': self.feature_extractor.parameters(), 'lr': self.surrogate_config['learning_rate']}],
        )

        X_train = data['X_train']
        train_budgets = data['train_budgets']
        train_curves = data['train_curves']
        state_dicts = data['state_dicts']
        architectures = data['architectures']
        y_train = data['y_train']

        initial_state = self.get_state()
        training_errored = False

        if self.restart:
            self.restart_optimization()
            nr_epochs = self.nr_epochs
            # 2 cases where the statement below is hit.
            # - We are switching from the full training phase in the beginning to refining.
            # - We are restarting because our refining diverged
            if self.initial_nr_points <= self.iterations:
                self.restart = False
        else:
            nr_epochs = self.refine_epochs

        # where the mean squared error will be stored
        # when predicting on the train set
        mse = 0.0

        for epoch_nr in range(0, nr_epochs):
            nr_examples_batch = X_train.size(dim=0)
            # if only one example in the batch, skip the batch.
            # Otherwise, the code will fail because of batchnorm
            if nr_examples_batch == 1:
                continue

            # Zero backprop gradients
            self.optimizer.zero_grad()
            projected_x = self.feature_extractor(X_train, train_budgets, train_curves, state_dicts, architectures)
            self.gp.set_train_data(projected_x, y_train, strict=False)
            output = self.gp(projected_x)

            try:
                # Calc loss and backprop derivatives
                loss = -self.mll(output, self.gp.train_targets)
                loss_value = loss.detach().to('cpu').item()
                mse = gpytorch.metrics.mean_squared_error(output, self.gp.train_targets)
                print(
                    f'Epoch {epoch_nr} - MSE {mse:.5f}, '
                    f'Loss: {loss_value:.3f}, '
                    f'lengthscale: {self.gp.covar_module.base_kernel.lengthscale.item():.3f}, '
                    f'noise: {self.gp.likelihood.noise.item():.3f}, '
                )
                loss.backward()
                self.optimizer.step()
            except Exception as training_error:
                print(f'The following error happened while training: {training_error}')
                # An error has happened, trigger the restart of the optimization and restart
                # the model with default hyperparameters.
                self.restart = True
                training_errored = True
                break

        """
        # metric too high, time to restart, or we risk divergence
        if mse > 0.15:
            if not self.restart:
                self.restart = True
        """
        if training_errored:
            self.save_checkpoint(initial_state)
            self.load_checkpoint()

        self.save_checkpoint(self.get_state())

    def generate_candidate_configurations(
        self,
        used_indices: Dict[int, int]
    ) -> Tuple[List, List, List, List]:
        """
        Generate candidate configurations that will be
        fantasized upon.

        Returns:
            (configurations, hp_indices, hp_budgets, learning_curves): Tuple
                A tuple of configurations, their indices in the hp list
                and the budgets that they should be fantasized upon.
        """
        hp_indices = []
        hp_budgets = []
        learning_curves = []
        state_dicts = []
        architectures = []

        for hp_index in range(0, len(self.data_interface.available_hp_indices)):
            if hp_index in used_indices.keys():
                budget = used_indices[hp_index]
                model = self.data_interface.get_pytorch_model(hp_index, budget)
                state_dict = self.data_interface.get_pytorch_model(hp_index, budget).state_dict()
                total_params = sum(p.numel() for p in model.parameters())
                # self.update_configuration(total_params)
                # Take the max budget evaluated for a certain hpc
                max_budget = budget
                next_budget = max_budget + self.fantasize_step
                # take the learning curve until the point we have evaluated so far
                curve = self.data_interface.get_performance_curve(hp_index, 'test_accuracy', max_budget)
                # if the curve is shorter than the length of the kernel size,
                # pad it with zeros
                difference_curve_length = self.surrogate_config['cnn_kernel_size'] - len(curve)
                if difference_curve_length > 0:
                    curve.extend([0.0] * difference_curve_length)
            else:
                # The hpc was not evaluated before, so fantasize its
                # performance
                next_budget = self.fantasize_step
                curve = [0, 0, 0]
                model = self.data_interface.get_pytorch_model(hp_index, next_budget)
                state_dict = self.data_interface.get_pytorch_model(hp_index, next_budget).state_dict()
                total_params = sum(p.numel() for p in model.parameters())
                # self.update_configuration(total_params)

            # this hyperparameter configuration is not evaluated fully
            if next_budget <= self.data_interface.max_budget:
                state_dicts.append(state_dict)
                architectures.append(self.data_interface.get_architecture(hp_index))
                hp_indices.append(hp_index)
                hp_budgets.append(next_budget)
                learning_curves.append(curve)

        configurations = self.data_interface.hp_indices_to_configs(hp_indices)
        return configurations, hp_indices, hp_budgets, learning_curves, state_dicts, architectures

    def acq(
        self,
        best_value: float,
        mean: float,
        std: float,
        explore_factor: Optional[float] = 0.25,
        acq_fc: str = 'ei',
    ) -> float:
        """
        The acquisition function that will be called
        to evaluate the score of a hyperparameter configuration.

        Parameters
        ----------
        best_value: float
            Best observed function evaluation. Individual per fidelity.
        mean: float
            Point mean of the posterior process.
        std: float
            Point std of the posterior process.
        explore_factor: float
            The exploration factor for when ucb is used as the
            acquisition function.
        ei_calibration_factor: float
            The factor used to calibrate expected improvement.
        acq_fc: str
            The type of acquisition function to use.

        Returns
        -------
        acq_value: float
            The value of the acquisition function.
        """
        if acq_fc == 'ei':
            if std == 0:
                return 0
            z = (mean - best_value) / std
            acq_value = (mean - best_value) * norm.cdf(z) + std * norm.pdf(z)
        elif acq_fc == 'ucb':
            acq_value = mean + explore_factor * std
        elif acq_fc == 'thompson':
            acq_value = np.random.normal(mean, std)
        elif acq_fc == 'exploit':
            acq_value = mean
        else:
            raise NotImplementedError(
                f'Acquisition function {acq_fc} has not been'
                f'implemented',
            )

        return acq_value

    def find_suggested_config(self, mean_predictions: np.ndarray, mean_stds: np.ndarray, budgets: List, used_indices: List):
        """Find the config with the best score with the acquisition function.
        
        Args:
            mean_predictions: The mean predictions of the posterior.
            mean_stds: The mean standard deviations of the posterior.
            budgets: The next budgets that the hyperparameter configurations
                will be evaluated for.

        Returns:
            best_index: The index of the hyperparameter configuration with the
                highest score.
        """
        highest_acq_value = np.NINF
        best_index = -1

        index = 0
        for mean_value, std in zip(mean_predictions, mean_stds):
            budget = int(budgets[index])
            best_value = self.calculate_fidelity_ymax(budget, used_indices)
            acq_value = self.acq(best_value, mean_value, std, acq_fc='ei')
            if acq_value > highest_acq_value:
                highest_acq_value = acq_value
                best_index = index
            index += 1

        return best_index

    def calculate_fidelity_ymax(self, fidelity: int, used_indices: Dict[int, int]):
        """
        Find ymax for a given fidelity level.

        If there are hyperparameters evaluated for that fidelity
        take the maximum from their values. Otherwise, take
        the maximum from all previous fidelity levels for the
        hyperparameters that we have evaluated.

        Args:
            fidelity: The fidelity of the hyperparameter
                configuration.

        Returns:
            best_value: The best value seen so far for the
                given fidelity.
        """
        exact_fidelity_config_values = []
        lower_fidelity_config_values = []

        for hp_index in used_indices.keys():
            if used_indices[hp_index] >= fidelity:
                # we've evaluated this config at this fidelity
                performance = self.data_interface.get_performance_curve(hp_index, 'test_accuracy', fidelity)[-1]
                exact_fidelity_config_values.append(performance)
            else:
                # we haven't evaluated this config at this fidelity; get max from previous evaluations
                performance_curve = self.data_interface.get_performance_curve(hp_index, 'test_accuracy', used_indices[hp_index])
                performance = max(performance_curve)
                lower_fidelity_config_values.append(performance)

        if len(exact_fidelity_config_values) > 0:
            # lowest error corresponds to best value
            best_value = max(exact_fidelity_config_values)
        else:
            best_value = max(lower_fidelity_config_values)

        return best_value

    def _prepare_dataset_and_budgets(self, used_indices: Dict[int, int]) -> Dict[str, torch.Tensor]:
        """
        Prepare the data that will be the input to the surrogate.

        Returns:
            data: A Dictionary that contains inside the training examples,
            the budgets, the curves, the state dicts, and lastly the labels.
        """
        train_examples, train_labels, train_budgets, train_curves, state_dicts, architectures = self.history_configurations(used_indices)

        train_examples = np.array(train_examples, dtype=np.single)
        train_labels = np.array(train_labels, dtype=np.single)
        train_budgets = np.array(train_budgets, dtype=np.single)
        train_curves = self.patch_curves_to_same_length(train_curves)
        train_curves = np.array(train_curves, dtype=np.single)

        # scale budgets to [0, 1]
        train_budgets = train_budgets / self.data_interface.max_budget

        train_examples = torch.tensor(train_examples)
        train_labels = torch.tensor(train_labels)
        train_budgets = torch.tensor(train_budgets)
        train_curves = torch.tensor(train_curves)

        train_examples = train_examples.to(device=self.device)
        train_labels = train_labels.to(device=self.device)
        train_budgets = train_budgets.to(device=self.device)
        train_curves = train_curves.to(device=self.device)

        data = {
            'X_train': train_examples,
            'train_budgets': train_budgets,
            'train_curves': train_curves,
            'state_dicts': state_dicts,
            'architectures': architectures,
            'y_train': train_labels,
        }

        return data

    def history_configurations(
        self,
        used_indices: dict,
    ) -> Tuple[List, List, List, List]:
        """
        Generate the configurations, labels, budgets and curves based on
        the history of evaluated configurations.

        Returns:
            (train_examples, train_labels, train_budgets, train_curves):
                A tuple of examples, labels, budgets and curves for the
                configurations evaluated so far.
        """
        train_examples = []
        train_labels = []
        train_budgets = []
        train_curves = []
        state_dicts = []
        architectures = []

        for hp_index in used_indices.keys():
            budget = used_indices[hp_index]
            budgets = np.arange(0, budget)
            performances = self.data_interface.get_performance_curve(hp_index, 'test_accuracy', budget)
            example = self.data_interface.prepare_config(self.data_interface.find_config_by_hp_index(hp_index=hp_index))

            # Original state dict and architecture
            model = self.data_interface.get_pytorch_model(hp_index, budget)
            state_dict = model.state_dict()
            architecture = self.data_interface.get_architecture(hp_index)

            for budget, performance in zip(budgets, performances):
                train_examples.append(example)
                train_budgets.append(budget)
                train_labels.append(performance)
                state_dicts.append(state_dict)
                architectures.append(architecture)
                train_curve = performances[:budget - 1] if budget > 1 else [0.0]
                difference_curve_length = self.surrogate_config['cnn_kernel_size'] - len(train_curve)
                if difference_curve_length > 0:
                    train_curve.extend([0.0] * difference_curve_length)
                train_curves.append(train_curve)

                # Generate permuted state dicts and add to training data
                permuted_state_dicts = self.generate_permuted_state_dicts(state_dict, self.num_permutations)
                for permuted_state_dict in permuted_state_dicts:
                    train_examples.append(example)
                    train_budgets.append(budget)
                    train_labels.append(performance)
                    state_dicts.append(permuted_state_dict)
                    architectures.append(architecture)
                    train_curves.append(train_curve)

        return train_examples, train_labels, train_budgets, train_curves, state_dicts, architectures

    @staticmethod
    def patch_curves_to_same_length(curves):
        """
        Patch the given curves to the same length.

        Finds the maximum curve length and patches all
        other curves that are shorter in length with zeroes.

        Args:
            curves: The given hyperparameter curves.

        Returns:
            curves: The updated array where the learning
                curves are of the same length.
        """
        max_curve_length = 0
        for curve in curves:
            if len(curve) > max_curve_length:
                max_curve_length = len(curve)

        for curve in curves:
            difference = max_curve_length - len(curve)
            if difference > 0:
                curve.extend([0.0] * difference)

        return curves

    def load_checkpoint(self):
        """
        Load the state from a previous checkpoint.
        """
        checkpoint = torch.load(self.checkpoint_path)
        self.gp.load_state_dict(checkpoint['gp_state_dict'])
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])

    def save_checkpoint(self, state: Dict = None):
        """
        Save the given state or the current state in a
        checkpoint file.

        Args:
            state: The state to save, if none, it will
            save the current state.
        """
        if state is None:
            torch.save(
                self.get_state(),
                self.checkpoint_path,
            )
        else:
            torch.save(
                state,
                self.checkpoint_path,
            )

    def get_state(self) -> Dict[str, Dict]:
        """
        Get the current state of the surrogate.

        Returns:
            current_state: A dictionary that represents
                the current state of the surrogate model.
        """
        current_state = {
            'gp_state_dict': deepcopy(self.gp.state_dict()),
            'feature_extractor_state_dict': deepcopy(self.feature_extractor.state_dict()),
            'likelihood_state_dict': deepcopy(self.likelihood.state_dict()),
        }

        return current_state

    def update_configuration(self, total_params):
        # adjust layer sizes based on total parameters
        self.surrogate_config['flattened_params_units'] = total_params  # or some function of total_params...
        self.surrogate_config['state_dict_nr_channels'] = min(128, total_params // 100) # admittedly, a somewhat arbitrary formula


import random

class RandomSearchSurrogateModel:
    """Random search model that uses past performance data to predict rankings."""
    def __init__(self, park, use_cnn, use_weights):
        self.data_interface: SimpleCNNParkInterface | PretrainedModelParkInterface = get_data_interface(park)
        self.available_indices = self.data_interface.available_hp_indices
        self.nr_layers = 0
        self.seed(19)

    def suggest_next_hp_index(self, used_indices):
        """Randomly suggest a new hyperparameter index from available indices."""
        available_configs = [idx for idx in self.available_indices if used_indices[idx] < self.data_interface.max_budget]
        if not available_configs:
            raise Exception("No more unique configurations to suggest.")
        index = random.choice(available_configs)
        return index, self.data_interface.max_budget

    def train(self, used_indices):
        """Training the surrogate model - dummy for random search."""
        pass  # No training needed for random search

    def predict_rankings(self, used_indices):
        """Predict rankings based on historical test accuracy data."""
        rankings = []
        for idx in self.available_indices:
            results = self.data_interface.get_results(idx)
            if results:
                # Consider the last available result for ranking
                latest_result = sorted(results, key=lambda x: x['epoch'])[-1]
                rankings.append((idx, latest_result['test_accuracy']))
        # Sort by test accuracy, higher is better
        rankings.sort(key=lambda x: x[1], reverse=True)
        return [rank[0] for rank in rankings]  # Return only indices

    def seed(self, seed_value=42):
        """Set seed for reproducibility."""
        random.seed(seed_value)
        np.random.seed(seed_value)


