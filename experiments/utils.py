# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from surrogates import FmsSurrogateModel, RandomSearchSurrogateModel
from dataset.pretrained_model_park.interface import PretrainedModelParkInterface
from dataset.simple_cnn_park.interface import SimpleCNNParkInterface

def get_surrogate_model(experiment_type, park, use_cnn, use_weights):
    if experiment_type == "random-search":
        return RandomSearchSurrogateModel(park=park, use_cnn=use_cnn, use_weights=use_weights)
    else:
        return FmsSurrogateModel(experiment_type=experiment_type, park=park, use_cnn=use_cnn, use_weights=use_weights)

def get_data_interface(park):
    if park == "simple_cnn_park" or park == "simple_cnn_park_transfer":
        return SimpleCNNParkInterface()
    elif park == "pretrained_model_park_cifar10" or park == "pretrained_model_park_transfer_cifar10":
        return PretrainedModelParkInterface("cifar10")
    elif park == "pretrained_model_park_svhn" or park == "pretrained_model_park_transfer_svhn":
        return PretrainedModelParkInterface("svhn")
    else:
        raise ValueError(f"Unknown park: {park}")