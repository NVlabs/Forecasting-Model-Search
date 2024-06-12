# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# on one-cnn-benchmark
python fms.py --experiment_type fms-flat --park simple_cnn_park
python fms.py --experiment_type fms-flat --park simple_cnn_park --ablate-cnn

# on pretrained_model_park
python fms.py --experiment_type fms-flat --park pretrained_model_park_cifar10
python fms.py --experiment_type fms-flat --park pretrained_model_park_cifar10 --ablate-cnn
python fms.py --experiment_type fms-flat --park pretrained_model_park_svhn
python fms.py --experiment_type fms-flat --park pretrained_model_park_svhn --ablate-cnn

# assessing transfer performance
python fms.py --experiment_type fms-flat --park pretrained_model_park_transfer_cifar10
python fms.py --experiment_type fms-flat --park pretrained_model_park_transfer_svhn
python fms.py --experiment_type fms-flat --park simple_cnn_park_transfer