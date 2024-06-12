# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from io import open
from os import path

from setuptools import find_packages, setup

def requirements():
    list_requirements = []
    with open("requirements.txt") as f:
        for line in f:
            list_requirements.append(line.rstrip())
    return list_requirements


setup(
    name="fms",
    version="0.0.1",
    description="Forecasting Model Search",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    author="Nikhil Mehta",
    author_email="nikmehta@nvidia.com",
    packages=find_packages(exclude=[]),
    install_requires=requirements(),  # Optional
)
