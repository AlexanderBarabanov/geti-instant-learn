# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod


class Process(ABC):
    """This is the base class of processes within pipelines."""

    @abstractmethod
    def __init__(self) -> None:
        """This initializes the process."""

    @abstractmethod
    def __call__(self) -> None:
        """This runs the process."""
