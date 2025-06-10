# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from abc import abstractmethod

from visionprompt.context_learner.processes import Process


class Calculator(Process):
    """This is the base class for calculators."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self) -> None:
        """Calculate the metrics."""
