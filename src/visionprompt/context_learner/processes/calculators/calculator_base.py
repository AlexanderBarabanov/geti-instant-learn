# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from visionprompt.context_learner.processes import Process
from visionprompt.context_learner.types import State


class Calculator(Process):
    """This is the base class for calculators."""

    def __init__(self, state: State) -> None:
        super().__init__(state)

    def __call__(self, *args, **kwargs) -> None:
        """Calculate the metrics."""
        raise NotImplementedError
