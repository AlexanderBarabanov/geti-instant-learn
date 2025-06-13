"""Base class for calculators."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from abc import abstractmethod

from visionprompt.processes import Process


class Calculator(Process):
    """This is the base class for calculators.

    Examples:
        >>> from visionprompt.processes.calculators import Calculator
        >>>
        >>> class MyCalculator(Calculator):
        ...     def __call__(self, *args, **kwargs):
        ...         pass
        >>>
        >>> my_calculator = MyCalculator()
        >>> my_calculator()
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, *args, **kwargs) -> None:
        """Calculate the metrics."""
