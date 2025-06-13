"""Base class for prior filters."""
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from visionprompt.filters import Filter
from visionprompt.types import Priors


class PriorFilter(Filter):
    """This is the base class for all prior filters.

    Example:
        >>> filter = PriorFilter()
        >>> filtered_priors = filter(priors)
    """

    def __call__(self, priors: list[Priors]) -> list[Priors]:
        """Filter the priors."""
