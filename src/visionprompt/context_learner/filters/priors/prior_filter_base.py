# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from visionprompt.context_learner.filters import Filter
from visionprompt.context_learner.types import Priors


class PriorFilter(Filter):
    """This is the base class for all prior filters."""

    def __call__(self, priors: list[Priors]) -> list[Priors]:
        """Filter the priors."""
