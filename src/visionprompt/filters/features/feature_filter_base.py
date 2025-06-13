# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from visionprompt.filters import Filter
from visionprompt.types import Features


class FeatureFilterBase(Filter):
    """This is the base class for all feature filters."""

    def __call__(self, features: list[Features]) -> list[Features]:
        """Filter the features."""
