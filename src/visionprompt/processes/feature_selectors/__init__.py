"""Feature selectors."""
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from visionprompt.processes.feature_selectors.all_features import AllFeaturesSelector
from visionprompt.processes.feature_selectors.average_features import AverageFeatures
from visionprompt.processes.feature_selectors.cluster_features import ClusterFeatures
from visionprompt.processes.feature_selectors.feature_selector_base import (
    FeatureSelector,
)

__all__ = [
    "AverageFeatures",
    "ClusterFeatures",
    "FeatureSelector",
    "AllFeaturesSelector",
]
