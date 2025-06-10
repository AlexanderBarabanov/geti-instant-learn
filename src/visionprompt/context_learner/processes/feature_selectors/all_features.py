# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import torch

from visionprompt.context_learner.processes.feature_selectors.feature_selector_base import (
    FeatureSelector,
)
from visionprompt.context_learner.types import Features


class AllFeaturesSelector(FeatureSelector):
    """This class selects all features over all prior images without averaging."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, features_per_image: list[Features]) -> list[Features]:
        """This method merges all features over all prior images without averaging.

        Each class will maintain all its feature vectors from all images.

        Args:
            features_per_image: A list of features for each reference image.

        Returns:
            A list of Features object containing all features per class.
        """
        result_features = Features()

        # save global features by stacking with extra first dimension
        global_features = torch.cat(
            [image_features.global_features.unsqueeze(0) for image_features in features_per_image],
            dim=0,
        )
        result_features.global_features = global_features
        result_features.local_features = self.get_all_local_class_features(
            features_per_image,
        )

        return [result_features]
