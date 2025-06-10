# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import torch

from visionprompt.context_learner.processes.feature_selectors.feature_selector_base import (
    FeatureSelector,
)
from visionprompt.context_learner.types import Features


class AverageFeatures(FeatureSelector):
    """This class averages features across all reference images and their masks for each class."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, features_per_image: list[Features]) -> list[Features]:
        """This method averages all features across all reference images and their masks for each class.

        The result will be a single averaged feature vector per class.

        Args:
            features_per_image: A list of features for each reference image.

        Returns:
            A list of Features object with the averaged features per class.
        """
        result_features = Features()

        # Average features for each class
        for class_id, feature_list in self.get_all_local_class_features(
            features_per_image,
        ).items():
            stacked_features = torch.cat(feature_list, dim=0)
            averaged_features = stacked_features.mean(dim=0, keepdim=True)
            # allthough features are already normalized, we make sure that the average is normalized too
            averaged_features = averaged_features / averaged_features.norm(
                dim=-1,
                keepdim=True,
            )
            result_features.add_local_features(averaged_features, class_id)

        return [result_features]
