# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch

from context_learner.processes import Process
from context_learner.types import Features


class FeatureSelector(Process):
    def __call__(self, features: List[Features]) -> List[Features]:
        """
        This method merges features.

        This class has the same interface as the FeatureFilter() but,
        is defined a process because it it an integral part of a pipeline flow.

        Args:
            features: A list of features.

        Returns:
            A list of new features.

        Examples:
            >>> from context_learner.types.state import State
            >>> state = State()
            >>> select = FeatureSelector(state=state)
            >>> r = select([Features()])
        """
        return [Features()]

    def get_all_local_class_features(
        self, features_per_image: List[Features]
    ) -> dict[int, list[torch.Tensor]]:
        """
        This method gets all features for all classes over all images.

        Args:
            features_per_image: A list of features for each reference image.

        Returns:
            A dictionary of features for each class.
        """
        all_features_per_class = {}

        # First collect all features per class over all images
        for features in features_per_image:
            for class_id, local_features_list in features.local_features.items():
                if class_id not in all_features_per_class:
                    all_features_per_class[class_id] = []
                all_features_per_class[class_id].extend(local_features_list)

        return all_features_per_class
