from typing import List

from context_learner.processes.feature_selectors.feature_selector_base import (
    FeatureSelector,
)
from context_learner.types.features import Features
import torch


class AverageFeatures(FeatureSelector):
    def __call__(self, features_per_image: List[Features]) -> List[Features]:
        """
        This method averages all features across all reference images and their masks for each class.
        The result will be a single averaged feature vector per class.

        Args:
            features_per_image: A list of features for each reference image.
        Returns:
            A list containing a single Features object with the averaged features per class.
        """
        result_features = Features()

        # Average features for each class
        for class_id, feature_list in self.get_all_class_features(
            features_per_image
        ).items():
            stacked_features = torch.cat(feature_list, dim=0)
            averaged_features = stacked_features.mean(dim=0, keepdim=True)
            averaged_features = averaged_features / averaged_features.norm(
                dim=-1, keepdim=True
            )
            result_features.add_local_features(averaged_features, class_id)

        return [result_features]
