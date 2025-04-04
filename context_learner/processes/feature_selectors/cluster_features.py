from typing import List

import numpy as np
from sklearn.cluster import KMeans

from context_learner.processes.feature_selectors.feature_selector_base import (
    FeatureSelector,
)
from context_learner.types.features import Features
import torch


class ClusterFeatures(FeatureSelector):
    def __call__(
        self, features_per_image: List[Features], num_clusters: int = 8
    ) -> List[Features]:
        """
        This method clusters all features (across all reference images and their masks) and averages the features per cluster.

        Args:
            features_per_image: A list of features for each reference image.
            num_clusters: The number of clusters to use.
        Returns:
            A list of Features object with the averaged features per cluster.
        """
        result_features = Features()

        for class_id, feature_list in self.get_all_local_class_features(
            features_per_image
        ).items():
            original_device = feature_list[0].device
            stacked_features = torch.cat(feature_list, dim=0)
            features_np = stacked_features.cpu().numpy()
            kmeans = KMeans(n_clusters=num_clusters, init="k-means++", random_state=42)
            kmeans.fit(features_np)

            # use centroid of cluster as prototype
            part_level_features = []
            for c in range(num_clusters):
                part_level_feature = features_np[kmeans.labels_ == c].mean(axis=0)
                # Even though input features are normalized, when we take the mean of a cluster's features,
                # the resulting centroid is not guaranteed to have unit norm
                part_level_feature = part_level_feature / np.linalg.norm(
                    part_level_feature, axis=-1, keepdims=True
                )
                part_level_features.append(torch.from_numpy(part_level_feature))

            part_level_features = torch.stack(part_level_features, dim=0).to(
                original_device
            )  # n_clusters, 256
            result_features.add_local_features(part_level_features, class_id)

        return [result_features]
