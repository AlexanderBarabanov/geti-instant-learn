# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List
import torch

from context_learner.types.data import Data


class FeaturesMetaData:
    class_id: int
    cluster_id: int | None

    def __init__(
        self,
        class_id: int,
        cluster_id: int | None,
    ):
        self.class_id = class_id
        self.cluster_id = cluster_id


class Features(Data):
    """
    This class represents features of a single image. Each image has a global feature representation
    and can have multiple local feature representations based on the number of masks per class.
    """

    def __init__(self, global_features: torch.Tensor = None):
        """
        Initialize the features from a torch tensor.

        Args:
            global_features: The global features of the image.
        """
        self._global_features: torch.Tensor = global_features
        self._local_features: dict[int, List[torch.Tensor]] = {}
        self._meta_data: List[FeaturesMetaData] = []

    def add_local_features(
        self, local_features: torch.Tensor, class_id: int, cluster_id: int = 0
    ):
        """
        Add features to the features object.
        """
        if class_id not in self._local_features:
            self._local_features[class_id] = []
        self._local_features[class_id].append(local_features)
        self._meta_data.append(FeaturesMetaData(class_id, cluster_id))

    @property
    def global_features(self):
        return self._global_features

    @global_features.setter
    def global_features(self, global_features: torch.Tensor):
        self._global_features = global_features

    @property
    def local_features(self):
        return self._local_features

    @local_features.setter
    def local_features(self, local_features: dict[int, List[torch.Tensor]]):
        self._local_features = local_features

    def get_local_features(
        self, class_idx: int, cluster_id: int | None = None
    ) -> List[torch.Tensor]:
        """
        Get the local features for a specific class. If cluster_id is None, return all local features for the class.
        If cluster_id is not None, return the local features for the specific cluster.

        Args:
            class_idx: The class index of the features to get.
            cluster_id: The cluster index of the features to get. If None, return all features for the class.

        Returns:
            A list of local features.
        """

        if cluster_id is None:
            return self._local_features[class_idx]
        else:
            return [
                f
                for f, m in zip(self._local_features[class_idx], self._meta_data)
                if m.cluster_id == cluster_id
            ]

    @property
    def global_embedding_dim(self):
        return self._global_features.shape[-1]

    @property
    def local_embedding_dim(self):
        return self._local_features[0][0].shape[-1]

    @property
    def local_features_shape(self):
        """
        Get the shape of the features.
        """
        return self._local_features[0][0].shape

    @property
    def global_features_shape(self):
        """
        Get the shape of the global features.
        """
        return self._global_features.shape

    @property
    def embedding_dim(self):
        """
        Get the embedding dimension of the features.
        """
        return self._global_features.shape[-1]

    def __str__(self):
        return f"Features(shape={self.shape}, embedding_dim={self.embedding_dim})"
