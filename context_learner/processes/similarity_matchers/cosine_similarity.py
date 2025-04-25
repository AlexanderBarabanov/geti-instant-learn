# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List

from context_learner.processes.similarity_matchers.similarity_matcher_base import (
    SimilarityMatcher,
)
from context_learner.types import Features, Similarities


class CosineSimilarity(SimilarityMatcher):
    """
    This class computes the cosine similarity between the reference features and the target features.
    """

    def __call__(
        self, reference_features: List[Features], target_features: List[Features]
    ) -> List[Similarities]:
        """
        This function computes the cosine similarity between the reference features and the target features.
        This similarity matcher expects the features of multiple reference images to be reduced (averaged/clustered) into a single Features object.

        Args:
            reference_features: List[Features] List of reference features, one per prior image instance
            target_features: List[Features] List of target features, one per target image instance

        Returns:
            List[Similarities] List of similarities, one per target image instance which are resized to the original image size
        """
        reference_features = reference_features[0]
        per_image_similarities: List[Similarities] = []
        for i, target in enumerate(target_features):
            normalized_target = target.global_features / target.global_features.norm(
                dim=-1, keepdim=True
            )
            embedding_shape = target.global_features_shape
            original_image_size = self._state.target_images[i].size
            transformed_image_size = self._state.target_images[i].transformed_size

            # reshape from (encoder_shape, encoder_shape, embed_dim) to (encoder_shape*encoder_shape, embed_dim) if necessary
            if normalized_target.dim() == 3:
                normalized_target = normalized_target.reshape(
                    normalized_target.shape[0] * normalized_target.shape[1],
                    normalized_target.shape[2],
                )
            # compute cosine similarity of (1,1,embed_dim) and (encoder_shape*encoder_shape, embed_dim)
            all_similarities = Similarities()
            for (
                class_id,
                local_reference_features_per_mask,
            ) in reference_features.local_features.items():
                # Need to loop since number of reference features can differ per input mask.
                for local_reference_features in local_reference_features_per_mask:
                    similarities = local_reference_features @ normalized_target.T
                    similarities = self._resize_similarities(
                        similarities=similarities,
                        transformed_image_size=transformed_image_size,
                        original_image_size=original_image_size,
                        embedding_shape=embedding_shape,
                    )

                    all_similarities.add(
                        similarities=similarities,
                        class_id=class_id,
                    )

            per_image_similarities.append(all_similarities)
        return per_image_similarities
