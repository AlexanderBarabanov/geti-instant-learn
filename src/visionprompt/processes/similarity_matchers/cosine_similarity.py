"""Cosine similarity matcher."""
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from visionprompt.processes.similarity_matchers.similarity_matcher_base import (
    SimilarityMatcher,
)
from visionprompt.types import Features, Similarities, Image


class CosineSimilarity(SimilarityMatcher):
    """This class computes the cosine similarity.

    Examples:
        >>> from visionprompt.processes.similarity_matchers import CosineSimilarity
        >>> from visionprompt.types import Features, Image
        >>>
        >>> matcher = CosineSimilarity()
        >>> similarities = matcher(
        ...     reference_features=[Features()],
        ...     target_features=[Features()],
        ...     target_images=[Image()],
        ... )
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        reference_features: list[Features],
        target_features: list[Features],
        target_images: list[Image] | None = None,
    ) -> list[Similarities]:
        """This function computes the cosine similarity between the reference features and the target features.

        This similarity matcher expects the features of multiple reference images
        to be reduced (averaged/clustered) into a single Features object.

        Args:
            reference_features: List[Features] List of reference features, one per prior image instance
            target_features: List[Features] List of target features, one per target image instance
            target_images: List[Image] List of target images

        Returns:
            List[Similarities] List of similarities, one per target image instance which are resized to
              the original image size
        """
        reference_features = reference_features[0]
        per_image_similarities: list[Similarities] = []
        for i, target in enumerate(target_features):
            normalized_target = target.global_features / target.global_features.norm(
                dim=-1,
                keepdim=True,
            )

            # reshape from (encoder_shape, encoder_shape, embed_dim)
            # to (encoder_shape*encoder_shape, embed_dim) if necessary
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
                    similarities = self._resize_similarities_to_target_size(
                        similarities=similarities,
                        # scaling to 1024 can speed up prompt generation, however we can also resize the images at
                        #   the start of the pipeline
                        target_size=target_images[i].size,
                        unpadded_image_size=target_images[i].sam_preprocessed_size,
                    )

                    all_similarities.add(
                        similarities=similarities,
                        class_id=class_id,
                    )

            per_image_similarities.append(all_similarities)
        return per_image_similarities
