# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

import torch
import torch.nn.functional as F
from context_learner.processes.process_base import Process
from context_learner.types.features import Features
from context_learner.types.similarities import Similarities


class SimilarityMatcher(Process):
    def __call__(
        self, reference_features: List[Features], target_features: List[Features]
    ) -> List[Similarities]:
        """
        This method calculates the similarities between reference features and target features.

        Args:
            reference_features: The reference features per image.
            target_features: The target features per image.

        Returns:
            A list of similarities per target_features.
            Note: the number of elements in output list is usually the same as
                the number of items in the target_features list.

        Examples:
            >>> from context_learner.types.state import State
            >>> state = State()
            >>> sim = SimilarityMatcher(state=state)
            >>> r = sim([Features()], [Features()])
        """
        return [Similarities()]

    def _resize_similarities(
        self,
        similarities: torch.Tensor,
        original_image_size: tuple[int, int],
        embedding_shape: tuple[int, int],
        transformed_image_size: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        This function resizes the similarities to the target image size while removing padding.

        Args:
            similarities: torch.Tensor The similarities to resize
            transformed_image_size: tuple[int, int] The size of the transformed image
            original_image_size: tuple[int, int] The size of the original image
            embedding_shape: tuple[int, int] The shape of the embedding

        Returns:
            torch.Tensor The resized similarities
        """
        if transformed_image_size is not None:
            similarities = similarities.reshape(
                similarities.shape[0],
                1,  # needed for bilinear interpolation
                embedding_shape[0],
                embedding_shape[1],
            )
            # resize the similarities to the target image size
            similarities = F.interpolate(
                similarities,
                size=self._state.encoder_input_size,
                mode="bilinear",
                align_corners=False,
            )
            # remove padding
            similarities = similarities[
                ...,
                : transformed_image_size[0],
                : transformed_image_size[1],
            ]
            # resize back to original size
            similarities = F.interpolate(
                similarities,
                size=original_image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        else:
            similarities = similarities.reshape(
                similarities.shape[0],
                self._state.encoder_input_size // self._state.encoder_patch_size,
                self._state.encoder_input_size // self._state.encoder_patch_size,
            ).unsqueeze(0)
            similarities = F.interpolate(
                similarities,
                size=original_image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            if similarities.ndim == 4:
                similarities = similarities.squeeze(0)

        return similarities
