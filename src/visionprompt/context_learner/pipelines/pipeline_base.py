# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

from visionprompt.context_learner.types import Image, Priors, State
from visionprompt.context_learner.types.annotations import Annotations


class Pipeline:
    """This class is the base class for all pipelines."""

    def __init__(self, args: argparse.Namespace | None = None) -> None:
        self._state = State()
        self.args = args

    def get_state(self) -> State:
        """Get the state of the pipeline."""
        return self._state

    def reset_state(self, reset_references: bool = True) -> None:
        """Reset the state of the pipeline."""
        if reset_references:
            self._state.reference_images.clear()
            self._state.reference_priors.clear()
            self._state.reference_features.clear()
            self._state.processed_reference_masks.clear()
        self._state.target_images.clear()
        self._state.target_features.clear()
        self._state.similarities.clear()
        self._state.priors.clear()
        self._state.masks.clear()
        self._state.annotations.clear()
        self._state.used_points.clear()

    def learn(self, reference_images: list[Image], reference_priors: list[Priors]) -> None:
        """This method learns the context.

        Args:
            reference_images: A list of images ot learn from.
            reference_priors: A list of priors associated with the image.

        Returns:
            None

        Examples:
            >>> p = Pipeline()
            >>> p.learn([Image()], [Priors()])
        """

    def infer(self, target_images: list[Image]) -> list[Annotations]:
        """This method uses the learned context to infer object locations.

        Args:
            target_images: A List of images to infer.

        Returns:
            None

        Examples:
            >>> p = Pipeline()
            >>> p.infer([Image()])
        """
