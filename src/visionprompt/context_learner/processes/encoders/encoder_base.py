# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from abc import abstractmethod

from src.visionprompt.context_learner.types.features import Features
from src.visionprompt.context_learner.types.masks import Masks

from visionprompt.context_learner.processes.process_base import Process
from visionprompt.context_learner.types import Annotations, Image


class Encoder(Process):
    """This class is used to create feature embeddings from images."""

    def __init__(self) -> None:
        super().__init__()
        self._encoder_input_size = None
        self._feature_size = None
        self._patch_size = None

    @abstractmethod
    def __call__(
        self,
        images: list[Image] | None = None,
    ) -> tuple[list[Features], list[Masks]]:
        """This method creates an embedding from the images.

        Args:
            images: List of images to embed.
            *args: Positional arguments for the process
            **kwargs: Keyword arguments for the process

        Returns:
            A list of extracted features.

        """

    def _setup_model(self) -> None:
        """This method initializes the model."""

    @staticmethod
    def _preprocess(
        images: list[Image],
        annotations: list[Annotations] | None = None,
    ) -> tuple[list[Image], list[Annotations] | None]:
        """This method preprocesses the images and annotations."""
        return images, annotations

    @property
    def patch_size(self) -> int:
        """Property for storing the patch_size."""
        return self._patch_size

    @patch_size.setter
    def patch_size(self, patch_size: int) -> None:
        self._patch_size = patch_size

    @property
    def feature_size(self) -> int:
        """Property for storing the feature_size."""
        return self._feature_size

    @feature_size.setter
    def feature_size(self, feature_size: int) -> None:
        self._feature_size = feature_size

    @property
    def encoder_input_size(self) -> int:
        """Property for storing the encoder_input_size."""
        return self._encoder_input_size

    @encoder_input_size.setter
    def encoder_input_size(self, encoder_input_size: int) -> None:
        self._encoder_input_size = encoder_input_size
