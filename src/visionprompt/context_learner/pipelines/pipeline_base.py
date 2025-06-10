# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod

from visionprompt.context_learner.types import Image, Priors
from visionprompt.context_learner.types.results import Results


class Pipeline(ABC):
    """This class is the base class for all pipelines."""

    @abstractmethod
    def __init__(self) -> None:
        """Initialization method that caches all parameters."""

    @abstractmethod
    def learn(self, reference_images: list[Image], reference_priors: list[Priors]) -> Results:
        """This method learns the context.

        Args:
            reference_images: A list of images ot learn from.
            reference_priors: A list of priors associated with the image.

        Returns:
            None
        """

    @abstractmethod
    def infer(self, target_images: list[Image]) -> Results:
        """This method uses the learned context to infer object locations.

        Args:
            target_images: A List of images to infer.

        Returns:
            None
        """
