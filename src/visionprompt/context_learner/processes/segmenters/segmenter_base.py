# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from abc import abstractmethod

from visionprompt.context_learner.processes import Process
from visionprompt.context_learner.types import Image, Masks


class Segmenter(Process):
    """This class extracts segmentation masks."""

    @abstractmethod
    def __call__(self, images: list[Image]) -> list[Masks]:
        """This method extracts segmentation masks.

        Args:
            images: The images to segment.

        Returns:
            Segmentation masks.

        """
