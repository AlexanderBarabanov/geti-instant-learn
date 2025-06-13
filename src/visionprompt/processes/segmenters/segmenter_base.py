"""Base class for segmenters."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from abc import abstractmethod

from visionprompt.processes import Process
from visionprompt.types import Image, Masks


class Segmenter(Process):
    """This class extracts segmentation masks.

    Examples:
        >>> from visionprompt.processes.segmenters import Segmenter
        >>> from visionprompt.types import Image, Masks
        >>>
        >>> class MySegmenter(Segmenter):
        ...     def __call__(self, images: list[Image], **kwargs) -> list[Masks]:
        ...         return []
        >>>
        >>> my_segmenter = MySegmenter()
        >>> masks = my_segmenter([Image()])
    """

    @abstractmethod
    def __call__(self, images: list[Image], **kwargs) -> list[Masks]:
        """This method extracts segmentation masks.

        Args:
            images: The images to segment.
            kwargs: Additional arguments

        Returns:
            Segmentation masks.
        """
