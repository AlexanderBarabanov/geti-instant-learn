"""Base class for mask processors."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from abc import abstractmethod

from visionprompt.processes import Process
from visionprompt.types import Annotations, Masks


class MaskProcessor(Process):
    """This class processes masks to create annotations (polygons).

    Examples:
        >>> from visionprompt.processes.mask_processors import MaskProcessor
        >>> from visionprompt.types import Annotations, Masks
        >>>
        >>> class MyMaskProcessor(MaskProcessor):
        ...     def __call__(self, masks: list[Masks] | None = None) -> list[Annotations]:
        ...         return []
        >>>
        >>> my_processor = MyMaskProcessor()
        >>> annotations = my_processor([Masks()])
    """

    @abstractmethod
    def __call__(self, masks: list[Masks] | None = None) -> list[Annotations]:
        """This method extracts polygons from masks.

        Args:
            masks: A list of masks.

        Returns:
            A list of polygons that have been created from the masks.

        """
