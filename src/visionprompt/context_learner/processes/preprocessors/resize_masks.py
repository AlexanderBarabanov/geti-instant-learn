# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from visionprompt.context_learner.processes import Process
from visionprompt.context_learner.types import Priors


class ResizeMasks(Process):
    """This process resizes the masks to the given size."""

    def __init__(self, size: int | tuple[int, int] | None = None) -> None:
        """This initializes the process.

        Args:
            size: The size to resize the masks to. If a tuple is provided, the masks will be resized to the given width
              and height. If an integer is provided, the masks will be resized to the given size, maintaining aspect
                ratio. If None is provided, the masks will not be resized.
        """
        super().__init__()
        self.size = size

    def __call__(
        self,
        priors: list[Priors],
    ) -> list[Priors]:
        """Inspect overlapping areas between different label masks.

        Args:
            priors: List of Priors objects of which the masks will be resized

        Returns:
            List of Priors objects with resized masks
        """
        for prior in priors:
            prior.masks.resize_inplace(self.size)
        return priors
