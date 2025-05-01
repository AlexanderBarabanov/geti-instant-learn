# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from visionprompt.context_learner.processes import Process
from visionprompt.context_learner.types import Image, Masks, Priors


class Segmenter(Process):
    """This class extracts segmentation masks from priors."""

    def __call__(self, images: list[Image], priors: list[Priors]) -> list[Masks]:
        """This method extracts priors from similarities.

        Args:
            images: The images to segment.
            priors: The priors that are used for segmenting.

        Returns:
            Segmentation masks.

        Examples:
            >>> from visionprompt.context_learner.types.state import State
            >>> from visionprompt.context_learner.types.image import Image
            >>> state = State()
            >>> segment = Segmenter(state=state)
            >>> r = segment([Image()], [Priors()])
        """
        return [Masks()]
