# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List

from context_learner.processes import Process
from context_learner.types import Masks, Priors, Image


class Segmenter(Process):
    def __call__(self, images: List[Image], priors: List[Priors]) -> List[Masks]:
        """
        This method extracts priors from similarities.

        Args:
            priors: The priors that are used for segmenting.

        Returns:
            Segmentation masks.

        Examples:
            >>> from context_learner.types.state import State
            >>> from context_learner.types.image import Image
            >>> state = State()
            >>> segment = Segmenter(state=state)
            >>> r = segment([Image()], [Priors()])
        """
        return [Masks()]
