# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple

from context_learner.processes.process_base import Process
from context_learner.types import Features, Image, Annotations


class Encoder(Process):
    def __call__(
        self, images: List[Image], annotations: Optional[List[Annotations]] = None
    ) -> List[Features]:
        """
        This method creates an embedding from the images for locations inside the polygon.

        Args:
            images: A list of images.
            annotations: A list of a collection of annotations per image.

        Returns:
            A list of extracted features.

        Examples:
            >>> from context_learner.types.state import State
            >>> state = State()
            >>> enc = Encoder(state=state)
            >>> r = enc([Image()], [Annotations()])
        """
        return [Features()]

    def _setup_model(self):
        """
        This method initializes the model.
        """
        pass

    def _preprocess(
        self, images: List[Image], annotations: Optional[List[Annotations]] = None
    ) -> Tuple[List[Image], Optional[List[Annotations]]]:
        """
        This method preprocesses the images and annotations.
        """
        return images, annotations
