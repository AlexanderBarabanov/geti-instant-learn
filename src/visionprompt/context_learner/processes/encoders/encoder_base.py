# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from visionprompt.context_learner.processes.process_base import Process
from visionprompt.context_learner.types import Annotations, Features, Image


class Encoder(Process):
    """This class is used to create feature embeddings from images."""

    def __call__(
        self,
        images: list[Image],
        annotations: list[Annotations] | None = None,
    ) -> list[Features]:
        """This method creates an embedding from the images for locations inside the polygon.

        Args:
            images: A list of images.
            annotations: A list of a collection of annotations per image.

        Returns:
            A list of extracted features.

        Examples:
            >>> from visionprompt.context_learner.types.state import State
            >>> state = State()
            >>> enc = Encoder(state=state)
            >>> r = enc([Image()], [Annotations()])
        """
        return [Features()]

    def _setup_model(self) -> None:
        """This method initializes the model."""

    def _preprocess(
        self,
        images: list[Image],
        annotations: list[Annotations] | None = None,
    ) -> tuple[list[Image], list[Annotations] | None]:
        """This method preprocesses the images and annotations."""
        return images, annotations
