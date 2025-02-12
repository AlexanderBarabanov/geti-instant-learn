from typing import List

from context_learner.types.image import Image
from context_learner.types.annotations import Annotations
from context_learner.types.state import State


class Pipeline:
    def __init__(self):
        self._state = State()

    def get_state(self):
        return self._state

    def learn(self, reference_images: List[Image], reference_annotations: List[Annotations]):
        """
        This method learns the context

        Args:
            reference_images: A list of images ot learn from.
            reference_annotations: A list of multi-modal annotations associated with the image.

        Returns:
            None

        Examples:
            >>> p = Pipeline()
            >>> p.learn([Image()], [Annotations()])
        """
        pass

    def infer(self, target_images: List[Image]):
        """
        This method uses the learned context to infer object locations.

        Args:
            target_images: A List of images to infer.

        Returns:
            None

        Examples:
            >>> p = Pipeline()
            >>> p.infer([Image()])
        """
        pass
