import argparse
from typing import List

from context_learner.types import Image, Priors, State


class Pipeline:
    def __init__(self, args: argparse.Namespace = None):
        self._state = State()
        self.args = args

    def get_state(self):
        return self._state

    def reset_state(self):
        self._state.reference_images.clear()
        self._state.reference_priors.clear()
        self._state.reference_features.clear()
        self._state.processed_reference_masks.clear()
        self._state.target_images.clear()
        self._state.target_features.clear()
        self._state.similarities.clear()
        self._state.priors.clear()
        self._state.masks.clear()
        self._state.annotations.clear()
        self._state.used_points.clear()

    def learn(self, reference_images: List[Image], reference_priors: List[Priors]):
        """
        This method learns the context

        Args:
            reference_images: A list of images ot learn from.
            reference_priors: A list of priors associated with the image.

        Returns:
            None

        Examples:
            >>> p = Pipeline()
            >>> p.learn([Image()], [Priors()])
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
