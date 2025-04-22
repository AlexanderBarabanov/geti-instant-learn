from typing import List

from context_learner.processes import Process
from context_learner.types import Masks, Priors


class Segmenter(Process):
    def __call__(self, priors: List[Priors]) -> List[Masks]:
        """
        This method extracts priors from similarities.

        Args:
            priors: The priors that are used for segmenting.

        Returns:
            Segmentation masks.

        Examples:
            >>> from context_learner.types.state import State
            >>> state = State()
            >>> segment = Segmenter(state=state)
            >>> r = segment([Priors()])
        """
        return [Masks()]
