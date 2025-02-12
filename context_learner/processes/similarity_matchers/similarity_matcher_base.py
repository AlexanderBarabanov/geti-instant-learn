from typing import List

from context_learner.processes.process_base import Process
from context_learner.types.features import Features
from context_learner.types.similarities import Similarities


class SimilarityMatcher(Process):
    def __call__(self, reference_features: List[Features], target_features: List[Features]) -> List[Similarities]:
        """
        This method calculates the similarities between reference features and target features.

        Args:
            reference_features: The reference features per image.
            target_features: The target features per image.

        Returns:
            A list of similarities per target_features.
            Note: the number of elements in output list is usually the same as
                the number of items in the target_features list.

        Examples:
            >>> from context_learner.types.state import State
            >>> state = State()
            >>> sim = SimilarityMatcher(state=state)
            >>> r = sim([Features()], [Features()])
        """
        return [Similarities()]
