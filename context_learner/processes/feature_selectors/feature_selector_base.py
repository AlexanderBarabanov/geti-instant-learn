from typing import List

from context_learner.processes.process_base import Process
from context_learner.types.features import Features


class FeatureSelector(Process):
    def __call__(self, features: List[Features]) -> List[Features]:
        """
        This method merges features.

        This class has the same interface as the FeatureFilter() but,
        is defined a process because it it an integral part of a pipeline flow.

        Args:
            features: A list of features.

        Returns:
            A list of new features.

        Examples:
            >>> from context_learner.types.state import State
            >>> state = State()
            >>> select = FeatureSelector(state=state)
            >>> r = select([Features()])
        """
        return [Features()]
