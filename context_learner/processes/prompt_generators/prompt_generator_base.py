from typing import List

from context_learner.processes.process_base import Process
from context_learner.types.priors import Priors
from context_learner.types.similarities import Similarities


class PromptGenerator(Process):
    def __call__(self, similarities: List[Similarities]) -> List[Priors]:
        """
        This method extracts priors from similarities.

        Args:
            similarities: The similarities between reference features and target features.

        Returns:
            A priors that have been created from the similarities.

        Examples:
            >>> from context_learner.types.state import State
            >>> state = State()
            >>> prompt_gen = PromptGenerator(state=state)
            >>> r = prompt_gen([Similarities()])
        """
        return [Priors()]
