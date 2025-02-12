from typing import List

from context_learner.processes.process_base import Process
from context_learner.types.masks import Masks
from context_learner.types.annotations import Annotations


class MaskProcessor(Process):
    def __call__(self, masks: List[Masks]) -> List[Annotations]:
        """
        This method extracts polygons from masks.

        Args:
            masks: A list of masks.

        Returns:
            A list of polygons that have been created from the masks.

        Examples:
            >>> from context_learner.types.state import State
            >>> state = State()
            >>> process_masks = MaskProcessor(state=state)
            >>> r = process_masks([Masks()])
        """
        return [Annotations()]
