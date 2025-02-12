from typing import List

from context_learner.processes.mask_processors.mask_processor_base import MaskProcessor
from context_learner.processes.process_base import Process
from context_learner.types.masks import Masks
from context_learner.types.annotations import Annotations


class MasksToPolygons(MaskProcessor):
    def __call__(self, masks: List[Masks]) -> List[Annotations]:
        return [Annotations()]
