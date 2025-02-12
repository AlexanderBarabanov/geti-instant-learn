from typing import List

from context_learner.processes.segmenters.segmenter_base import Segmenter
from context_learner.types.masks import Masks
from context_learner.types.priors import Priors


class SamDecoder(Segmenter):
    def __call__(self, priors: List[Priors]) -> List[Masks]:
        return [Masks()]
