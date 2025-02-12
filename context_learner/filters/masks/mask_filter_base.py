from typing import List

from context_learner.filters.filter_base import Filter
from context_learner.types.masks import Masks


class MaskFilterBase(Filter):
    def __call__(self, masks: List[Masks]) -> List[Masks]:
        return masks
