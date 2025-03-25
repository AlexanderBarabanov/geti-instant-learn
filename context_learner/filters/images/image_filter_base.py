from typing import List

from context_learner.filters.filter_base import Filter
from context_learner.types.priors import Prompt


class PriorFilterBase(Filter):
    def __call__(self, priors: List[Prompt]) -> List[Prompt]:
        return priors
