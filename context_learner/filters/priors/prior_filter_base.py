from typing import List

from context_learner.filters.filter_base import Filter
from context_learner.types.priors import Priors


class PriorFilterBase(Filter):
    def __call__(self, priors: List[Priors]) -> List[Priors]:
        return priors
