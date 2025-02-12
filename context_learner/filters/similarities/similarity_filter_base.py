from typing import List

from context_learner.filters.filter_base import Filter
from context_learner.types.similarities import Similarities


class SimilarityFilterBase(Filter):
    def __call__(self, priors: List[Similarities]) -> List[Similarities]:
        return priors
