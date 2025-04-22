from typing import List

from context_learner.filters import Filter
from context_learner.types import Similarities


class SimilarityFilter(Filter):
    def __call__(self, priors: List[Similarities]) -> List[Similarities]:
        return priors
