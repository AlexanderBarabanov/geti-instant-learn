from typing import List

from context_learner.processes.similarity_matchers.similarity_matcher_base import SimilarityMatcher
from context_learner.types.features import Features
from context_learner.types.similarities import Similarities


class CosineSimilarity(SimilarityMatcher):
    def __call__(self, reference_features: List[Features], target_features: List[Features]) -> List[Similarities]:
        return [Similarities()]
