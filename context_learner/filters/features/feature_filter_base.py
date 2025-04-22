from typing import List

from context_learner.filters import Filter
from context_learner.types import Features


class FeatureFilterBase(Filter):
    def __call__(self, features: List[Features]) -> List[Features]:
        return features
