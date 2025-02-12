from typing import List

from context_learner.filters.filter_base import Filter
from context_learner.types.features import Features


class FeatureFilterBase(Filter):
    def __call__(self, features: List[Features]) -> List[Features]:
        return features
