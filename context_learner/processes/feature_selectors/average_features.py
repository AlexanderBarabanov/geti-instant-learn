from typing import List

from context_learner.processes.feature_selectors.feature_selector_base import FeatureSelector
from context_learner.types.features import Features


class AverageFeatures(FeatureSelector):
    def __call__(self, features: List[Features]) -> List[Features]:
        return [Features()]
