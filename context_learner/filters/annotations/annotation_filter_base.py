from typing import List

from context_learner.filters.filter_base import Filter
from context_learner.types.annotations import Annotations


class AnnotationFilterBase(Filter):
    def __call__(self, annotations: List[Annotations]) -> List[Annotations]:
        return annotations
