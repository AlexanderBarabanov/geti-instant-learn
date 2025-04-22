from typing import List

from context_learner.filters import Filter
from context_learner.types import Annotations


class AnnotationFilter(Filter):
    def __call__(self, annotations: List[Annotations]) -> List[Annotations]:
        return annotations
