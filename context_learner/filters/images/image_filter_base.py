from typing import List

from context_learner.filters.filter_base import Filter
from context_learner.types import Image


class ImageFilterBase(Filter):
    def __call__(self, images: List[Image]) -> List[Image]:
        return images
