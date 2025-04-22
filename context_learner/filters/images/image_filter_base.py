from typing import List

from context_learner.filters import Filter
from context_learner.types import Image


class ImageFilter(Filter):
    def __call__(self, images: List[Image]) -> List[Image]:
        return images
