# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from visionprompt.context_learner.filters import Filter
from visionprompt.context_learner.types import Image


class ImageFilter(Filter):
    """This is the base class for all images filters."""

    def __call__(self, images: list[Image]) -> list[Image]:
        """Filter the images."""
