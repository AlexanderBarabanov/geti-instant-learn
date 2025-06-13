# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from visionprompt.filters import Filter
from visionprompt.types import Image


class ImageFilter(Filter):
    """This is the base class for all images filters."""

    def __call__(self, images: list[Image]) -> list[Image]:
        """Filter the images."""
