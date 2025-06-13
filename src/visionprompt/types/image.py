# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np
import torch

from visionprompt.types.data import Data


class Image(Data):
    """This is a wrapper around a torch tensor or numpy array that represents an image.

    The default data type is a numpy array because of the use of OpenCV.
    """

    def __init__(self, image: torch.Tensor | np.ndarray) -> None:
        """Initialize the image.

        Args:
            image: The image data.
        """
        if isinstance(image, torch.Tensor):
            self._data = image.numpy()
        else:
            self._data = image

        self._size = self._data.shape[:2]
        self._sam_preprocessed_size: tuple[int, int] = None

    @property
    def data(self) -> np.ndarray:
        """Get the image data."""
        return self._data

    @property
    def tensor(self) -> torch.Tensor:
        """Get the image data as a torch tensor."""
        return torch.from_numpy(self._data)

    @property
    def size(self) -> tuple[int, int]:
        """Get the size of the image."""
        return self._size

    @property
    def sam_preprocessed_size(self) -> tuple[int, int]:
        """Get the size of the image after SAM preprocessing."""
        return self._sam_preprocessed_size

    @sam_preprocessed_size.setter
    def sam_preprocessed_size(self, value: tuple[int, int]) -> None:
        self._sam_preprocessed_size = value

    def __str__(self) -> str:
        """Get the string representation of the image."""
        return f"Image(size={self.size}, sam_preprocessed_size={self.sam_preprocessed_size})"

    def to_numpy(self) -> np.ndarray:
        """Get the image data as a numpy array."""
        return self._data.copy()

    def resize_inplace(self, size: int | tuple[int, int] | None = None) -> None:
        """Resizes the image in place.

        Args:
            size: The size to resize the image to. If a tuple is provided, the image will be resized to the given width
                and height. If an integer is provided, the image will be resized to the given size, maintaining aspect
                ratio. If None is provided, the image will not be resized.

        Returns:
            The resized image.
        """
        if size is None:
            return

        original_h, original_w = self._data.shape[:2]

        if isinstance(size, tuple):
            height, width = size
            target_h = height
            target_w = width
        else:
            target_largest_dim = size
            if original_h > original_w:
                target_h = target_largest_dim
                target_w = round(original_w * (target_largest_dim / original_h)) if original_h > 0 else 0
            else:
                target_w = target_largest_dim
                target_h = round(original_h * (target_largest_dim / original_w)) if original_w > 0 else 0

        if (target_h, target_w) == (original_h, original_w):
            return

        self._data = cv2.resize(self._data, (target_w, target_h))
        self._size = (target_h, target_w)
