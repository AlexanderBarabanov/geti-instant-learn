# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from context_learner.types.data import Data


class Image(Data):
    """This is a wrapper around a torch tensor or numpy array that represents an image.
    The default data type is a numpy array because of the use of OpenCV.
    """

    def __init__(self, image: torch.Tensor | np.ndarray):
        if isinstance(image, torch.Tensor):
            self._data = image.numpy()
        else:
            self._data = image

        self._size = self._data.shape[:2]
        self._transformed_size: tuple[int, int] = None

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def tensor(self) -> torch.Tensor:
        return torch.from_numpy(self._data)

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    @property
    def transformed_size(self) -> tuple[int, int]:
        return self._transformed_size

    @transformed_size.setter
    def transformed_size(self, value: tuple[int, int]):
        self._transformed_size = value

    def __str__(self):
        return f"Image(size={self.size}, transformed_size={self.transformed_size})"

    def to_numpy(self):
        return self._data.copy()
