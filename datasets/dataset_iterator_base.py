# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Sized, Iterable


class DatasetIter(Sized, Iterable):
    def __init__(self, parent: "Dataset"):  # type: ignore  # noqa: F821
        self._parent = parent

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    def __next__(self):
        raise NotImplementedError()

    def get_image_filename(self, *indices: int):
        """
         Returns the filename of the original image given a list of indices

        Args:
            *indices: Depending on the iterator type this can be a certain number of indices
                (e.g. category #, batch #, image #)

        Returns:
            The filename of the original image

        """
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()
