# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, BinaryIO


class Data:
    """
    This is a base class for all data types.
    It provides a way to save and load the data to and from a file.
    """

    def __init__(self, data: Any):
        self._data = data

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def data(self):
        return self._data

    def save(self, f: BinaryIO):
        pass

    def load(self, f: BinaryIO):
        pass
