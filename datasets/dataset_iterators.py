import math
from typing import Tuple, List, Dict
import numpy as np

from datasets.dataset_base import Dataset
from datasets.dataset_iterator_base import DatasetIter


class IndexIter(DatasetIter):
    """
    Standard PyTorch style iterator producing batches of images and masks
    """

    def __init__(self, parent: Dataset):
        super().__init__(parent)
        self.index = 0

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        return self._parent.get_image_by_index(index), self._parent.get_masks_by_index(
            index
        )

    def __iter__(self):
        self.index = 1
        return self

    def __len__(self):
        return self._parent.get_image_count()

    def __next__(self):
        if self.index < len(self):
            item = self.__getitem__(self.index)
            self.index += 1
            return item
        else:
            raise StopIteration

    def get_image_filename(self, *indices: int):
        return self._parent.get_image_filename(indices[0])


class CategoryIter(DatasetIter):
    """
    This class iterates over categories and return images and masks
    """

    def __init__(self, parent: Dataset):
        super().__init__(parent)
        self.index = 0

    def __getitem__(self, index: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        return self._parent.get_images_by_category(
            index
        ), self._parent.get_masks_by_category(index)

    def __iter__(self):
        self.index = 1
        return self

    def __len__(self):
        return self._parent.get_category_count()

    def __next__(self):
        if self.index < len(self):
            item = self.__getitem__(self.index)
            self.index += 1
            return item
        else:
            raise StopIteration

    def get_image_filename(self, *indices: int):
        return self._parent.get_image_filename_in_category(indices[0], indices[1])


class BatchedSingleCategoryIter(DatasetIter):
    """
    This class iterates over batches of images and masks of a given category
    """

    def __init__(self, parent: Dataset, batch_size: int, category_index: int):
        super().__init__(parent)
        self._batch_size = batch_size
        self._category_index = category_index
        self._batch_index = 0

    def __getitem__(
        self, batch_index: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        images = self._parent.get_images_by_category(
            self._category_index,
            start=self._batch_index * self._batch_size,
            end=(batch_index + 1) * self._batch_size,
        )
        masks = self._parent.get_masks_by_category(
            self._category_index,
            start=self._batch_index * self._batch_size,
            end=(batch_index + 1) * self._batch_size,
        )
        return images, masks

    def __iter__(self):
        self.batch_index = 1
        return self

    def __len__(self):
        return math.ceil(
            self._parent.get_image_count_per_category(self._category_index)
            / self._batch_size
        )

    def __next__(self):
        if self._batch_index < len(self):
            item = self.__getitem__(self._batch_index)
            self._batch_index += 1
            return item
        else:
            raise StopIteration

    def reset(self):
        self._batch_index = 0

    def get_image_filename(self, *indices: int):
        return self._parent.get_image_filename_in_category(
            self._category_index, indices[0] * self._batch_size + indices[1]
        )


class BatchedCategoryIter(DatasetIter):
    """
    This class iterates over categories and returns a new iterator for creating batches per category
    """

    def __init__(self, parent: Dataset, batch_size: int):
        super().__init__(parent)
        self._batch_size = batch_size
        self._category_index = 0

    def __getitem__(self, category_index: int) -> BatchedSingleCategoryIter:
        return BatchedSingleCategoryIter(
            self._parent, self._batch_size, self._category_index
        )

    def __iter__(self):
        self._category_index = 1
        return self

    def __len__(self):
        return self._parent.get_category_count()

    def __next__(self):
        if self._category_index < len(self):
            item = self.__getitem__(self._category_index)
            self._category_index += 1
            return item
        else:
            raise StopIteration

    def get_image_filename(self, *indices: int):
        return self._parent.get_image_filename_in_category(
            indices[0], indices[1] * self._batch_size + indices[2]
        )
