from typing import Tuple, List, Dict, Sized, Iterable

import requests
import zipfile
import os
import numpy as np
import torch

def log(msg: str):
    print(msg)

class Annotation:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def get_mask(self) -> np.ndarray:
        raise NotImplementedError()


class Image:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def get_image(self) -> np.ndarray:
        raise NotImplementedError()


class DatasetIter(Sized, Iterable):
    def __len__(self):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    def __init__(self, parent: 'Dataset'):
        self._parent = parent


class IndexIter(DatasetIter):
    def __init__(self, parent: 'Dataset'):
        super().__init__(parent)
        self.index = 0

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        return self._parent.get_image_by_index(index), self._parent.get_masks_by_index(index)

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


class CategoryIter(DatasetIter):
    def __init__(self, parent: 'Dataset'):
        super().__init__(parent)
        self.index = 0

    def __getitem__(self, index: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        return self._parent.get_images_by_category(index), self._parent.get_masks_by_category(index)

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


class Dataset(torch.utils.data.Dataset, Iterable):

    def __init__(self, iterator_type: type(DatasetIter)):
        self._iterator_type = iterator_type
        self.index = 0

    def get_image_by_index(self, index: int) -> np.ndarray:
        """
        This method returns an image based on its index.

        Args:
            index: The index of the image

        Returns:
            A numpy array containing an image

        """
        raise NotImplementedError()

    def get_masks_by_index(self, index: int) -> Dict[int, np.ndarray]:
        """
        This method returns a set of masks based on the image index.
            This method returns one mask per category where each individual instance
            has a unique pixel value.

        Args:
            index: the image index for which to return masks

        Returns:
            A dict of masks per category.
        """
        raise NotImplementedError()

    def get_images_by_category(self, category_index_or_name: int | str) -> List[np.ndarray]:
        """
        This method returns a list of images of a certain category.

        Args:
            category_index_or_name: The category name or category index

        Returns:
            A list of numpy arrays

        """
        raise NotImplementedError()

    def get_masks_by_category(self, category_index_or_name: int | str) -> List[np.ndarray]:
        """
        This method returns a list of masks of a certain category.
            each individual instance of the category has a unique pixel value.

        Args:
            category_index_or_name: The category name or category index

        Returns:
            A dict of masks per category.
        """
        raise NotImplementedError()

    def get_image_count(self):
        return 0

    def get_category_count(self):
        return 0

    def _download(self, source: str, destination: str):
        """ Helper function to download data with caching """
        if os.path.isfile(destination):
            log(f"Using cached downloaded file {destination}")
            return
        else:
            log(f"Downloading data from {source} to {destination}...")

        response = requests.get(source)
        if response.status_code == 200:
            with open(destination, 'wb') as file:
                file.write(response.content)
            log(f'Downloaded {source} to {destination}')
        else:
            log(f'Failed to download {source}')

    def _unzip(self, source: str, destination: str):
        """ Helper function to unzip data with caching """
        if os.path.isfile(destination) or os.path.isdir(destination):
            log(f"Using cached unzipped file or folder {destination}")
            return

        with zipfile.ZipFile(source, 'r') as zf:
            zf.extractall(os.path.dirname(source))

        log(f'Unzipped {source} to {destination}')

    def __iter__(self) -> DatasetIter:
        """
        Get a new instance of the iterator. This prevents the class from creating a new iterator
            for every index operation and if a slightly faster method.

        Returns:
            A descendent of DatasetIter
        """
        return self._iterator_type(self)

    def __len__(self) -> int:
        """
        Returns: the number of items in this dataset. What an item exactly entails is
            determined by the iterator.
        """
        return len(self._iterator_type(self))

    def __get_item__(self, index: int):
        """
        get_item is implemented for compatibility with torch's Dataset. What an item exactly
            entails is determined by the iterator.

        Args:
            index: The index to retrieve.

        Returns:
            A new item from the dataset iterator
        """
        return self._iterator_type(self)[index]
