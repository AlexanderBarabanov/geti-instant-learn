from typing import List, Dict, Iterable, Type
import requests
import zipfile
import os
import numpy as np
import torch
import logging

from datasets.dataset_iterators import DatasetIter


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


class Dataset(torch.utils.data.Dataset, Iterable):
    def __init__(self, iterator_type: Type[DatasetIter], iterator_kwargs={}):
        self._iterator_type = iterator_type
        self._iterator_kwargs = iterator_kwargs
        self.index = 0

    def get_categories(self):
        raise NotImplementedError()

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

    def get_images_by_category(
        self, category_index_or_name: int | str, start: int = None, end: int = None
    ) -> List[np.ndarray]:
        """
        This method returns a list of images of a certain category.
            The parameters start and end are passed through Python's slice() function.

        Args:
            category_index_or_name: The category name or category index
            start: The first index to return
            end: end-1 is the last index to return

        Returns:
            A list of numpy arrays

        """
        raise NotImplementedError()

    def get_masks_by_category(
        self, category_index_or_name: int | str, start: int = None, end: int = None
    ) -> List[np.ndarray]:
        """
        This method returns a list of masks of a certain category.
            each individual instance of the category has a unique pixel value.
            The parameters start and end are passed through Python's slice() function.

        Args:
            category_index_or_name: The category name or category index
            start: The first index to return
            end: end-1 is the last index to return

        Returns:
            A dict of masks per category.
        """
        raise NotImplementedError()

    def get_image_count(self):
        """
        This method returns the number of images in the dataset
        """
        return 0

    def get_image_count_per_category(self, category_index_or_name: int | str):
        """
        This method returns the number of images per category.

        Args:
            category_index_or_name: The category name or category index

        Returns:
            The number of images in a certain category
        """
        return 0

    def get_instance_count_per_category(self, category_index_or_name: int | str):
        """
        This method returns the number of instances per category.

        Args:
            category_index_or_name: The category name or category index

        Returns:
            The number of instances in a certain category
        """
        return 0

    def get_category_count(self):
        """This method returns the number of categories"""
        return 0

    def get_image_filename(self, index: int):
        """
        Gives the source filename of the image.

        Args:
            index: The index for which to return the filename

        Returns:
            The filename of the image.

        """
        return ""

    def get_image_filename_in_category(
        self, category_index_or_name: int | str, index: int
    ):
        """
        Gives the source filename of the image.

        Args:
            category_index_or_name: The category name or category index
            index: The index for which to return the filename (index within the category)

        Returns:
            The filename of the image.

        """
        return ""

    def category_index_to_id(self, index: int) -> int:
        """Return the category id given the category index"""
        return -1

    def category_id_to_name(self, id: int) -> str:
        """Return the category name given a category id"""
        return ""

    def category_name_to_id(self, name: str) -> int:
        """Return the category id given a category name"""
        return -1

    def category_name_to_index(self, name: str) -> int:
        """Return the category index given a category name"""
        return -1

    def category_index_to_name(self, index: int) -> str:
        """Return the category index given a category name"""
        return self.category_id_to_name(self.category_index_to_id(index))

    def _download(self, source: str, destination: str):
        """Helper function to download data with caching"""
        if os.path.isfile(destination):
            logging.info(f"Using cached downloaded file {destination}")
            return
        else:
            logging.info(f"Downloading data from {source} to {destination}...")

        response = requests.get(source)
        if response.status_code == 200:
            with open(destination, "wb") as file:
                file.write(response.content)
            logging.info(f"Downloaded {source} to {destination}")
        else:
            logging.info(f"Failed to download {source}")

    def _unzip(self, source: str, destination: str):
        """Helper function to unzip data with caching"""
        if os.path.isfile(destination) or os.path.isdir(destination):
            logging.info(f"Using cached unzipped file or folder {destination}")
            return

        with zipfile.ZipFile(source, "r") as zf:
            zf.extractall(os.path.dirname(source))

        logging.info(f"Unzipped {source} to {destination}")

    def __len__(self) -> int:
        """
        Returns: the number of items in this dataset. What an item exactly entails is
            determined by the iterator.
        """
        return len(self._iterator_type(parent=self, **self._iterator_kwargs))

    def __iter__(self):
        return self._iterator_type(parent=self, **self._iterator_kwargs)

    def __get_item__(self, index: int):
        """
        get_item is implemented for compatibility with torch's Dataset. What an item exactly
            entails is determined by the iterator.

        Args:
            index: The index to retrieve.

        Returns:
            A new item from the dataset iterator
        """
        return self._iterator_type(parent=self, **self._iterator_kwargs)[index]
