# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datasets.dataset_base import Dataset, Annotation, Image, DatasetIter
import os
from typing import List, Dict, Type
import numpy as np
from PIL import Image as PILImage
import cv2

from datasets.dataset_iterators import IndexIter, CategoryIter
from utils.utils import color_overlay


class PerSegAnnotation(Annotation):
    def __init__(self, filename: str, category_id: int):
        super().__init__(0, 0)
        self.category_id = category_id
        self.filename = filename

    def get_mask(self) -> np.ndarray:
        pil_image = PILImage.open(self.filename).convert("RGB")
        arr = np.array(pil_image)
        return (arr[:, :, 0] > 0).astype(np.uint8)


class PerSegImage(Image):
    def __init__(self, filename: str):
        """
        Initializes the Image object.
            If the filename is not found it is either taken from source_filename or
            downloaded from the url (in that order).

        Args:
            filename: The filename of the image
        """
        super().__init__(0, 0)
        self.filename = filename

    def get_image(self) -> np.ndarray:
        pil_image = PILImage.open(self.filename).convert("RGB")
        arr = np.array(pil_image)
        return arr


class PerSegDataset(Dataset):
    def __init__(
        self,
        root_path=os.path.expanduser(os.path.join("~", "data", "perseg")),
        iterator_type: Type[DatasetIter] = IndexIter,
        iterator_kwargs={},
    ):
        """
        This method initializes the PerSeg dataset.

        Args:
            root_path: The path to the root directory of the perseg dataset.
            iterator_kwargs: Keyword arguments passed to the iterator_type
        """
        super().__init__(iterator_type=iterator_type, iterator_kwargs=iterator_kwargs)
        self._root_path = root_path
        os.makedirs(self._root_path, exist_ok=True)
        self._files = {
            "downloads_source": "https://drive.usercontent.google.com/download?id=18TbrwhZtAPY5dlaoEqkPa5h08G9Rjcio&confirm=t&export=download",
            "downloads_destination": os.path.join(
                self._root_path, "downloads", "PerSeg.zip"
            ),
            "unzipped_destination": os.path.join(
                self._root_path, "downloads", "data 3"
            ),  # files are extracted here
        }

        self.instance_count: Dict[str, int] = {}  # name: count
        self.image_count: Dict[str, int] = {}  # name: count
        self.instances_per_image: Dict[str, float] = {}  # name: count per image

        # Category information
        self._category_index_to_name: List[str] = []
        self._category_name_to_index: Dict[str, int] = {}

        # Images and Annotations
        self._images: List[PerSegImage] = []
        self._annotations: List[PerSegAnnotation] = []

        self._annotation_to_category: Dict[
            int, int
        ] = {}  # [annotation_id: category_id]
        self._category_to_annotations: Dict[
            int, List[int]
        ] = {}  # [category_id: [annotation_id]]
        self._image_to_annotations: Dict[int, int] = {}  # [image_id: annotation_id]

        # Download metadata (these are automatically cached)
        self._download_dataset()
        self._load_data()

    def _load_data(self):
        # For this dataset the image_id and annotation_id are equal to the indices
        images_folder = os.path.join(self._files["unzipped_destination"], "Images")
        annotations_folder = os.path.join(
            self._files["unzipped_destination"], "Annotations"
        )
        self._category_index_to_name = [
            name for name in os.listdir(images_folder) if not name.startswith(".")
        ]
        self._category_name_to_index = {
            name: index for index, name in enumerate(self._category_index_to_name)
        }
        self._category_to_annotations = {
            i: [] for i, _ in enumerate(self._category_index_to_name)
        }
        # Loop through all categories
        for cat_index, category in enumerate(self._category_index_to_name):
            images_sub_folder = os.path.join(images_folder, category)
            images_filenames = [
                name
                for name in os.listdir(images_sub_folder)
                if not name.startswith(".")
            ]
            for image_filename in images_filenames:
                # Fill statistics
                self.image_count[category] = len(images_filenames)
                self.instance_count[category] = len(images_filenames)
                self.instances_per_image[category] = 1.0

                # Get all paths
                image_full_path = os.path.join(images_sub_folder, image_filename)
                annotation_full_path = os.path.join(
                    annotations_folder,
                    category,
                    os.path.splitext(image_filename)[0] + ".png",
                )

                # Fill objects
                self._images.append(PerSegImage(image_full_path))
                self._annotations.append(
                    PerSegAnnotation(annotation_full_path, cat_index)
                )

                # fill references for easy access
                annot_id = len(self._annotations) - 1
                image_id = len(self._images) - 1
                self._annotation_to_category[cat_index] = annot_id
                self._category_to_annotations[cat_index].append(annot_id)
                self._image_to_annotations[image_id] = annot_id
        return

    def get_root_path(self):
        return self._root_path

    def get_categories(self):
        return self._category_index_to_name

    def category_index_to_id(self, index: int) -> int:
        return index

    def category_id_to_name(self, id: int) -> str:
        return self._category_index_to_name[id]

    def category_name_to_id(self, name: str) -> int:
        return self._category_name_to_index[name]

    def category_name_to_index(self, name: str) -> int:
        return self.category_name_to_id(name)

    def get_image_filename(self, index: int):
        return self._images[index].filename

    def get_image_filename_in_category(
        self, category_index_or_name: int | str, index: int
    ):
        if isinstance(category_index_or_name, int):
            category_id = category_index_or_name
        elif isinstance(category_index_or_name, str):
            category_id = self._category_name_to_index[category_index_or_name]
        else:
            raise ValueError(f"Unknown category type: {type(category_index_or_name)}")

        image_ids = self._category_to_annotations[category_id]
        return self._images[image_ids[index]].filename

    def get_image_by_index(self, index: int) -> np.ndarray:
        image = self._images[index]
        return image.get_image()

    def get_masks_by_index(self, index: int) -> Dict[int, np.ndarray]:
        masks = {
            self._annotations[index].category_id: self._annotations[index].get_mask()
        }
        return masks

    def get_images_by_category(
        self, category_index_or_name: int | str, start: int = None, end: int = None
    ) -> List[np.ndarray]:
        if isinstance(category_index_or_name, int):
            category_id = category_index_or_name
        elif isinstance(category_index_or_name, str):
            category_id = self._category_name_to_index[category_index_or_name]
        else:
            raise ValueError(f"Unknown category type: {type(category_index_or_name)}")

        image_ids = self._category_to_annotations[category_id]
        image_ids = image_ids[slice(start, end)]

        return [self._images[i].get_image() for i in image_ids]

    def get_masks_by_category(
        self, category_index_or_name: int | str, start: int = None, end: int = None
    ) -> List[np.ndarray]:
        if isinstance(category_index_or_name, int):
            category_id = category_index_or_name
        elif isinstance(category_index_or_name, str):
            category_id = self._category_name_to_index[category_index_or_name]
        else:
            raise ValueError(f"Unknown category type: {type(category_index_or_name)}")

        annotation_ids = self._category_to_annotations[category_id]
        annotation_ids = annotation_ids[slice(start, end)]

        return [self._annotations[i].get_mask() for i in annotation_ids]

    def get_category_count(self):
        return len(self._category_index_to_name)

    def get_image_count(self):
        return len(self._images)

    def get_image_count_per_category(self, category_index_or_name: int | str):
        if isinstance(category_index_or_name, int):
            category_id = category_index_or_name
        elif isinstance(category_index_or_name, str):
            category_id = self._category_name_to_index[category_index_or_name]
        else:
            raise ValueError(f"Unknown category type: {type(category_index_or_name)}")

        image_ids = self._category_to_annotations[category_id]
        return len(image_ids)

    def get_instance_count_per_category(self, category_index_or_name: int | str):
        if isinstance(category_index_or_name, int):
            category_id = category_index_or_name
        elif isinstance(category_index_or_name, str):
            category_id = self._category_name_to_index[category_index_or_name]
        else:
            raise ValueError(f"Unknown category type: {type(category_index_or_name)}")

        cat_name = self._category_index_to_name[category_id]
        return self.instance_count[cat_name]

    def _download_dataset(self):
        """
        Downloads the dataset
        """
        os.makedirs(os.path.join(self._root_path, "downloads"), exist_ok=True)
        download_src = self._files["downloads_source"]
        download_dest = self._files["downloads_destination"]
        unzip_dest = self._files["unzipped_destination"]
        self._download(download_src, download_dest)
        self._unzip(download_dest, unzip_dest)
        if not os.path.exists(unzip_dest):
            raise RuntimeError(f"Failed to produce {unzip_dest}")


def test_index_iter():
    # Use default index iterator (PyTorch style)
    dataset = PerSegDataset()

    for image_index, (image, masks) in enumerate(dataset):
        # Generate and save overlays
        for category_id, mask in masks.items():
            overlay = color_overlay(image, mask)
            cat = dataset.get_categories()[category_id]
            output_folder = os.path.join(dataset.get_root_path(), "overlays", cat)
            orig_filename = os.path.splitext(
                os.path.basename(dataset.get_image_filename(image_index))
            )[0]
            os.makedirs(output_folder, exist_ok=True)
            cv2.imwrite(
                os.path.join(output_folder, f"{orig_filename}_{cat}.jpg"), overlay
            )


def test_category_iter():
    # Use category iterator
    dataset = PerSegDataset(iterator_type=CategoryIter)

    for category_index, (images, masks) in enumerate(dataset):
        for image_index, (image, mask) in enumerate(zip(images, masks)):
            # Generate and save overlays
            overlay = color_overlay(image, mask)
            cat = dataset.get_categories()[category_index]
            output_folder = os.path.join(dataset.get_root_path(), "overlays", cat)
            orig_filename = os.path.splitext(
                os.path.basename(
                    dataset.get_image_filename_in_category(category_index, image_index)
                )
            )[0]
            os.makedirs(output_folder, exist_ok=True)
            cv2.imwrite(
                os.path.join(output_folder, f"{orig_filename}_{cat}.jpg"), overlay
            )


if __name__ == "__main__":
    test_index_iter()
    test_category_iter()
