import colorsys
import json
import shutil
from collections import OrderedDict

from datasets.dataset_base import Dataset, Annotation, Image, DatasetIter
import os
from typing import List, Dict, Any, Iterable, Union, Optional
import pickle
import pycocotools.mask as mask_utils
import numpy as np
import requests
from PIL import Image as PILImage
import cv2
import logging

from datasets.dataset_iterators import IndexIter, CategoryIter
from utils.utils import color_overlay


def segment_to_mask(segment: List[float], height: int, width: int):
    """
    This method converts a segment to a mask using RLE

    Args:
        segment: The segment to be converted
        height: height of the image
        width: width of the image

    Returns:
        Decoded mask
    """
    if isinstance(segment, list):
        # Merge all parts
        rle_segments = mask_utils.frPyObjects(segment, height, width)
        rle_segment = mask_utils.merge(rle_segments)
    elif isinstance(segment["counts"], list):
        # uncompressed
        rle_segment = mask_utils.frPyObjects(segment, height, width)
    else:
        # regular RLE
        rle_segment = segment
    return mask_utils.decode(rle_segment)


class LVISAnnotation(Annotation):
    def __init__(self, height: int, width: int, segments: List[float], category_id: int):
        super().__init__(height, width)
        self.segments = segments
        self.category_id = category_id

    def get_mask(self) -> np.ndarray:
        arr = segment_to_mask(self.segments, self.height, self.width)
        return arr


class LVISImage(Image):
    def __init__(self, filename: str, height: int, width: int, source_url: str = None, source_filename: str = None, copy_file = False):
        """
        Initializes the Image object.
            If the filename is not found it is either taken from source_filename or
            downloaded from the url (in that order).

        Args:
            filename: The filename of the image
            height: The height of the image
            width: The width of the image
            source_url: The url of the image from which the image can be downloaded
            source_filename: The filename where the image can be found
            copy_file: If True this will copy the file from source_filename to filename
        """
        super().__init__(height, width)
        self.source_url = source_url
        self.source_filename = source_filename
        self.filename = filename
        self.copy_file = copy_file

    def get_image(self) -> np.ndarray:
        folder = os.path.dirname(self.filename)
        os.makedirs(folder, exist_ok=True)

        image_filename = self.filename

        if self.source_filename is not None:
            if self.copy_file:
                if os.path.isfile(self.source_filename) and not os.path.isfile(self.filename):
                    logging.info(f"Copy {self.source_filename} to {self.filename}")
                    shutil.copyfile(self.source_filename, self.filename)
            else:
                image_filename = self.source_filename

        if not os.path.isfile(image_filename):
            logging.info(f"Downloading {self.source_url}")
            img_data = requests.get(self.source_url).content
            with open(self.filename, 'wb') as handler:
                handler.write(img_data)

        pil_image = PILImage.open(image_filename).convert('RGB')
        arr = np.array(pil_image)
        return arr


class LVISDataset(Dataset):

    def __init__(self, root_path=os.path.expanduser(os.path.join("~", "data", "lvis")), whitelist: Optional[Union[str, List[str]]]=None, name="training", iterator_type: type(DatasetIter) = IndexIter, download_full_dataset=False, copy_files=False, iterator_kwargs={}):
        """
        This method initializes the LVIS dataset. This class downloads and inflates all files.

        Args:
            root_path: The path to the root directory of the LVIS dataset.
            whitelist: The classes that are selected, if empty, all categories are used
            download_full_dataset: If True download the full dataset otherwise,
                                   each image is downloaded on demand.
            copy_files: If the full dataset is download then copy_files will copy files from the
                        COCO dataset to the LVIS dataset folders.
                        If copy_files is True, then after copying, download_full_dataset can be set to false.
            iterator_kwargs: Keyword arguments passed to the iterator_type
        """
        super().__init__(iterator_type=iterator_type, iterator_kwargs=iterator_kwargs)
        self._root_path = root_path
        os.makedirs(self._root_path, exist_ok=True)

        self._subset_files = {
            "sources": {  # original sources of the json annotations.
                "training": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
                "validation": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip"
                },
            "files": {  # extracted file location of the annotations.
                "training": os.path.join(self._root_path, "lvis_v1_train.json"),
                 "validation": os.path.join(self._root_path, "lvis_v1_val.json")
            },
            "downloads": {  # downloads of the full dataset.
                "training": "http://images.cocodataset.org/zips/train2017.zip",
                "validation": "http://images.cocodataset.org/zips/val2017.zip"
            },
            "source_folders": {  # folders where the downloaded files are extracted.
                "training": os.path.join(self._root_path, "downloads", "train2017"),
                "validation": os.path.join(self._root_path, "downloads", "val2017")
            },
        }

        if whitelist is None:
            self._whitelist = []
        elif isinstance(whitelist, str):
            self._whitelist = [whitelist]
        else:
            self._whitelist = whitelist
        self._name = name  # training or validation

        # Category information
        self._category_index_to_id: List[int] = []  # only white listed categories appear here
        self._category_id_to_name: Dict[int, str] = {}
        self._category_name_to_id: Dict[str, int] = {}

        self.instance_count = None
        self.image_count = None
        self.instances_per_image = None

        # Images and Annotations
        self._image_index_to_id: Dict[str, List[int]] = {}  # subset_name: [image_id]
        self._images: Dict[str, Dict[int, LVISImage]] = {}  # subset_name: [image_id: image]]
        self._annotations: Dict[str, Dict[int, LVISAnnotation]] = {} # subset_name: [annotation_id, image]

        self._annotation_to_image: Dict[str, Dict[int, int]] = {}  # subset_name: [annotation_id: image_id]]
        self._annotation_to_category: Dict[str, Dict[int, int]] = {}  # subset_name: [annotation_id: category_id]]
        self._category_to_annotations: Dict[str, Dict[int, List[int]]] = {}  # subset_name: [category_id: [annotation]]
        self._image_to_annotations: Dict[str, Dict[int, List[int]]] = {}  # subset_name: [image_id: [annotation_id]]

        # Download metadata (these are automatically cached)
        self._download_metadata()
        if download_full_dataset:
            self._download_images()
        self._copy_files = copy_files

        # Check if cache needs to be invalidated (delete any of these files to invalidate cache)
        self._cache_check_file = os.path.join(self._root_path, "cache_check.bin")
        self._cached_metadata = os.path.join(self._root_path, "metadata.bin")
        self._cached_data = os.path.join(self._root_path, "data.bin")
        valid = self._check_cache()

        # Load metadata and data
        if valid:
            logging.info(f"Using cached {self._cached_metadata} and {self._cached_data}")
            self._load_metadata(self._cached_metadata)
            self._load_data(self._cached_data)
        else:
            logging.info(f"Cache files have been invalidated, data is reloaded")
            categories_info, images_info, annotations_info = self._get_metadata()
            self._set_metadata(categories_info)
            self._set_data(images_info, annotations_info)
            self._save_metadata(self._cached_metadata)
            self._save_data(self._cached_data)
        if len(self._whitelist) == 0:  # Populate whitelist from all categories
            self._whitelist = list(self._category_name_to_id.keys())

    def get_root_path(self):
        return self._root_path

    def get_categories(self):
        return [self._category_id_to_name[cid] for cid in self._category_index_to_id]

    def category_index_to_id(self, index: int) -> int:
        return self._category_index_to_id[index]

    def category_id_to_name(self, id: int) -> str:
        return self._category_id_to_name[id]

    def category_name_to_id(self, name: str) -> int:
        return self._category_name_to_id[name]

    def category_name_to_index(self, name: str) -> int:
        cat_id = self._category_name_to_id[name]
        return self._category_index_to_id.index(cat_id)

    def get_image_filename(self, index: int):
        return self._images[self._name][self._image_index_to_id[self._name][index]].filename

    def get_image_filename_in_category(self, category_index_or_name: int | str, index: int):
        if isinstance(category_index_or_name, int):
            category_id = self._category_index_to_id[category_index_or_name]
        elif isinstance(category_index_or_name, str):
            category_id = self._category_name_to_id[category_index_or_name]
        else:
            raise ValueError(f"Unknown category type: {type(category_index_or_name)}")

        image_ids = []
        annotations = self._category_to_annotations[self._name][category_id]
        for annotation_id in annotations:
            image_id = self._annotation_to_image[self._name][annotation_id]
            image_ids.append(image_id)

        image_ids = list(OrderedDict.fromkeys(image_ids))  # preserves order
        return self._images[self._name][image_ids[index]].filename

    def set_name(self, name: str):
        self._name = name

    def get_image_by_index(self, index: int) -> np.ndarray:
        image_id = self._image_index_to_id[self._name][index]
        image = self._images[self._name][image_id]
        return image.get_image()

    def get_masks_by_index(self, index: int) -> Dict[int, np.ndarray]:
        image_id = self._image_index_to_id[self._name][index]
        annotation_ids = self._image_to_annotations[self._name][image_id]
        masks = {}

        # Merge all masks from the same class and set each pixel value to the instance_id
        for annotation_id in annotation_ids:
            annot = self._annotations[self._name][annotation_id]
            if annot.category_id not in masks.keys():
                masks[annot.category_id] = [annot.get_mask().astype(int)]
            else:
                instance_id = len(masks[annot.category_id]) + 1
                masks[annot.category_id].append(annot.get_mask().astype(int) * instance_id)

        for category_id in masks.keys():
            # Merge all instances into one mask
            masks[category_id] = np.max(masks[category_id], axis=0)

        return masks

    def get_images_by_category(self, category_index_or_name: int | str, start: int = None, end: int = None) -> List[np.ndarray]:
        if isinstance(category_index_or_name, int):
            category_id = self._category_index_to_id[category_index_or_name]
        elif isinstance(category_index_or_name, str):
            category_id = self._category_name_to_id[category_index_or_name]
        else:
            raise ValueError(f"Unknown category type: {type(category_index_or_name)}")

        image_ids = []
        annotations = self._category_to_annotations[self._name][category_id]
        image_ids = [self._annotation_to_image[self._name][annotation_id] for annotation_id in annotations]
        image_ids = list(OrderedDict.fromkeys(image_ids))  # remove redundant, preserves same order as get_masks_by_category
        image_ids = image_ids[slice(start, end)]

        return [self._images[self._name][i].get_image() for i in image_ids]

    def get_masks_by_category(self, category_index_or_name: int | str, start: int = None, end: int = None) -> List[np.ndarray]:
        if isinstance(category_index_or_name, int):
            category_id = self._category_index_to_id[category_index_or_name]
        elif isinstance(category_index_or_name, str):
            category_id = self._category_name_to_id[category_index_or_name]
        else:
            raise ValueError(f"Unknown category type: {type(category_index_or_name)}")

        annotations = self._category_to_annotations[self._name][category_id]
        image_ids = [self._annotation_to_image[self._name][annotation_id] for annotation_id in annotations]
        image_ids = list(dict.fromkeys(image_ids))  # remove redundant, preserves same order as get_images_by_category
        image_ids = image_ids[slice(start, end)]

        all_masks = []
        for image_id in image_ids:
            annotation_ids = self._image_to_annotations[self._name][image_id]
            category_ids = [self._annotation_to_category[self._name][a_id] for a_id in annotation_ids]
            # Only keep image annotations if the category matches
            annotation_ids = [a_id for (a_id, c_id) in zip(annotation_ids, category_ids) if c_id == category_id]
            masks = []
            for instance_id, annotation_id in enumerate(annotation_ids):
                annot = self._annotations[self._name][annotation_id]
                masks.append(annot.get_mask().astype(int) * (instance_id + 1))
            all_masks.append(np.max(masks, axis=0))

        return all_masks

    def get_category_count(self):
        return len(self._category_index_to_id)

    def get_image_count(self):
        return len(self._image_index_to_id[self._name])

    def get_image_count_per_category(self, category_index_or_name: int | str):
        if isinstance(category_index_or_name, int):
            category_id = self._category_index_to_id[category_index_or_name]
        elif isinstance(category_index_or_name, str):
            category_id = self._category_name_to_id[category_index_or_name]
        else:
            raise ValueError(f"Unknown category type: {type(category_index_or_name)}")

        annotations = self._category_to_annotations[self._name][category_id]
        image_ids = [self._annotation_to_image[self._name][annotation_id] for annotation_id in annotations]
        image_ids = list(dict.fromkeys(image_ids))  # remove redundant
        return len(image_ids)

    def get_instance_count_per_category(self, category_index_or_name: int | str):
        if isinstance(category_index_or_name, int):
            category_id = self._category_index_to_id[category_index_or_name]
        elif isinstance(category_index_or_name, str):
            category_id = self._category_name_to_id[category_index_or_name]
        else:
            raise ValueError(f"Unknown category type: {type(category_index_or_name)}")

        cat_name = self._category_id_to_name[category_id]
        return self.instance_count[cat_name]

    def _set_metadata(self, categories_info):
        """
        Creates statistics about the dataset.
        """
        self._category_id_to_name = {d["id"]: d["name"] for d in categories_info}
        self._category_name_to_id = {d["name"]: d["id"] for d in categories_info}
        self.image_count = {d["name"]: d["image_count"] for d in categories_info}
        self.instance_count = {d["name"]: d["instance_count"] for d in categories_info}
        self.instances_per_image = [c["instance_count"] / c["image_count"] for c in categories_info]
        if len(self._whitelist) == 0:
            self._whitelist = list(self._category_name_to_id.keys())
        self._category_index_to_id = [self._category_name_to_id[cn] for cn in self._whitelist]

    def _set_data(self, images_info, annotations_info):
        """
        Reads through the dictionaries of the LVIS dataset and creates data containers
        """
        for name in images_info.keys():
            # iterate through annotations
            self._annotations[name] = {}
            self._annotation_to_image[name] = {}
            self._annotation_to_category[name] = {}
            self._image_to_annotations[name] = {}
            self._category_to_annotations[name] = {}
            for annotation_info in annotations_info[name]:
                if self._category_id_to_name[annotation_info["category_id"]] in self._whitelist:
                    image_id, annotation_id, category_id = annotation_info["image_id"], annotation_info["id"], annotation_info["category_id"]
                    a = LVISAnnotation(height=0, width=0, segments=annotation_info["segmentation"], category_id=category_id)
                    self._annotations[name][annotation_info["id"]] = a

                    # Create mapping between images <-> annotations <-> categories
                    self._annotation_to_image[name][annotation_id] = image_id
                    self._annotation_to_category[name][annotation_id] = category_id

                    if category_id not in self._category_to_annotations[name].keys():
                        self._category_to_annotations[name][category_id] = [annotation_id]
                    else:
                        self._category_to_annotations[name][category_id].append(annotation_id)

                    if image_id not in self._image_to_annotations[name].keys():
                        self._image_to_annotations[name][image_id] = [annotation_id]
                    else:
                        self._image_to_annotations[name][image_id].append(annotation_id)
            self._images[name] = {}
            self._image_index_to_id[name] = []
            for image_info in images_info[name]:
                image_id = image_info["id"]
                if image_id in self._image_to_annotations[name].keys():
                    base_name = os.path.basename(image_info["coco_url"])
                    output_filename = os.path.join(self._root_path, name, base_name)
                    source_filename = os.path.join(self._subset_files["source_folders"][name], base_name)
                    i = LVISImage(output_filename, image_info["height"], image_info["width"], source_url=image_info["coco_url"], source_filename=source_filename, copy_file=self._copy_files)
                    for annotation_id in self._image_to_annotations[name][image_id]:
                        annotation = self._annotations[name][annotation_id]
                        annotation.height = image_info["height"]
                        annotation.width = image_info["width"]
                    self._images[name][image_info["id"]] = i

                    # this is used to iterate through the images in order
                    self._image_index_to_id[name].append(image_id)

    def _download_images(self):
        """
        Downloads the COCO datasets
        """
        os.makedirs(os.path.join(self._root_path, "downloads"), exist_ok=True)
        for name, source in self._subset_files["downloads"].items():
            destination = self._subset_files["source_folders"][name]
            dst = os.path.join(self._root_path, "downloads", os.path.basename(source))  # temp
            self._download(source, dst)
            if os.path.splitext(dst)[1] == ".zip":
                self._unzip(dst, destination)
            else:
                destination = dst
            if not os.path.exists(destination):
                raise RuntimeError(f"Failed to produce {destination}")

    def _download_metadata(self):
        """
        Downloads the LVIS dataset metadata
        """
        for name, source in self._subset_files["sources"].items():
            destination = os.path.join(self._root_path, self._subset_files["files"][name])
            dst = os.path.join(self._root_path, os.path.basename(source))  # temp
            self._download(source, dst)
            if os.path.splitext(dst)[1] == ".zip":
                self._unzip(dst, destination)
            else:
                destination = dst
            if not os.path.exists(destination):
                raise RuntimeError(f"Failed to produce {destination}")

    def _get_metadata(self):
        """
        Extract the relevant metadata from the LVIS metadata.
        """
        images_info: Dict[str, Dict] = {}
        annotations_info: Dict[str, Dict] = {}
        categories_info = None
        for name, filename in self._subset_files["files"].items():
            with open(filename, "rt") as f:
                data = json.load(f)
                if categories_info is None:
                    categories_info = data["categories"]
                images_info[name] = data["images"]
                annotations_info[name] = data["annotations"]
        return categories_info, images_info, annotations_info

    def _save_metadata(self, filename):
        """
        Saves the metadata in a cache file

        Args:
            filename: The filename of the metadata
        """
        with open(filename, 'wb') as f:
            data = {"category_id_to_name": self._category_id_to_name,
                    "category_name_to_id": self._category_name_to_id,
                    "instance_count": self.instance_count,
                    "image_count": self.image_count,
                    "instances_per_image": self.instances_per_image,
                    "category_index_to_id": self._category_index_to_id
                    }
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def _load_metadata(self, filename):
        """
        Load the metadata from a cache file

        Args:
            filename: The filename of the metadata
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self._category_id_to_name = data["category_id_to_name"]
            self._category_name_to_id = data["category_name_to_id"]
            self.instance_count = data["instance_count"]
            self.image_count = data["image_count"]
            self.instances_per_image = data["instances_per_image"]
            self._category_index_to_id = data["category_index_to_id"]

    def _save_data(self, filename):
        """
        Saves the data in a cache file

        Args:
            filename: The filename of the data
        """
        with open(filename, 'wb') as f:
            data = {"image_index_to_id": self._image_index_to_id,
                    "images": self._images,
                    "annotations": self._annotations,
                    "annotation_to_image": self._annotation_to_image,
                    "annotation_to_category": self._annotation_to_category,
                    "image_to_annotations": self._image_to_annotations,
                    "category_to_annotations": self._category_to_annotations}

            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def _load_data(self, filename):
        """
        Load the data from a cache file

        Args:
            filename: The filename of the data
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self._image_index_to_id = data["image_index_to_id"]
            self._images = data["images"]
            self._annotations = data["annotations"]
            self._annotation_to_image = data["annotation_to_image"]
            self._annotation_to_category = data["annotation_to_category"]
            self._image_to_annotations = data["image_to_annotations"]
            self._category_to_annotations = data["category_to_annotations"]

    def _check_cache(self):
        """
        Check if a list has been changed to determine when to invalidate caches. Also checks if
        the cached files exist.
        """
        valid = True

        # Check if whitelist is the same
        if os.path.isfile(self._cache_check_file):
            identical = True
            with open(self._cache_check_file, "rb") as f:
                data = pickle.load(f)
                identical = identical and data["whitelist"] == self._whitelist
        else:
            identical = False

        with open(self._cache_check_file, "wb") as f:
            data = {"whitelist": self._whitelist}
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        valid = valid and identical

        # Check if cached files exist
        valid = valid and os.path.isfile(self._cached_metadata) and os.path.isfile(self._cached_data)

        if not valid:
            if os.path.isfile(self._cached_metadata):
                os.remove(self._cached_metadata)
            if os.path.isfile(self._cached_data):
                os.remove(self._cached_data)

        return valid


def test_index_iter():
    # Use default index iterator (PyTorch style)
    dataset = LVISDataset(whitelist=['teacup', 'doughnut'], download_full_dataset=True, copy_files=False)

    for image_index, (image, masks) in enumerate(dataset):
        # Generate and save overlays
        for category_id, mask in masks.items():
            overlay = color_overlay(image, mask)
            cat = dataset._category_id_to_name[category_id]
            output_folder = os.path.join(dataset.get_root_path(), "overlays", cat)
            orig_filename = os.path.splitext(os.path.basename(dataset.get_image_filename(image_index)))[0]
            os.makedirs(output_folder, exist_ok=True)
            cv2.imwrite(os.path.join(output_folder, f"{orig_filename}_{cat}.jpg"), overlay)


def test_category_iter():
    # Use category iterator
    dataset = LVISDataset(whitelist=['teacup', 'doughnut'], iterator_type=CategoryIter, download_full_dataset=True, copy_files=False)

    for category_index, (images, masks) in enumerate(dataset):
        for image_index, (image, mask) in enumerate(zip(images, masks)):
            # Generate and save overlays
            overlay = color_overlay(image, mask)
            cat = dataset._category_id_to_name[dataset._category_index_to_id[category_index]]
            output_folder = os.path.join(dataset.get_root_path(), "overlays", cat)
            orig_filename = os.path.splitext(os.path.basename(dataset.get_image_filename_in_category(category_index, image_index)))[0]
            os.makedirs(output_folder, exist_ok=True)
            cv2.imwrite(os.path.join(output_folder, f"{orig_filename}_{cat}.jpg"), overlay)


if __name__ == "__main__":
    test_index_iter()
    test_category_iter()
