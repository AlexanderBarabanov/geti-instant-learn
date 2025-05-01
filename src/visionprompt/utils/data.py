# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from visionprompt.datasets.dataset_base import Dataset
from visionprompt.datasets.dataset_iterators import BatchedCategoryIter
from visionprompt.datasets.lvis.lvis_dataset import LVISDataset
from visionprompt.datasets.perseg.perseg_dataset import PerSegDataset


def load_dataset(dataset_name: str, whitelist: list[str] | None = None) -> Dataset:
    """Load a dataset.

    Args:
        dataset_name: Name of the dataset
        whitelist: Whitelist of categories

    Returns:
        Dataset

    Raises:
        ValueError: If the dataset name is not recognized
    """
    # add logging that we are loading the dataset
    logging.info(f"Loading dataset: {dataset_name}")
    if dataset_name == "PerSeg":
        return PerSegDataset(
            whitelist=whitelist,
            iterator_type=BatchedCategoryIter,
            iterator_kwargs={"batch_size": 5},
        )
    if dataset_name == "lvis":
        whitelist = whitelist if whitelist is not None else ("cupcake", "sheep", "pastry", "doughnut")
        return LVISDataset(
            whitelist=whitelist,
            iterator_type=BatchedCategoryIter,
            iterator_kwargs={"batch_size": 5},
        )
    if dataset_name == "lvis_validation":
        whitelist = whitelist if whitelist is not None else ("cupcake", "sheep", "pastry", "doughnut")
        return LVISDataset(
            whitelist=whitelist,
            iterator_type=BatchedCategoryIter,
            iterator_kwargs={"batch_size": 5},
            name="validation",
        )
    msg = f"Unknown dataset name {dataset_name}"
    raise ValueError(msg)
