import os
from functools import partial

import pandas as pd

from datasets.dataset_iterators import BatchedCategoryIter
from datasets.lvis.lvis_dataset import LVISDataset
from datasets.perseg.perseg_dataset import PerSegDataset


def load_dataset(dataset_name: str, whitelist=None):
    if dataset_name == "PerSeg":
        return PerSegDataset(iterator_type=BatchedCategoryIter, iterator_kwargs={"batch_size": 5},)
    elif dataset_name == "lvis":
        whitelist = (
            whitelist
            if whitelist is not None
            else ("cupcake", "sheep", "pastry", "doughnut")
        )
        return LVISDataset(
            whitelist=whitelist,
            iterator_type=BatchedCategoryIter,
            iterator_kwargs={"batch_size": 5},
        )
    elif dataset_name == "lvis_validation":
        whitelist = (
            whitelist
            if whitelist is not None
            else ("cupcake", "sheep", "pastry", "doughnut")
        )
        return LVISDataset(
            whitelist=whitelist,
            iterator_type=BatchedCategoryIter,
            iterator_kwargs={"batch_size": 5},
            name="validation",
        )
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}")
