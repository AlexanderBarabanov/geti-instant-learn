# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch
import numpy as np
from context_learner.processes import Process
from context_learner.types import Points, Priors, Masks, State


class Visualization(Process):
    def __init__(self, state: State):
        super().__init__(state)

    @staticmethod
    def masks_from_priors(priors: List[Priors]) -> List[Masks]:
        return [m.masks for m in priors]

    @staticmethod
    def points_from_priors(priors: List[Priors]) -> List[Points]:
        return [m.points for m in priors]

    @staticmethod
    def arrays_to_masks(arrays: List[np.ndarray], class_id=0) -> List[Masks]:
        """
        Converts a list of shape 1HW to a List of Masks
        Note: The first channel of arrays contains instance ids in range [0..max()]
        """
        masks = []
        for instance_masks in arrays:
            # 1HW -> HWN
            n_values = np.max(instance_masks) + 1
            one_hot_masks = np.eye(n_values, dtype=bool)[instance_masks]
            # HWN -> NHW tensor
            one_hot_tensor = torch.from_numpy(np.moveaxis(one_hot_masks, 2, 0))
            # Remove background mask and create Mask instance
            m = Masks({class_id: one_hot_tensor[1:]})
            masks.append(m)
        return masks

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
