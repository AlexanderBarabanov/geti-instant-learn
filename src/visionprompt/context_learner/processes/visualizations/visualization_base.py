# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import torch

from visionprompt.context_learner.processes import Process
from visionprompt.context_learner.types import Masks, Points, Priors


class Visualization(Process):
    """This is the base class for all visualization processes.

    It provides a way to visualize the data.
    """

    @staticmethod
    def masks_from_priors(priors: list[Priors]) -> list[Masks]:
        """Converts a list of shape 1HW to a List of Masks.

        Note: The first channel of arrays contains instance ids in range [0..max()].

        Args:
            priors: The list of priors to convert

        Returns:
            The list of masks
        """
        return [m.masks for m in priors]

    @staticmethod
    def points_from_priors(priors: list[Priors]) -> list[Points]:
        """Extracts points from priors.

        Args:
            priors: The list of priors to extract points from

        Returns:
            The list of points
        """
        return [m.points for m in priors]

    @staticmethod
    def arrays_to_masks(arrays: list[np.ndarray], class_id: int = 0) -> list[Masks]:
        """Converts a list of shape 1HW to a List of Masks.

        Note: The first channel of arrays contains instance ids in range [0..max()].

        Args:
            arrays: The list of arrays to convert
            class_id: The class id to use for the masks

        Returns:
            The list of masks
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

    def __call__(self) -> None:
        """Call visualization process."""
