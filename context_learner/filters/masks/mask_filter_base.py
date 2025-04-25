# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch

from context_learner.filters import Filter
from context_learner.types import Masks


class MaskFilter(Filter):
    def __call__(self, masks: List[Masks]) -> List[Masks]:
        return masks

    def _calculate_mask_iou(
        self,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
    ) -> tuple[float, torch.Tensor | None]:
        """
        Calculate the IoU between two masks.
        :param mask1:    First mask
        :param mask2:    Second mask
        :return:    IoU between the two masks and the intersection
        """
        assert mask1.dim() == 2
        assert mask2.dim() == 2
        # Avoid division by zero
        union = (mask1 | mask2).sum().item()
        if union == 0:
            return 0.0, None
        intersection = mask1 & mask2
        return intersection.sum().item() / union, intersection
