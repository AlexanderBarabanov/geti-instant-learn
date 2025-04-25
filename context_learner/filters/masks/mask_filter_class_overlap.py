# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from itertools import product
from typing import List

import torch

from context_learner.filters.masks.mask_filter_base import MaskFilter
from context_learner.types import Masks, Points


class ClassOverlapMaskFilter(MaskFilter):
    def __call__(
        self,
        masks_per_image: List[Masks],
        used_points_per_image: List[Points],
        threshold_iou: float = 0.8,
    ) -> List[Masks]:
        """
        Inspect overlapping areas between different label masks.
        :param masks_per_image:    Predicted mask for each image and all labels
        :param used_points_per_image:        Used points for each image and all labels
        :param threshold_iou:    Threshold for IOU between the masks
        :return:
        """
        for image_masks, image_used_points in zip(
            masks_per_image, used_points_per_image
        ):
            for (label, masks), (other_label, other_masks) in product(
                image_masks.data.items(), image_masks.data.items()
            ):
                if other_label <= label:
                    continue

                overlapped_label = []
                overlapped_other_label = []
                for (im, mask), (jm, other_mask) in product(
                    enumerate(masks), enumerate(other_masks)
                ):
                    _mask_iou, _intersection = self._calculate_mask_iou(
                        mask, other_mask
                    )
                    if _mask_iou > threshold_iou:
                        if (
                            image_used_points.data[label][im][2]
                            > image_used_points.data[other_label][jm][2]
                        ):
                            overlapped_other_label.append(jm)
                        else:
                            overlapped_label.append(im)
                    elif _mask_iou > 0:
                        # refine the slightly overlapping region
                        overlapped_coords = torch.where(_intersection)
                        if (
                            image_used_points.data[label][im][2]
                            > image_used_points.data[other_label][jm][2]
                        ):
                            other_mask[overlapped_coords] = 0.0
                        else:
                            mask[overlapped_coords] = 0.0

                for im in sorted(set(overlapped_label), reverse=True):
                    # Create new tensor excluding the mask at index im
                    new_masks = torch.cat([masks[:im], masks[im + 1 :]], dim=0)
                    image_masks.data[label] = new_masks
                    used_points_per_image.data[label].pop(im)

                for jm in sorted(set(overlapped_other_label), reverse=True):
                    # Create new tensor excluding the mask at index jm
                    new_other_masks = torch.cat(
                        [other_masks[:jm], other_masks[jm + 1 :]], dim=0
                    )
                    image_masks.data[other_label] = new_other_masks
                    used_points_per_image.data[other_label].pop(jm)

        return masks_per_image
