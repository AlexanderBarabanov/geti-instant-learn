import cv2
import numpy as np
from typing import List

from context_learner.processes.mask_processors.mask_processor_base import (
    MaskProcessor,
)
from context_learner.types import Annotations, Masks


class MasksToPolygons(MaskProcessor):
    def __call__(self, masks: List[Masks]) -> List[Annotations]:
        """
        Convert a list of masks to a list of annotations (polygons).
        """
        annotations_list = []

        for mask_obj in masks:
            annotation = Annotations()

            for class_id in mask_obj._data:
                instance_masks = mask_obj.data[class_id].cpu().numpy()

                for instance_idx in range(len(instance_masks)):
                    mask = instance_masks[instance_idx].astype(np.uint8) * 255
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    for contour in contours:
                        # Simplify the contour to reduce number of points
                        epsilon = 0.005 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)

                        # Convert to list of [x, y] coordinates
                        polygon = approx.reshape(-1, 2).tolist()

                        # Only add polygons with at least 3 points
                        if len(polygon) >= 3:
                            annotation.add_polygon(polygon, class_id)

            annotations_list.append(annotation)

        return annotations_list
