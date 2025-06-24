"""Export visualization to file."""
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import cv2
import numpy as np

from visionprompt.processes.visualizations.visualization_base import (
    Visualization,
)
from visionprompt.types import Image, Masks, Points
from visionprompt.utils import get_colors


class ExportMaskVisualization(Visualization):
    """The class exports the images for visualization.

    Examples:
        >>> import os
        >>> import numpy as np
        >>> from visionprompt.processes.visualizations import ExportMaskVisualization
        >>> from visionprompt.types import Image, Masks, Points
        >>>
        >>> visualizer = ExportMaskVisualization(output_folder="visualizations")
        >>> sample_image = Image(np.zeros((10, 10, 3), dtype=np.uint8))
        >>> visualizer(
        ...     images=[sample_image],
        ...     masks=[Masks()],
        ...     names=["test.png"],
        ...     points=[Points()],
        ... )
        >>> # Check if the visualization was saved
        >>> os.path.exists("visualizations/test.png")
        True
        >>> os.remove("visualizations/test.png")
    """

    def __init__(self, output_folder: str) -> None:
        super().__init__()
        self.output_folder = output_folder

    @staticmethod
    def create_overlay(
        image: np.ndarray,
        masks: np.ndarray,
        points: list[Points] | None = None,
        scores: list[float] | None = None,
        types: list[int] | None = None,
    ) -> np.ndarray:
        """Save a visualization of the segmentation mask overlaid on the image.

        Args:
            image: RGB image as numpy array
            masks: Segmentation mask object with containing instance masks
            points: Optional points to visualize
            scores: Optional confidence scores for the points
            types: The type of point (usually, 0 for background, 1 for foreground)
        """
        image_vis = image.copy()

        if masks is not None:
            # Get unique colors for each instance mask
            mask_colors = get_colors(len(masks))

            # Draw each instance mask with a different color
            for i, instance in enumerate(masks):
                masked_img = np.where(instance[..., None], mask_colors[i], image_vis)
                image_vis = cv2.addWeighted(image_vis, 0.2, masked_img, 0.8, 0)

            # Draw points and confidence scores if provided
            if points is not None and scores is not None and types is not None:
                for i, point in enumerate(points):
                    # Draw star marker
                    x, y = int(point[0]), int(point[1])
                    size = int(image.shape[0] / 50)  # Scale marker size with image
                    cv2.drawMarker(
                        image_vis,
                        (x, y),
                        (255, 255, 255),
                        cv2.MARKER_STAR if types[i] == 1.0 else cv2.MARKER_SQUARE,
                        size,
                    )

                    # Add confidence score text
                    confidence = float(scores[i])
                    cv2.putText(
                        image_vis,
                        f"{confidence:.2f}",
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        image.shape[0] / 1500,
                        (255, 255, 255),
                        1,
                    )
        return image_vis

    def __call__(
        self,
        images: list[Image] | None = None,
        masks: list[Masks] | None = None,
        names: list[str] | None = None,
        points: list[Points] | None = None,
    ) -> None:
        """This method exports the visualization images.

        Args:
            images: List of input images
            masks: List of input masks
            names: List of filenames
            points: List of points to visualize
        """
        # Generate overlay
        if names is None:
            names = []
        if masks is None:
            masks = []
        if images is None:
            images = []
        for i in range(len(images)):
            # Get correct datas
            masks_per_class = masks[i]

            image_np = images[i].to_numpy()
            name = names[i]

            output_filename = Path(self.output_folder) / name
            Path.mkdir(Path(output_filename, parents=True).parent, exist_ok=True, parents=True)

            if len(masks_per_class.class_ids()) > 1:
                msg = "Multiple class masks not supported yet."
                raise RuntimeError(msg)
            image_vis = image_np

            for class_id in masks_per_class.class_ids():
                mask_np = masks_per_class.to_numpy(class_id)
                if points is not None and i < len(points) and points[i] is not None:
                    current_points = points[i].data[class_id][0]
                    yxs, scores, types = (
                        current_points.cpu().numpy()[:, :2],
                        current_points.cpu().numpy()[:, 2],
                        current_points.cpu().numpy()[:, 3],
                    )
                else:
                    yxs = scores = types = None
                image_vis = self.create_overlay(
                    image=image_np,
                    masks=mask_np,
                    points=yxs,
                    scores=scores,
                    types=types,
                )

            # Save visualization
            cv2.imwrite(output_filename, cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))
