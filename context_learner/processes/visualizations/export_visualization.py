import os
import cv2
from typing import List, Tuple
import numpy as np

from context_learner.processes.visualizations.visualization_base import Visualization
from context_learner.types import Points
from context_learner.types.image import Image
from context_learner.types.masks import Masks
from context_learner.types.priors import Priors
from context_learner.types.state import State
from utils.utils import get_colors


class ExportMaskVisualization(Visualization):
    def __init__(self, state: State, output_folder: str):
        super().__init__(state)
        self.output_folder = output_folder

    @staticmethod
    def create_overlay(image: np.ndarray, masks: np.ndarray, points=None, scores=None, types=None) -> np.ndarray:
        """
        Save a visualization of the segmentation mask overlaid on the image.

        Args:
            image: RGB image as numpy array
            masks: Segmentation mask object with containing instance masks
            points: Optional points to visualize
            scores: Optional confidence scores for the points\
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
                    cv2.drawMarker(image_vis, (x, y), (255, 255, 255), cv2.MARKER_STAR if types[i] == 1. else cv2.MARKER_SQUARE , size)

                    # Add confidence score text
                    confidence = float(scores[i])
                    cv2.putText(image_vis,f"{confidence:.2f}",(x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, image.shape[0] / 1500, (255, 255, 255), 1)
        return image_vis

    def __call__(self, images: List[Image], masks: List[Masks], names: List[str], points: List[Points] = None):
        # Use points from the state if they are not passed
        if points is None:
            points = [None] * len(images)
            for i in range(len(points)):
                if len(self._state.used_points) > 0 and len(self._state.used_points[i].data.keys()) > 0:
                    points[i] = self._state.used_points[i]

        # Generate overlay
        for i in range(len(images)):
            # Get correct datas
            masks_per_class = masks[i]

            image_np = images[i].to_numpy()
            name = names[i]

            output_filename = os.path.join(self.output_folder, name)
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)

            if len(masks_per_class.class_ids()) > 1:
                raise RuntimeError("Multiple class masks not supported yet.")
            image_vis = image_np

            for class_id in masks_per_class.class_ids():
                mask_np = masks_per_class.to_numpy(class_id)
                if points[i] is not None:
                    current_points = points[i].data[class_id][0]
                    yxs, scores, types = current_points.cpu().numpy()[:, :2], current_points.cpu().numpy()[:, 2], current_points.cpu().numpy()[:, 3]
                else:
                    yxs = scores = types = None
                image_vis = self.create_overlay(image=image_np, masks=mask_np, points=yxs, scores=scores, types=types)

            # Save visualization
            cv2.imwrite(output_filename, cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))




