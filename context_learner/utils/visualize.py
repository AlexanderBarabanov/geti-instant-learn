import cv2
import numpy as np
import torch
from typing import List, Dict

from context_learner.types import Image, Masks, Points

# Define some colors for visualization
COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (0, 128, 128),
    (128, 0, 128),
]


def overlay_masks_and_points(
    image: Image, masks: Masks, points: Points, alpha: float = 0.3
) -> np.ndarray:
    """
    Overlays masks onto an image using fixed blending weights (0.7 image, 0.3 mask),
    similar to the provided main.py snippet. Creates a combined mask first.
    Draws points on top with distinct colors per class.

    Args:
        image: The input Image object.
        masks: A Masks object containing masks per class.
        points: A Points object containing points per class.
        alpha: Transparency level (currently ignored for mask blending, uses 0.3).

    Returns:
        The image (as a numpy array) with mask overlays and points.
    """
    img_np = image.data.copy()  # Work on a copy

    # Ensure image is 3-channel BGR
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 4:  # RGBA
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)

    # Create a combined mask from all classes
    combined_mask_np = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
    for class_id, mask_tensor in masks.data.items():
        if mask_tensor.numel() == 0:
            continue
        class_combined_mask_tensor = torch.any(mask_tensor, dim=0)
        mask_np = class_combined_mask_tensor.cpu().numpy().astype(np.uint8)

        if mask_np.ndim == 3:
            mask_np = mask_np.squeeze()

        if mask_np.shape != (img_np.shape[0], img_np.shape[1]):
            mask_np = cv2.resize(
                mask_np,
                (img_np.shape[1], img_np.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Combine with the overall mask using logical OR
        combined_mask_np = np.logical_or(combined_mask_np, mask_np).astype(np.uint8)

    # Create a 3-channel BGR image from the final combined mask (white where mask is > 0)
    final_mask_img = np.zeros_like(img_np, dtype=np.uint8)
    final_mask_img[combined_mask_np > 0] = (255, 255, 255)

    # Blend the original image with the final 3-channel mask image using fixed weights
    overlay_image = cv2.addWeighted(img_np, 0.7, final_mask_img, 0.3, 0)

    # Draw points on top of the blended image
    color_idx = 0
    for class_id, points_per_map_tensor in points.data.items():
        color = COLORS[color_idx % len(COLORS)]
        color_idx += 1
        for points_tensor in points_per_map_tensor:
            for point in points_tensor:
                # Use .item() to get scalar value from tensor
                x, y = int(point[0].item()), int(point[1].item())
                cv2.circle(
                    overlay_image, (x, y), radius=5, color=color, thickness=-1
                )  # Filled circle
                # Optional: Add a border for better visibility
                cv2.circle(
                    overlay_image, (x, y), radius=5, color=(0, 0, 0), thickness=1
                )

    return overlay_image
