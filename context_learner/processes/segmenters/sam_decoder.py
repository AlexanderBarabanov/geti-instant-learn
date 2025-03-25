from typing import List

import numpy as np

from context_learner.processes.segmenters.segmenter_base import Segmenter
from context_learner.types.image import Image
from context_learner.types.masks import Masks
from context_learner.types.priors import Priors
from context_learner.types.points import Points
from context_learner.types.state import State

from PersonalizeSAM.per_segment_anything.predictor import SamPredictor
import torch


class SamDecoder(Segmenter):
    def __init__(
        self,
        state: State,
        sam_predictor: SamPredictor,
        apply_mask_refinement: bool = False,
        target_guided_attention: bool = False,
    ):
        super().__init__(state)
        self.predictor = sam_predictor
        self.apply_mask_refinement = apply_mask_refinement
        self.target_guided_attention = target_guided_attention

    def __call__(
        self, images: List[Image], priors: List[Priors]
    ) -> tuple[List[Masks], List[Points]]:
        """Create masks from priors using SAM.

        Args:
            images: List of target images.
            priors: A list of priors, one for each target image.

        Returns:
            A tuple of a list of masks, one for each class in each target image,
            and a list of points, one for each class in each target image.
        """
        masks_per_image: List[Masks] = []
        points_per_image: List[Points] = []

        for image, priors_per_image in zip(images, priors):
            masks, points_used = self._predict_by_individual_point(
                image, priors_per_image.points
            )
            masks_per_image.append(masks)
            points_per_image.append(points_used)
        return masks_per_image, points_per_image

    def _predict_by_individual_point(
        self, image: Image, points: Points
    ) -> tuple[Masks, Points]:
        """
        Predict masks from a list of points.

        Args:
            image: The image to predict masks from.
            points: The points to predict masks from.

        Returns:
            A tuple of generated masks and actual points used.
        """
        all_masks = Masks()
        all_used_points = Points()

        self.predictor.set_image(image.data)
        for class_id, points_per_map in points.data.items():
            # iterate over each point list of each similarity map
            for points in points_per_map:
                if len(points) == 0:
                    continue

                points_used = []
                # point list is of shape (n, 4), each item is (x, y, score, label), label is 1 for foreground and 0 for background
                background_points = points[points[:, 3] == 0].cpu().numpy()
                foreground_points = points[points[:, 3] == 1].cpu().numpy()

                # predict masks
                for i, (x, y, score, label) in enumerate(foreground_points):
                    # remove points with very low confidence
                    if score in [-1.0, 0.0]:
                        continue
                    # filter out points that lie inside a previously found mask
                    is_done = False
                    for mask in all_masks.get(class_id):
                        if mask[int(y), int(x)]:
                            is_done = True
                            break
                    if is_done:
                        continue

                    point_coords = np.concatenate(
                        (
                            np.array([[x, y]]),
                            background_points[:, :2],
                        ),
                        axis=0,
                        dtype=np.float32,
                    )
                    point_labels = np.array(
                        [label] + [0] * len(background_points), dtype=np.float32
                    )

                    # Use predict torch here instead of predict
                    masks, scores, low_res_logits, high_res_logits = (
                        self.predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            multimask_output=False,
                            # TODO target guided attention and target-semantic prompting
                        )
                    )

                    if not self.apply_mask_refinement:
                        final_mask = masks[np.argmax(scores)]
                    else:
                        final_mask = self.refine_masks(
                            low_res_logits, point_coords, point_labels
                        )

                    all_masks.add(final_mask, class_id)
                    points_used.append([x, y, score, label])

                points_used.extend(background_points)
                # save the points used for the current
                all_used_points.add(torch.tensor(np.array(points_used)), class_id)

        return all_masks, all_used_points

    def refine_masks(
        self, logits: torch.Tensor, point_coords: np.ndarray, point_labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Refines the predicted mask by reapplying the decoder with step wise increase of input information.

        Args:
            logits: logits from the decoder
            point_coordinates: point coordinates (x, y)
            point_labels: point labels (1 for foreground, 0 for background)

        Returns:
            final_mask: refined mask
            masks: all masks
            final_score: score of the refined mask
        """
        best_idx = 0
        # Cascaded Post-refinement-1
        masks, scores, logits, *_ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=logits[best_idx : best_idx + 1, :, :],
            multimask_output=True,
        )
        best_idx = np.argmax(scores)

        # Cascaded Post-refinement-2
        y, x = np.nonzero(masks[best_idx])
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        input_box = np.array([x_min, y_min, x_max, y_max])
        masks, scores, logits, *_ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box[None, :],
            mask_input=logits[best_idx : best_idx + 1, :, :],
            multimask_output=True,
        )
        best_idx = np.argmax(scores)
        final_mask = masks[best_idx]
        final_score = scores[best_idx]
        return final_mask, masks, final_score
