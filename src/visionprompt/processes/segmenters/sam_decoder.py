"""SAM decoder."""
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from visionprompt.models.per_segment_anything import SamPredictor
from visionprompt.processes.segmenters.segmenter_base import Segmenter
from visionprompt.types import Image, Masks, Points, Priors
from visionprompt.types.similarities import Similarities


class SamDecoder(Segmenter):
    """This Segmenter uses SAM to create masks based on points.

    Examples:
        >>> from visionprompt.models.per_segment_anything import SamPredictor
        >>> from visionprompt.processes.segmenters import SamDecoder
        >>> from visionprompt.types import Image, Priors, Similarities
        >>>
        >>> sam_predictor = SamPredictor(...)
        >>> segmenter = SamDecoder(sam_predictor)
        >>> masks, points = segmenter(
        ...     images=[Image()],
        ...     priors=[Priors()],
        ...     similarities=[Similarities()],
        ... )
    """

    def __init__(
        self,
        sam_predictor: SamPredictor,
        apply_mask_refinement: bool = False,
        target_guided_attention: bool = False,
        mask_similarity_threshold: float = 0.45,
        skip_points_in_existing_masks: bool = True,
    ) -> None:
        """This Segmenter uses SAM to create masks based on points.

        The resulting masks are then filtered based on the average similarity of that mask.

        Args:
            sam_predictor: SamPredictor the SAM predictor
            apply_mask_refinement: bool whether to apply mask refinement
            target_guided_attention: bool whether to use target guided attention, TODO: not implemented yet
            mask_similarity_threshold: float the threshold for the average similarity of a mask
            skip_points_in_existing_masks: bool whether to skip points that fall within already generated masks
              for the same class
        """
        super().__init__()
        self.predictor = sam_predictor
        self.apply_mask_refinement = apply_mask_refinement
        self.target_guided_attention = target_guided_attention
        self.mask_similarity_threshold = mask_similarity_threshold
        self.skip_points_in_existing_masks = skip_points_in_existing_masks
        # Store similarity scores for analysis
        self.similarity_scores = defaultdict(list)
        self.image_counter = 0

    def __call__(
        self,
        images: list[Image],
        priors: list[Priors] | None = None,
        similarities: list[Similarities] | None = None,
    ) -> tuple[list[Masks], list[Points]]:
        """Create masks from priors using SAM.

        Args:
            images: List of target images.
            priors: A list of priors, one for each target image.
            similarities: A list of similarities, one for each target image.

        Returns:
            A tuple of a list of masks, one for each class in each target image,
            and a list of points, one for each class in each target image.
        """
        if similarities is None:
            similarities = []
        masks_per_image: list[Masks] = []
        points_per_image: list[Points] = []

        for i, (image, priors_per_image, similarities_per_image) in enumerate(
            iterable=zip_longest(images, priors, similarities, fillvalue=None),
        ):
            if priors_per_image is None:
                continue
            masks, points_used = self._predict_by_individual_point(
                image,
                priors_per_image.points,
                similarities_per_image,
                image_id=self.image_counter + i,
            )
            masks_per_image.append(masks)
            points_per_image.append(points_used)

        self.image_counter += len(images)

        return masks_per_image, points_per_image

    def _predict_by_individual_point(  # noqa: C901
        self,
        image: Image,
        map_points: Points,
        similarities: Similarities | None = None,
        image_id: int = 0,
    ) -> tuple[Masks, Points]:
        """Predict masks from a list of points.

        Args:
            image: The image to predict masks from.
            map_points: The points to predict masks from.
            similarities: Optional similarities.
            image_id: ID to identify which image is being processed.

        Returns:
            A tuple of generated masks and actual points used.
        """
        all_masks = Masks()
        all_used_points = Points()

        self.predictor.set_image(image.data)
        for class_id, points_per_map in map_points.data.items():  # noqa: PLR1702
            # iterate over each point list of each similarity map
            for points_in_current_map in points_per_map:
                if len(points_in_current_map) == 0:
                    # no points for this class, add empty "used_points" for this class
                    all_used_points.add(torch.tensor(np.array([])), class_id)
                    continue

                points_used = []
                # point list is of shape (n, 4), each item is (x, y, score, label),
                # label is 1 for foreground and 0 for background
                background_points = points_in_current_map[points_in_current_map[:, 3] == 0].cpu().numpy()
                foreground_points = points_in_current_map[points_in_current_map[:, 3] == 1].cpu().numpy()

                # predict masks
                for _i, (x, y, score, label) in enumerate(foreground_points):
                    # filter out points that lie inside a previously found mask
                    if self.skip_points_in_existing_masks:
                        is_covered = False
                        for mask in all_masks.get(class_id):
                            if int(y) < 0 or int(x) < 0:
                                continue

                            # move edge points inside mask
                            if int(y) == mask.shape[0]:
                                y -= 1
                            if int(x) == mask.shape[1]:
                                x -= 1

                            if mask[int(y), int(x)]:
                                is_covered = True
                                break
                        if is_covered:
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
                        [label] + [0] * len(background_points),
                        dtype=np.float32,
                    )

                    masks, mask_scores, low_res_logits, *_ = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=False,
                        # TODO(Daankrol): target guided attention and target-semantic prompting
                    )

                    if not self.apply_mask_refinement:
                        final_mask = masks[np.argmax(mask_scores)]
                    else:
                        final_mask, _, _ = self.refine_masks(
                            low_res_logits,
                            point_coords,
                            point_labels,
                        )

                    # Filter the mask based on average similarity
                    if similarities is not None and not self._filter_mask(
                        final_mask,
                        similarities.data[class_id],
                        image_id,
                        class_id,
                    ):
                        continue

                    all_masks.add(final_mask, class_id)
                    points_used.append([x, y, score, label])

                points_used.extend(background_points)
                # save the points used for the current
                all_used_points.add(torch.tensor(np.array(points_used)), class_id)

        return all_masks, all_used_points

    def _filter_mask(
        self,
        mask: np.ndarray,
        similarities: torch.Tensor,
        image_id: int = 0,
        class_id: int = 0,
    ) -> bool:
        """Filter the mask based on the average similarity with the reference masks.

        Args:
            mask: The mask to filter
            similarities: The similarity tensor
            image_id: ID to identify which image this mask belongs to
            class_id: The class ID of this mask

        Returns:
            True if the mask should be kept, False otherwise
        """
        # For odd sized images, SAM longest-side resize operation can result in different shape than sim map
        if mask.shape != similarities.shape[1:]:
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            mask_tensor = (
                F.interpolate(
                    mask_tensor,
                    size=(similarities.shape[1], similarities.shape[2]),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .bool()
                .cpu()
                .numpy()
            )
            mask = mask_tensor

        mask_similarity = similarities[0, mask]
        average_similarity = mask_similarity.mean()

        # Store the score for analysis
        key = f"image_{image_id}_class_{class_id}"
        self.similarity_scores[key].append(float(average_similarity))

        return average_similarity > self.mask_similarity_threshold

    def refine_masks(
        self,
        logits: torch.Tensor,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Refines the predicted mask by reapplying the decoder with step wise increase of input information.

        Args:
            logits: logits from the decoder
            point_coords: point coordinates (x, y)
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

    def reset_similarity_scores(self) -> None:
        """Reset the stored similarity scores."""
        self.similarity_scores = defaultdict(list)
        self.image_counter = 0

    def plot_similarity_distributions(self, save_path: str | None = "similarity_distributions") -> None:
        """Plot the distribution of similarity scores for each image.

        Args:
            save_path: Optional path to save the plot to
            combined_plot: If True, creates a single plot with all scores grouped by class
        """
        if not self.similarity_scores:
            return

        class_scores_combined = defaultdict(list)
        for key, scores in self.similarity_scores.items():
            _, class_id = key.split("_class_")
            class_scores_combined[class_id].extend(scores)

        fig, ax = plt.subplots(figsize=(14, 8))

        for class_id, scores in sorted(
            class_scores_combined.items(),
            key=lambda x: x[0],
        ):
            ax.hist(scores, alpha=0.7, bins=20, label=f"Class {class_id}")

        ax.axvline(
            x=self.mask_similarity_threshold,
            color="r",
            linestyle="--",
            label=f"Threshold ({self.mask_similarity_threshold})",
        )
        ax.set_title("Combined Similarity Score Distribution - All Images")
        ax.set_xlabel("Average Similarity Score")
        ax.set_ylabel("Frequency")
        ax.legend()
        path = Path(save_path)
        base, ext = (path.stem, path.suffix) if path.suffix else (str(path), ".png")
        combined_save_path = f"{base}_combined{ext}"
        plt.savefig(combined_save_path)
        plt.tight_layout()
        plt.show()
        plt.close(fig)
