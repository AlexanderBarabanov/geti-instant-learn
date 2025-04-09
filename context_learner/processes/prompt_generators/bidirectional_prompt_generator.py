from typing import List, Tuple

import torch
import numpy as np

from context_learner.processes.prompt_generators.prompt_generator_base import (
    PromptGenerator,
)
from context_learner.types.priors import Priors
from context_learner.types.features import Features
from context_learner.types.masks import Masks
from context_learner.types.state import State
from scipy.optimize import linear_sum_assignment


class BidirectionalPromptGenerator(PromptGenerator):
    def __init__(self, state: State):
        super().__init__(state)

    def __call__(
        self,
        reference_features: List[Features],
        reference_masks: List[Masks],
        target_features_list: List[Features],
    ) -> List[Priors]:
        """
        This generates prompt candidates (or priors) based on the similarities between the reference and target images.
        It uses bidirectional matching to create prompts for the segmenter.
        This Prompt Generator computes the similarity map internally.

        Args:
            reference_features: List[Features] List of reference features, one per reference image instance
            reference_masks: List[Masks] List of reference masks, one per reference image instance
            target_features_list: List[Features] List of target features, one per target image instance

        Returns:
            List[Priors] List of priors, one per target image instance
        """
        priors_per_image: List[Priors] = []

        reference_features = reference_features[0]
        flattened_global_features = reference_features.global_features.reshape(
            -1, reference_features.global_features.shape[-1]
        )
        reference_masks = self._merge_masks(reference_masks)

        for i, target_features in enumerate(target_features_list):
            priors = Priors()
            similarity_map = (
                flattened_global_features @ target_features.global_features.T
            )

            for class_id, mask in reference_masks.data.items():
                (
                    matched_indices,
                    similarity_scores,
                    _,
                    background_point_info,
                ) = self._perform_matching(similarity_map, mask)
                bg_target_idx, bg_similarity_score = background_point_info

                # Process foreground points
                if len(similarity_scores) > 0:
                    fg_points = self._extract_point_coordinates(
                        matched_indices, similarity_scores
                    )
                    image_level_fg_points = self._transform_to_image_coordinates(
                        fg_points,
                        original_image_size=self._state.target_images[i].size,
                    )
                    fg_point_labels = torch.ones(
                        (len(image_level_fg_points), 1),
                        device=image_level_fg_points.device,
                    )
                    image_level_fg_points = torch.cat(
                        [image_level_fg_points, fg_point_labels], dim=1
                    )
                    fg_bg_points = image_level_fg_points
                else:
                    # Add empty tensor if no foreground points
                    fg_bg_points = torch.empty(0, 4, device=similarity_map.device)

                # Process background point
                if bg_target_idx is not None:
                    bg_point = self._extract_point_coordinates(
                        [None, bg_target_idx.unsqueeze(0)],  # Pass target index
                        bg_similarity_score.unsqueeze(0),  # Pass similarity score
                    )
                    image_level_bg_point = self._transform_to_image_coordinates(
                        bg_point,
                        original_image_size=self._state.target_images[i].size,
                    )
                    bg_point_label = torch.zeros(
                        (1, 1), device=image_level_bg_point.device
                    )
                    image_level_bg_point = torch.cat(
                        [image_level_bg_point, bg_point_label], dim=1
                    )
                    fg_bg_points = torch.cat([fg_bg_points, image_level_bg_point])
                else:
                    # Handle case where no background point could be found (e.g., empty mask)
                    # Depending on desired behavior, could add an empty tensor or skip
                    print(f"No BG point found for class {class_id}")

                priors.points.add(fg_bg_points, class_id)
            priors_per_image.append(priors)

        return priors_per_image

    def _perform_matching(
        self,
        similarity_map: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[
        list, torch.Tensor, list, Tuple[torch.Tensor | None, torch.Tensor | None]
    ]:
        """
        Perform bidirectional matching using the similarity map.
        Linear sum assignment finds the optimal pairing between masked reference features and target features to maximize overall similarity.

        Args:
            similarity_map: torch.Tensor - Similarity matrix [num_ref_features, num_target_features]
            mask: torch.Tensor - Mask [num_ref_features]
        Returns:
            tuple containing:
                matched_indices: list - Indices of matched foreground points [ref_indices, target_indices]
                similarity_scores: torch.Tensor - Similarity scores of matched foreground points
                indices_forward: list - Original forward matching indices
                background_point: tuple - (target_index, similarity_score) for the background point, or (None, None)
        """
        masked_ref_indices = mask.flatten().nonzero(as_tuple=True)[0]
        if masked_ref_indices.numel() == 0:
            # Handle case where mask is empty
            return (
                [torch.empty(0, dtype=torch.int64, device=similarity_map.device)] * 2,
                torch.empty(
                    0, dtype=similarity_map.dtype, device=similarity_map.device
                ),
                [torch.empty(0, dtype=torch.int64, device=similarity_map.device)] * 2,
                (None, None),
            )

        # Forward matching (reference -> target)
        forward_sim = similarity_map[
            masked_ref_indices
        ]  # select only the features within the mask

        # Find background point (minimum similarity within the masked reference features)
        # First, find the minimum similarity for each target feature across all masked ref features
        # Then, find the target feature with the overall minimum similarity
        min_sim_per_target, _ = torch.min(forward_sim, dim=0)
        min_similarity_score, min_target_idx = torch.min(min_sim_per_target, dim=0)

        # Perform linear sum assignment for foreground points
        indices_forward = linear_sum_assignment(
            forward_sim.cpu().numpy(), maximize=True
        )
        indices_forward = [
            torch.as_tensor(index, dtype=torch.int64, device=similarity_map.device)
            for index in indices_forward
        ]
        # Map masked reference indices back to original similarity map indices
        original_ref_indices = masked_ref_indices[indices_forward[0]]
        sim_scores_forward = similarity_map[original_ref_indices, indices_forward[1]]

        # Backward matching (target -> reference)
        backward_sim = similarity_map.t()[indices_forward[1]]
        indices_backward = linear_sum_assignment(
            backward_sim.cpu().numpy(), maximize=True
        )
        indices_backward = [
            torch.as_tensor(index, dtype=torch.int64, device=similarity_map.device)
            for index in indices_backward
        ]

        # Filter matches - keep only those where backward match points back to a reference feature within the mask
        # indices_backward[1] contains indices relative to the original similarity map's reference dimension
        indices_to_keep = torch.isin(indices_backward[1], masked_ref_indices)

        percentage_to_keep = (
            indices_to_keep.sum() / len(indices_to_keep)
            if len(indices_to_keep) > 0
            else 0
        )
        print(
            f"Of the total amount of {len(indices_to_keep)} matches, {indices_to_keep.sum()} were kept. ({percentage_to_keep * 100:.2f}%)"
        )

        # Ensure indices_forward uses original reference indices before filtering
        original_indices_forward = [original_ref_indices, indices_forward[1]]

        if indices_to_keep.any():  # Check if any True values exist
            filtered_indices = [
                original_indices_forward[0][indices_to_keep],
                original_indices_forward[1][indices_to_keep],
            ]
            filtered_sim_scores = sim_scores_forward[indices_to_keep]
        else:
            # If no matches pass the filter, keep the original matches from forward assignment
            filtered_indices = original_indices_forward
            filtered_sim_scores = sim_scores_forward

        background_point_info = (min_target_idx, min_similarity_score)

        return (
            filtered_indices,
            filtered_sim_scores,
            original_indices_forward,
            background_point_info,
        )

    def _extract_point_coordinates(
        self, matched_indices: list, similarity_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract point coordinates from matched indices.

        Args:
            matched_indices: List of matched indices [reference_indices, target_indices]
            similarity_scores: Similarity scores for the matched points

        Returns:
            torch.Tensor: Points with their similarity scores (N, 3)
        """
        # Get target indices
        target_indices = matched_indices[1]

        # Extract y and x coordinates from the target indices
        feature_size = self._state.encoder_feature_size
        y_coords = target_indices // feature_size
        x_coords = target_indices % feature_size

        # Stack coordinates with similarity scores
        points = torch.stack([x_coords, y_coords, similarity_scores], dim=1)

        return points

    def _transform_to_image_coordinates(
        self, points: torch.Tensor, original_image_size: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform points from feature grid coordinates to original image coordinates.

        Args:
            points: Points in feature grid coordinates (x, y, score)
            original_image_size: Original image size (height, width)

        Returns:
            torch.Tensor: Points in image coordinates (x, y, score)
        """
        # Get encoder configuration from state
        patch_size = self._state.encoder_patch_size
        encoder_input_size = self._state.encoder_input_size

        # Convert feature grid coordinates to patch coordinates
        x_image = points[:, 0] * patch_size + patch_size // 2
        y_image = points[:, 1] * patch_size + patch_size // 2

        # Scale to original image size
        scale_w = original_image_size[1] / encoder_input_size
        scale_h = original_image_size[0] / encoder_input_size

        x_image = x_image * scale_w
        y_image = y_image * scale_h

        # Combine with similarity scores and round coordinates to nearest integer
        image_points = torch.stack(
            [
                torch.round(x_image).to(torch.int32),
                torch.round(y_image).to(torch.int32),
                points[:, 2],
            ],
            dim=1,
        )

        return image_points

    def _merge_masks(self, reference_masks: List[Masks]) -> Masks:
        """
        Merge the reference masks from multiple instances into a single Masks object,
        concatenating masks for the same class ID.

        Masks are merged so we can do linear sum assignment in one go over multiple reference images (multi-shot).
        """
        merged_masks = Masks()
        for masks_instance in reference_masks:
            for class_id, mask_tensor in masks_instance.data.items():
                merged_masks.add(mask_tensor, class_id)
        return merged_masks
