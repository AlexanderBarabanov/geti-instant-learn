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
                matched_indices, similarity_scores, _ = self._perform_matching(
                    similarity_map, mask
                )

                if len(similarity_scores) > 0:
                    points = self._extract_point_coordinates(
                        matched_indices, similarity_scores
                    )
                else:
                    priors.points.add(
                        torch.empty((0, 4), device=similarity_map.device),
                        class_id,
                    )
                    continue

                # Transform points to image level coordinates
                image_level_points = self._transform_to_image_coordinates(
                    points,
                    original_image_size=self._state.target_images[i].size,
                )

                # Add label information
                fg_point_labels = torch.ones(
                    (len(image_level_points), 1), device=image_level_points.device
                )
                image_level_points = torch.cat(
                    [image_level_points, fg_point_labels], dim=1
                )

                # Add the points to priors
                priors.points.add(image_level_points, class_id)

            priors_per_image.append(priors)

        return priors_per_image

    def _perform_matching(
        self,
        similarity_map: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[list, torch.Tensor, list]:
        """
        Perform bidirectional matching using the similarity map.

        Args:
            similarity_map: torch.Tensor - Similarity matrix [num_ref_features, num_target_features]
            mask: torch.Tensor - Mask [num_ref_features]
        Returns:
            tuple containing:
                matched_indices: list - Indices of matched points
                similarity_scores: torch.Tensor - Similarity scores of matched points
                indices_forward: list - Original forward matching indices
        """

        # Forward matching (reference -> target)
        forward_sim = similarity_map[
            mask.flatten().bool()
        ]  # select only the features within the mask
        indices_forward = linear_sum_assignment(forward_sim.cpu(), maximize=True)
        indices_forward = [
            torch.as_tensor(index, dtype=torch.int64, device=similarity_map.device)
            for index in indices_forward
        ]
        sim_scores_forward = similarity_map[indices_forward[0], indices_forward[1]]
        non_zero_mask_indices = mask.flatten().nonzero()[:, 0]

        # Backward matching (target -> reference)
        backward_sim = similarity_map.t()[indices_forward[1]]
        indices_backward = linear_sum_assignment(backward_sim.cpu(), maximize=True)
        indices_backward = [
            torch.as_tensor(index, dtype=torch.int64, device=similarity_map.device)
            for index in indices_backward
        ]

        # Filter matches - keep only those where backward match points to a reference feature
        indices_to_keep = torch.isin(indices_backward[1], non_zero_mask_indices)

        if not (indices_to_keep == False).all().item():
            filtered_indices = [
                indices_forward[0][indices_to_keep],
                indices_forward[1][indices_to_keep],
            ]
            filtered_sim_scores = sim_scores_forward[indices_to_keep]
        else:
            # If no matches pass the filter, keep the original matches
            filtered_indices = indices_forward
            filtered_sim_scores = sim_scores_forward

        return filtered_indices, filtered_sim_scores, indices_forward

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

        # Combine with similarity scores and convert coordinates to integers
        image_points = torch.stack(
            [x_image.to(torch.int32), y_image.to(torch.int32), points[:, 2]], dim=1
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
