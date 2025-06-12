# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from abc import abstractmethod

import torch

from visionprompt.context_learner.processes import Process
from visionprompt.context_learner.types import Features, Priors, Similarities


class PromptGenerator(Process):
    """This class generates priors."""

    @abstractmethod
    def __call__(self) -> list[Priors]:
        """This method extracts priors."""

    @staticmethod
    def _filter_duplicate_points(priors: Priors) -> Priors:
        """Filter out duplicate points, handling foreground and background points separately.

        This is applied for the points of each similarity map.

        Args:
            priors: Priors object containing point data for each class

        Returns:
            Priors object with duplicates removed, keeping highest scoring foreground points
            and lowest scoring background points
        """
        for class_id, class_points_per_map in priors.points.data.items():
            for similarity_map_id, class_points in enumerate(class_points_per_map):
                # Filter foreground points (keep highest scores)
                foreground_points = class_points[class_points[:, 3] == 1]
                sorted_indices = torch.argsort(foreground_points[:, 2], descending=True)
                foreground_points = foreground_points[sorted_indices]
                _, unique_indices = torch.unique(
                    foreground_points[:, :2],
                    dim=0,
                    return_inverse=True,
                )
                unique_points_foreground = foreground_points[unique_indices]

                # Filter background points (keep lowest scores)
                background_points = class_points[class_points[:, 3] == 0]
                sorted_indices = torch.argsort(
                    background_points[:, 2],
                    descending=False,
                )
                background_points = background_points[sorted_indices]
                _, unique_indices = torch.unique(
                    background_points[:, :2],
                    dim=0,
                    return_inverse=True,
                )
                unique_points_background = background_points[unique_indices]

                # Update points for this map and class_id
                priors.points.data[class_id][similarity_map_id] = torch.cat(
                    [unique_points_foreground, unique_points_background],
                    dim=0,
                )

        return priors

    @staticmethod
    def _convert_points_to_original_size(
        input_coords: torch.Tensor,
        input_map_shape: tuple[int, int],
        original_image_size: tuple[int, int],
    ) -> torch.Tensor:
        """Converts point coordinates from an input map's space to original image space.

        Args:
            input_coords: Tensor of shape (N, k) with [x, y, ...] coordinates.
                                   Assumes input_coords[:, 0] is x and input_coords[:, 1] is y.
            original_image_size: Tuple (width, height) of the original image.
            input_map_shape: Tuple (height, width) of the input similarity map from which points were derived.

        Returns:
            Tensor of shape (N, k) with [x, y, ...] coordinates scaled to original_image_size.
        """
        points_original_coords = input_coords.clone()
        original_width, original_height = original_image_size
        map_w, map_h = input_map_shape
        if map_w == 0 or map_h == 0:
            return points_original_coords

        scale_x = original_width / map_w
        points_original_coords[:, 0] = points_original_coords[:, 0] * scale_x
        scale_y = original_height / map_h
        points_original_coords[:, 1] = points_original_coords[:, 1] * scale_y
        return points_original_coords


class SimilarityPromptGenerator(PromptGenerator):
    """This class generates priors from similarities."""

    @abstractmethod
    def __call__(self, target_similarities: list[Similarities] | None = None) -> tuple:
        """This method extracts priors from similarities.

        Args:
            target_similarities: The similarities between reference features and target features.

        Returns:
            A priors that have been created from the similarities.

        """


class FeaturePromptGenerator(PromptGenerator):
    """This class generates priors from feature maps."""

    @abstractmethod
    def __call__(
        self, reference_features: list[Features] | None = None, target_features: list[Features] | None = None
    ) -> tuple:
        """This method extracts priors from similarities.

        Args:
            reference_features: List of reference features
            target_features: List of target features

        Returns:
            A priors that have been created from the feature maps.

        """
