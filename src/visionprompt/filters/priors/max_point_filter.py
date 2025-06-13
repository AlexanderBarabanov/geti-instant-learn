"""This module provides a filter that choces the top K foreground points."""
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from visionprompt.filters.priors.prior_filter_base import PriorFilter
from visionprompt.types import Priors


class MaxPointFilter(PriorFilter):
    """Filter that reduces the number of points in priors to a maximum value.

    This selects the points with the highest scores.

    Example:
        >>> filter = MaxPointFilter(max_num_points=40)
        >>> filtered_priors = filter(priors)
    """

    def __init__(self, max_num_points: int = 40) -> None:
        """Initialize the max point filter.

        Args:
            state: The state object
            max_num_points: Maximum number of points to keep per class
        """
        super().__init__()
        self.max_num_points = max_num_points

    def __call__(self, priors: list[Priors]) -> list[Priors]:
        """Filter points in the priors, keeping the ones with the highest scores.

        Modifies the priors in-place to preserve all other information.

        Args:
            priors: List of Priors objects to filter

        Returns:
            The same Priors list with filtered points
        """
        for prior in priors:
            for class_id, points_per_sim_map in prior.points.data.items():
                for i, points in enumerate(points_per_sim_map):
                    filtered_points = self._filter_points(points)
                    prior.points.data[class_id][i] = filtered_points

        return priors

    def _filter_points(self, points: torch.Tensor) -> torch.Tensor:
        """Filter a single list of points based on scores. This method adds all background points.

        Args:
            points: Tensor of points with shape (N, 3) where each row is [x, y, score]

        Returns:
            Filtered points tensor
        """
        # If points is empty or fewer than max_num_points, return as is
        if points.shape[0] <= self.max_num_points:
            return points

        fg_indices = (points[:, 3] == 1).nonzero()[:, 0]
        bg_indices = (points[:, 3] == 0).nonzero()[:, 0]

        fg_points = points[fg_indices]
        bg_points = points[bg_indices]

        _, fg_indices_sorted = torch.sort(fg_points[:, 2], descending=True)
        fg_indices_select = fg_indices_sorted[: self.max_num_points]
        fg_points_select = fg_points[fg_indices_select]

        # return best matching foreground points and add all background_points
        return torch.cat([fg_points_select, bg_points])
