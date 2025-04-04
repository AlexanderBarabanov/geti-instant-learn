from typing import List

import torch

from context_learner.filters.priors.prior_filter_base import PriorFilter
from context_learner.types.priors import Priors
from context_learner.types.state import State


class MaxPointFilter(PriorFilter):
    """
    Filter that reduces the number of points in priors to a maximum value,
    selecting the points with the highest scores.
    """

    def __init__(self, state: State, max_num_points: int = 40):
        """
        Initialize the max point filter.

        Args:
            state: The state object
            max_num_points: Maximum number of points to keep per class
        """
        super().__init__(state)
        self.max_num_points = max_num_points

    def __call__(self, priors: List[Priors]) -> List[Priors]:
        """
        Filter points in the priors, keeping the ones with highest scores.
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
        """
        Filter a single list of points based on scores.

        Args:
            points: Tensor of points with shape (N, 3) where each row is [x, y, score]

        Returns:
            Filtered points tensor
        """
        # If points is empty or fewer than max_num_points, return as is
        if points.shape[0] <= self.max_num_points:
            return points
        _, indices = torch.sort(points[:, 2], descending=True)
        filtered_indices = indices[: self.max_num_points]
        return points[filtered_indices]
