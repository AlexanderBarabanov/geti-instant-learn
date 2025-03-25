from typing import List

import torch

from context_learner.processes.process_base import Process
from context_learner.types.priors import Priors
from context_learner.types.similarities import Similarities


class PromptGenerator(Process):
    def __call__(self, similarities: List[Similarities]) -> List[Priors]:
        """
        This method extracts priors from similarities.

        Args:
            similarities: The similarities between reference features and target features.

        Returns:
            A priors that have been created from the similarities.

        Examples:
            >>> from context_learner.types.state import State
            >>> state = State()
            >>> prompt_gen = PromptGenerator(state=state)
            >>> r = prompt_gen([Similarities()])
        """
        return [Priors()]

    def _filter_duplicate_points(self, priors: Priors) -> Priors:
        """
        Filter out duplicate points, handling foreground and background points separately.
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
                    foreground_points[:, :2], dim=0, return_inverse=True
                )
                unique_points_foreground = foreground_points[unique_indices]

                # Filter background points (keep lowest scores)
                background_points = class_points[class_points[:, 3] == 0]
                sorted_indices = torch.argsort(
                    background_points[:, 2], descending=False
                )
                background_points = background_points[sorted_indices]
                _, unique_indices = torch.unique(
                    background_points[:, :2], dim=0, return_inverse=True
                )
                unique_points_background = background_points[unique_indices]

                # Update points for this map and class_id
                priors.points.data[class_id][similarity_map_id] = torch.cat(
                    [unique_points_foreground, unique_points_background], dim=0
                )

        return priors
