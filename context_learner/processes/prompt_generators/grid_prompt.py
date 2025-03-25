from typing import List

import torch

from context_learner.processes.prompt_generators.prompt_generator_base import (
    PromptGenerator,
)
from context_learner.types.priors import Priors, Prompt
from context_learner.types.similarities import Similarities
from context_learner.types.state import State


class GridPromptGenerator(PromptGenerator):
    def __init__(
        self,
        state: State,
        downsizing: int = 64,
        similarity_threshold: float = 0.65,
        num_bg_points: int = 1,
    ):
        super().__init__(state)
        self.downsizing = downsizing
        self.similarity_threshold = similarity_threshold
        self.num_bg_points = num_bg_points

    def __call__(
        self,
        image_similarities: List[Similarities],
    ) -> List[Priors]:
        """
        This generates prompt candidates (or priors) based on the similarities between the reference and target images.
        It uses a grid based approach to create multi object aware prompt for the segmenter.

        Args:
            image_similarities: List[Similarities] List of similarities, one per target image instance

        Returns:
            List[Priors] List of priors, one per target image instance
        """
        encoder_input_size = self._state.encoder_input_size
        priors_per_image: List[Priors] = []

        for i, similarities_per_image in enumerate(image_similarities):
            priors = Priors()
            original_image_size = self._state.target_images[i].size
            for class_id, similarities in similarities_per_image.data.items():
                for similarity_map in similarities:
                    foreground_points = self._get_foreground_points(
                        similarity_map, original_image_size, encoder_input_size
                    )

                    # Skip if no foreground points found
                    if len(foreground_points) == 0:
                        # add empty points for this similarity map
                        priors.points.add(
                            torch.empty((0, 3), device=foreground_points.device),
                            class_id,
                        )
                        continue

                    background_points = self._get_background_points(similarity_map)

                    fg_point_labels = torch.ones(
                        (len(foreground_points), 1), device=foreground_points.device
                    )
                    bg_point_labels = torch.zeros(
                        (len(background_points), 1), device=background_points.device
                    )

                    all_points = torch.cat(
                        [
                            torch.cat([foreground_points, fg_point_labels], dim=1),
                            torch.cat([background_points, bg_point_labels], dim=1),
                        ],
                        dim=0,
                    )
                    # Add the all_points as a list item. Cannot be stacked since they have different shapes per mask
                    priors.points.add(all_points, class_id)

            priors = self._filter_duplicate_points(priors)
            priors_per_image.append(priors)

        return priors_per_image

    def _get_foreground_points(
        self,
        similarity: torch.Tensor,
        original_image_size: torch.Tensor,
        encoder_input_size: int,
    ) -> torch.Tensor:
        """
        Select foreground points based on the similarity mask and grid-based filtering.
        Args:
            similarity: Similarity mask tensor
            original_image_size: Original image size
            encoder_input_size: Size of the encoder input
        Returns:
            Foreground points coordinates and scores with shape (N, 3) where each row is [x, y, score]
        """
        point_coords = torch.where(similarity > self.similarity_threshold)
        foreground_coords = torch.stack(
            point_coords[::-1] + (similarity[point_coords],), axis=0
        ).T

        # skip if there are no foreground coords
        if len(foreground_coords) == 0:
            return torch.empty((0, 3), device=similarity.device)

        # create grid of the original image size
        ratio = encoder_input_size / max(original_image_size)
        width = int(original_image_size[1] * ratio)
        number_of_grid_cells = width // self.downsizing

        # create grid numbers
        idx_grid = (
            foreground_coords[:, 1] * ratio // self.downsizing * number_of_grid_cells
            + foreground_coords[:, 0] * ratio // self.downsizing
        )

        # filter out points that are in the same location
        idx_unique = torch.unique(idx_grid.int())
        matched_matrix = idx_grid.unsqueeze(-1) == idx_unique
        # sample foreground coords matched by matched matrix
        matched_grid = foreground_coords.unsqueeze(1) * matched_matrix.unsqueeze(-1)

        # get highest score points for each grid cell
        matched_indices = torch.topk(matched_grid[..., -1], k=1, dim=0, largest=True)[
            1
        ][0]
        points_scores = matched_grid[matched_indices].diagonal().T

        # sort by highest score
        sorted_indices = torch.argsort(points_scores[:, -1], descending=True)
        points_scores = points_scores[sorted_indices]

        return points_scores

    def _get_background_points(self, similarity: torch.Tensor) -> torch.Tensor:
        """
        Select background points based on the similarity mask.
        Args:
            similarity: Similarity mask tensor
        Returns:
            Background points coordinates with shape (num_bg_points, 3) where each row is [x, y, score]
        """
        if self.num_bg_points == 0:
            return torch.empty((0, 3), device=similarity.device)

        _, w_sim = similarity.shape
        bg_values, bg_indices = torch.topk(
            similarity.flatten(), self.num_bg_points, largest=False
        )
        bg_x = (bg_indices // w_sim).unsqueeze(0)
        bg_y = bg_indices - bg_x * w_sim

        # Stack coordinates and similarity scores
        bg_coords = torch.cat((bg_y, bg_x, bg_values.unsqueeze(0)), dim=0).T

        return bg_coords.float()
