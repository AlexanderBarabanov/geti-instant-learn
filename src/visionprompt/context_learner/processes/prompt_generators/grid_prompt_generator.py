# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import torch

from visionprompt.context_learner.processes.prompt_generators.prompt_generator_base import SimilarityPromptGenerator
from visionprompt.context_learner.types import Priors, Similarities
from visionprompt.context_learner.types.image import Image


class GridPromptGenerator(SimilarityPromptGenerator):
    """This class generates prompts for the segmenter.

    This is based on the similarities between the reference and target images.
    """

    def __init__(
        self,
        encoder_input_size: int,
        downsizing: int = 64,
        similarity_threshold: float = 0.65,
        num_bg_points: int = 1,
    ) -> None:
        """Generate prompts for the segmenter based on the similarities between the reference and target images.

        Args:
            encoder_input_size: int the size of the encoder input image.
            downsizing: int the downsizing factor for the grid
            similarity_threshold: float the threshold for the similarity mask
            num_bg_points: int the number of background points to sample
        """
        super().__init__()
        self.downsizing = downsizing
        self.similarity_threshold = similarity_threshold
        self.num_bg_points = num_bg_points
        self.encoder_input_size = encoder_input_size

    def __call__(
        self, target_similarities: list[Similarities] | None = None, target_images: list[Image] | None = None
    ) -> list[Priors]:
        """This generates prompt candidates (or priors).

        Ths is based on the similarities between the reference and target images.
        It uses a grid based approach to create multi object aware prompt for the segmenter.

        Args:
            target_similarities: List[Similarities] List of similarities, one per target image instance
            target_images: List[Image] List of target image instances

        Returns:
            List[Priors] List of priors, one per target image instance
        """
        encoder_input_size = self.encoder_input_size
        priors_per_image: list[Priors] = []

        for i, similarities_per_image in enumerate(target_similarities):
            priors = Priors()
            original_image_size = target_images[i].size
            for class_id, similarities in similarities_per_image.data.items():
                background_points = self._get_background_points(similarities)

                for similarity_map in similarities:
                    foreground_points = self._get_foreground_points(
                        similarity_map,
                        original_image_size,
                        encoder_input_size,
                    )

                    # Skip if no foreground points found
                    if len(foreground_points) == 0:
                        # add empty points for this similarity map
                        priors.points.add(
                            torch.empty((0, 4), device=foreground_points.device),
                            class_id,
                        )
                        continue

                    fg_point_labels = torch.ones(
                        (len(foreground_points), 1),
                        device=foreground_points.device,
                    )
                    bg_point_labels = torch.zeros(
                        (len(background_points), 1),
                        device=background_points.device,
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
        """Select foreground points based on the similarity mask and grid-based filtering.

        Args:
            similarity: Similarity mask tensor
            original_image_size: Original image size
            encoder_input_size: Size of the encoder input
        Returns:
            Foreground points coordinates and scores with shape (N, 3) where each row is [x, y, score]
        """
        point_coords = torch.where(similarity > self.similarity_threshold)
        foreground_coords = torch.stack(
            (*point_coords[::-1], similarity[point_coords]),
            axis=0,
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
        matched_indices = torch.topk(matched_grid[..., -1], k=1, dim=0, largest=True)[1][0]
        points_scores = matched_grid[matched_indices].diagonal().T

        # sort by highest score
        sorted_indices = torch.argsort(points_scores[:, -1], descending=True)
        return points_scores[sorted_indices]

    def _get_background_points(self, similarity: torch.Tensor) -> torch.Tensor:
        """Select background points based on the similarity mask.

        Args:
            similarity: Similarity mask tensor
        Returns:
            Background points coordinates with shape (num_bg_points, 3) where each row is [x, y, score]
        """
        if self.num_bg_points == 0:
            return torch.empty((0, 3), device=similarity.device)

        # Stack all similarity maps
        if similarity.ndim == 3:
            similarity = similarity.sum(dim=0)
        _, w_sim = similarity.shape
        bg_values, bg_indices = torch.topk(
            similarity.flatten(),
            self.num_bg_points,
            largest=False,
        )
        bg_x = (bg_indices // w_sim).unsqueeze(0)
        bg_y = bg_indices - bg_x * w_sim

        # Stack coordinates and similarity scores
        bg_coords = torch.cat((bg_y, bg_x, bg_values.unsqueeze(0)), dim=0).T

        return bg_coords.float()
