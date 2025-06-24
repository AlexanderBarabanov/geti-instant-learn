"""PerDino pipeline."""
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import torch
import numpy as np

from visionprompt.filters.masks import ClassOverlapMaskFilter, MaskFilter
from visionprompt.filters.priors.max_point_filter import MaxPointFilter
from visionprompt.models.per_segment_anything import SamPredictor
from visionprompt.pipelines.pipeline_base import Pipeline
from visionprompt.processes.encoders import DinoEncoder, Encoder
from visionprompt.processes.feature_selectors import (
    AverageFeatures,
    FeatureSelector,
)
from visionprompt.processes.mask_processors import MaskProcessor, MasksToPolygons
from visionprompt.processes.prompt_generators import (
    GridPromptGenerator,
)
from visionprompt.processes.segmenters import SamDecoder, Segmenter
from visionprompt.processes.similarity_matchers import (
    CosineSimilarity,
    SimilarityMatcher,
)
from visionprompt.types import Image, Priors, Results
from visionprompt.utils.decorators import track_duration

if TYPE_CHECKING:
    from visionprompt.filters.priors.prior_filter_base import PriorFilter
    from visionprompt.processes.prompt_generators.prompt_generator_base import (
        PromptGenerator,
    )


class PerDino(Pipeline):
    """This is the PerDino algorithm pipeline.

    It is very similar to the PerSam pipeline but uses DinoV2 for encoding the images.

    Examples:
        >>> import torch
        >>> import numpy as np
        >>> from visionprompt.pipelines import PerDino
        >>> from visionprompt.types import Image, Priors, Results
        >>> from visionprompt.models.models import load_sam_model
        >>>
        >>> # Load a real SAM model for the doctest
        >>> sam_predictor = load_sam_model(backbone_name="MobileSAM")
        >>> perdino = PerDino(sam_predictor=sam_predictor)
        >>>
        >>> # Create mock inputs
        >>> ref_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        >>> target_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        >>> ref_priors = Priors()
        >>> ref_priors.masks.add(torch.ones(30, 30, dtype=torch.bool), class_id=1)
        >>>
        >>> # Run learn and infer
        >>> learn_results = perdino.learn([Image(ref_image)], [ref_priors])
        >>> infer_results = perdino.infer([Image(target_image)])
        >>>
        >>> isinstance(learn_results, Results) and isinstance(infer_results, Results)
        True
        >>> infer_results.masks is not None
        True
        >>> infer_results.annotations is not None
        True
    """

    def __init__(
        self,
        sam_predictor: SamPredictor,
        num_foreground_points: int = 40,
        num_background_points: int = 2,
        apply_mask_refinement: bool = True,
        skip_points_in_existing_masks: bool = True,
        num_grid_cells: int = 16,
        similarity_threshold: float = 0.65,
        mask_similarity_threshold: float | None = 0.42,
        precision: torch.dtype = torch.bfloat16,
        compile_models: bool = False,
        verbose: bool = False,
        image_size: int | tuple[int, int] | None = None,
    ) -> None:
        """Initialize the PerDino pipeline.

        Args:
            sam_predictor: The SAM predictor to use.
            num_foreground_points: The number of foreground points to use.
            num_background_points: The number of background points to use.
            apply_mask_refinement: Whether to apply mask refinement.
            skip_points_in_existing_masks: Whether to skip points in existing masks.
            num_grid_cells: The number of grid cells to use.
            similarity_threshold: The similarity threshold for the similarity matcher.
            mask_similarity_threshold: The similarity threshold for the mask.
            precision: The precision to use for the model.
            compile_models: Whether to compile the models.
            verbose: Whether to print verbose output of the model optimization process.
            image_size: The size of the image to use, if None, the image will not be resized.
        """
        super().__init__(image_size=image_size)

        self.encoder: Encoder = DinoEncoder(
            precision=precision,
            compile_models=compile_models,
            verbose=verbose,
        )
        self.feature_selector: FeatureSelector = AverageFeatures()
        self.similarity_matcher: SimilarityMatcher = CosineSimilarity()
        self.prompt_generator: PromptGenerator = GridPromptGenerator(
            num_grid_cells=num_grid_cells,
            similarity_threshold=similarity_threshold,
            num_bg_points=num_background_points,
        )
        self.point_filter: PriorFilter = MaxPointFilter(
            max_num_points=num_foreground_points,
        )
        self.segmenter: Segmenter = SamDecoder(
            sam_predictor=sam_predictor,
            apply_mask_refinement=apply_mask_refinement,
            mask_similarity_threshold=mask_similarity_threshold,
            skip_points_in_existing_masks=skip_points_in_existing_masks,
        )
        self.mask_processor: MaskProcessor = MasksToPolygons()
        self.class_overlap_mask_filter: MaskFilter = ClassOverlapMaskFilter()
        self.reference_features = None

    @track_duration
    def learn(self, reference_images: list[Image], reference_priors: list[Priors]) -> Results:
        """Perform learning step on the reference images and priors."""
        reference_images = self.resize_images(reference_images)
        reference_priors = self.resize_masks(reference_priors)

        # Start running the pipeline
        reference_features, _ = self.encoder(
            reference_images,
            reference_priors,
        )
        self.reference_features = self.feature_selector(reference_features)
        return Results()

    @track_duration
    def infer(self, target_images: list[Image]) -> Results:
        """Perform inference step on the target images."""
        target_images = self.resize_images(target_images)

        # Start running the pipeline
        target_features, _ = self.encoder(target_images)
        similarities = self.similarity_matcher(self.reference_features, target_features, target_images)
        priors = self.prompt_generator(similarities, target_images)
        priors = self.point_filter(priors)
        masks, used_points = self.segmenter(target_images, priors, similarities)
        masks = self.class_overlap_mask_filter(masks, used_points)
        annotations = self.mask_processor(masks)

        # write output
        results = Results()
        results.priors = priors
        results.used_points = used_points
        results.masks = masks
        results.annotations = annotations
        results.similarities = similarities
        return results
