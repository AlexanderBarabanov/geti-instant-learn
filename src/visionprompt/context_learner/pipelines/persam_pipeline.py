# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from visionprompt.context_learner.filters.masks import ClassOverlapMaskFilter, MaskFilter
from visionprompt.context_learner.filters.priors import MaxPointFilter
from visionprompt.context_learner.pipelines.pipeline_base import Pipeline
from visionprompt.context_learner.processes.encoders import Encoder, SamEncoder
from visionprompt.context_learner.processes.feature_selectors import AverageFeatures, FeatureSelector
from visionprompt.context_learner.processes.mask_processors import MaskProcessor, MasksToPolygons
from visionprompt.context_learner.processes.prompt_generators import (
    GridPromptGenerator,
)
from visionprompt.context_learner.processes.prompt_generators.prompt_generator_base import PromptGenerator
from visionprompt.context_learner.processes.segmenters import SamDecoder, Segmenter
from visionprompt.context_learner.processes.similarity_matchers import (
    CosineSimilarity,
    SimilarityMatcher,
)
from visionprompt.context_learner.types import Image, Priors, Results
from visionprompt.third_party.Matcher.segment_anything import SamPredictor
from visionprompt.utils.decorators import track_duration

if TYPE_CHECKING:
    from visionprompt.context_learner.filters.priors.prior_filter_base import PriorFilter


class PerSam(Pipeline):
    """This is the PerSam algorithm pipeline.

    It's based on the paper "Personalize Segment Anything Model with One Shot"
    https://arxiv.org/abs/2305.03048.

    It matches reference objects to target images by comparing their features extracted by SAM
    and using Cosine Similarity. A grid prompt generator is used to generate prompts for the
    segmenter and to allow for multi object target images.
    """

    def __init__(
        self,
        sam_predictor: SamPredictor,
        num_foreground_points: int,
        num_background_points: int,
        apply_mask_refinement: bool,
        skip_points_in_existing_masks: bool,
        num_grid_cells: int,
        similarity_threshold: float,
        mask_similarity_threshold: float,
        image_size: int | tuple[int, int] | None = None,
    ) -> None:
        super().__init__(image_size=image_size)
        self.encoder: Encoder = SamEncoder(sam_predictor=sam_predictor)
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
        similarities = self.similarity_matcher(
            self.reference_features,
            target_features,
            target_images,
        )
        priors = self.prompt_generator(similarities, target_images)
        priors = self.point_filter(priors)
        masks, used_points = self.segmenter(target_images, priors)
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
