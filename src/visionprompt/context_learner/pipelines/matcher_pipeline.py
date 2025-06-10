# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from visionprompt.context_learner.filters.masks import ClassOverlapMaskFilter, MaskFilter
from visionprompt.context_learner.filters.priors import MaxPointFilter, PriorFilter
from visionprompt.context_learner.pipelines.pipeline_base import Pipeline
from visionprompt.context_learner.processes.encoders import DinoEncoder, Encoder
from visionprompt.context_learner.processes.feature_selectors import (
    AllFeaturesSelector,
    FeatureSelector,
)
from visionprompt.context_learner.processes.mask_processors import MaskProcessor, MasksToPolygons
from visionprompt.context_learner.processes.prompt_generators import (
    BidirectionalPromptGenerator,
)
from visionprompt.context_learner.processes.segmenters import SamDecoder, Segmenter
from visionprompt.context_learner.processes.similarity_matchers import (
    CosineSimilarity,
    SimilarityMatcher,
)
from visionprompt.context_learner.types import Image, Priors
from visionprompt.context_learner.types.results import Results
from visionprompt.third_party.Matcher.segment_anything import SamPredictor


class Matcher(Pipeline):
    """This is the Matcher pipeline.

    It's based on the paper "[ICLR'24] Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching"
    https://arxiv.org/abs/2305.13310.

    Main novelties:
    - Uses DinoV2 patch encoding instead of SAM for encoding the images, resulting in a more robust feature extractor
    - Uses a bidirectional prompt generator to generate prompts for the segmenter
    - Has a more complex mask postprocessing step to remove and merge masks

    Note that the post processing mask filtering techniques are different from that of the original paper.
    """

    def __init__(
        self,
        sam_predictor: SamPredictor,
        num_foreground_points: int,
        num_background_points: int,
        apply_mask_refinement: bool,
        mask_similarity_threshold: float,
        skip_points_in_existing_masks: bool,
    ) -> None:
        super().__init__()

        self.encoder: Encoder = DinoEncoder()
        self.feature_selector: FeatureSelector = AllFeaturesSelector()
        self.similarity_matcher: SimilarityMatcher = CosineSimilarity(
            encoder_input_size=self.encoder.encoder_input_size, encoder_patch_size=self.encoder.patch_size
        )
        self.prompt_generator: BidirectionalPromptGenerator = BidirectionalPromptGenerator(
            encoder_input_size=self.encoder.encoder_input_size,
            encoder_patch_size=self.encoder.patch_size,
            encoder_feature_size=self.encoder.feature_size,
            num_background_points=num_background_points,
        )
        self.point_filter: PriorFilter = MaxPointFilter(max_num_points=num_foreground_points)
        self.segmenter: Segmenter = SamDecoder(
            sam_predictor=sam_predictor,
            apply_mask_refinement=apply_mask_refinement,
            mask_similarity_threshold=mask_similarity_threshold,
            skip_points_in_existing_masks=skip_points_in_existing_masks,
        )
        self.mask_processor: MaskProcessor = MasksToPolygons()
        self.class_overlap_mask_filter: MaskFilter = ClassOverlapMaskFilter()
        self.reference_features = None
        self.reference_masks = None

    def learn(self, reference_images: list[Image], reference_priors: list[Priors]) -> Results:
        """Perform learning step on the reference images and priors."""
        # Start running the pipeline
        reference_features, self.reference_masks = self.encoder(reference_images, reference_priors)
        self.reference_features = self.feature_selector(reference_features)
        return Results()

    def infer(self, target_images: list[Image]) -> Results:
        """Perform inference step on the target images."""
        # Start running the pipeline
        target_features, _ = self.encoder(target_images)
        priors, similarities = self.prompt_generator(
            self.reference_features, target_features, self.reference_masks, target_images
        )
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
        return results
