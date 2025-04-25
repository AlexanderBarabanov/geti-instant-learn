# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
from typing import List

from context_learner.filters.priors.max_point_filter import MaxPointFilter
from context_learner.filters.priors.prior_filter_base import PriorFilter
from third_party.Matcher.segment_anything import SamPredictor
from context_learner.filters.masks import MaskFilter, ClassOverlapMaskFilter
from context_learner.pipelines.pipeline_base import Pipeline
from context_learner.processes.encoders import DinoEncoder, Encoder
from context_learner.processes.feature_selectors import (
    FeatureSelector,
    AverageFeatures,
)
from context_learner.processes.mask_processors import MaskProcessor, MasksToPolygons
from context_learner.processes.prompt_generators import (
    GridPromptGenerator,
    PromptGenerator,
)
from context_learner.processes.segmenters import SamDecoder, Segmenter
from context_learner.processes.similarity_matchers import (
    CosineSimilarity,
    SimilarityMatcher,
)
from context_learner.types import Image, Priors, State


class PerDino(Pipeline):
    """
    This is the PerDino algorithm pipeline.
    It is very similar to the PerSam pipeline but uses DinoV2 for encoding the images.
    """

    def __init__(self, sam_predictor: SamPredictor, args: argparse.Namespace):
        super().__init__(args)

        self.encoder: Encoder = DinoEncoder(self._state)
        self.feature_selector: FeatureSelector = AverageFeatures(self._state)
        # self.feature_selector: FeatureSelector = ClusterFeatures(
        #     self._state, num_clusters=self.args.num_clusters
        # )
        self.similarity_matcher: SimilarityMatcher = CosineSimilarity(self._state)
        self.prompt_generator: PromptGenerator = GridPromptGenerator(
            self._state,
            downsizing=32,
            similarity_threshold=self.args.similarity_threshold,
            num_bg_points=self.args.num_background_points,
        )
        self.point_filter: PriorFilter = MaxPointFilter(
            self._state, max_num_points=self.args.num_foreground_points
        )
        self.segmenter: Segmenter = SamDecoder(
            self._state,
            sam_predictor=sam_predictor,
            apply_mask_refinement=self.args.apply_mask_refinement,
            mask_similarity_threshold=self.args.mask_similarity_threshold,
            skip_points_in_existing_masks=self.args.skip_points_in_existing_masks,
        )
        self.mask_processor: MaskProcessor = MasksToPolygons(self._state)
        self.class_overlap_mask_filter: MaskFilter = ClassOverlapMaskFilter(self._state)

    def learn(self, reference_images: List[Image], reference_priors: List[Priors]):
        s: State = self._state
        s.reference_images = reference_images
        s.reference_priors = reference_priors

        # Start running the pipeline
        s.reference_features, s.processed_reference_masks = self.encoder(
            s.reference_images, s.reference_priors
        )
        s.reference_features = self.feature_selector(s.reference_features)

    def infer(self, target_images: List[Image]):
        s: State = self._state
        s.target_images = target_images

        # Start running the pipeline
        s.target_features, _ = self.encoder(s.target_images)
        s.similarities = self.similarity_matcher(
            s.reference_features, s.target_features
        )
        s.priors = self.prompt_generator(s.similarities)
        s.priors = self.point_filter(s.priors)
        s.masks, s.used_points = self.segmenter(
            s.target_images, s.priors, s.similarities
        )
        s.masks = self.class_overlap_mask_filter(s.masks, s.used_points)
        s.annotations = self.mask_processor(s.masks)

        return s.annotations
