import argparse
from typing import List

from third_party.Matcher.segment_anything import SamPredictor
from context_learner.filters.masks import MaskFilter
from context_learner.filters.masks import ClassOverlapMaskFilter
from context_learner.pipelines.pipeline_base import Pipeline
from context_learner.processes.encoders import SamEncoder, Encoder
from context_learner.processes.feature_selectors import (
    ClusterFeatures,
    FeatureSelector,
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


class PerSam(Pipeline):
    """
    This is the PerSam algorithm pipeline. Its based on the paper "Personalize Segment Anything Model with One Shot"
    https://arxiv.org/abs/2305.03048

    It matches reference objects to target images by comparing their features extracted by SAM and using Cosine Similarity.
    A grid prompt generator is used to generate prompts for the segmenter and to allow for multi object target images.
    """

    def __init__(self, sam_predictor: SamPredictor, args: argparse.Namespace):
        super().__init__(args)

        self.encoder: Encoder = SamEncoder(self._state, sam_predictor)
        # self.feature_selector: FeatureSelector = AverageFeatures(self._state)
        self.feature_selector: FeatureSelector = ClusterFeatures(
            self._state, num_clusters=self.args.num_clusters
        )
        self.similarity_matcher: SimilarityMatcher = CosineSimilarity(self._state)
        self.prompt_generator: PromptGenerator = GridPromptGenerator(
            self._state,
            similarity_threshold=self.args.similarity_threshold,
            num_bg_points=self.args.num_background_points,
        )
        self.segmenter: Segmenter = SamDecoder(
            self._state,
            sam_predictor=sam_predictor,
            apply_mask_refinement=self.args.apply_mask_refinement,
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
        s.masks, s.used_points = self.segmenter(s.target_images, s.priors)
        s.masks = self.class_overlap_mask_filter(s.masks, s.used_points)
        s.annotations = self.mask_processor(s.masks)

        return s.annotations
