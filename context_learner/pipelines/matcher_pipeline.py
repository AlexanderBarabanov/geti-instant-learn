import argparse
from typing import List

from third_party.Matcher.segment_anything import SamPredictor
from context_learner.filters.masks import MaskFilter, ClassOverlapMaskFilter
from context_learner.filters.priors import MaxPointFilter, PriorFilter
from context_learner.pipelines.pipeline_base import Pipeline
from context_learner.processes.encoders import DinoEncoder, Encoder
from context_learner.processes.feature_selectors import (
    FeatureSelector,
    AllFeaturesSelector,
)
from context_learner.processes.mask_processors import MaskProcessor, MasksToPolygons
from context_learner.processes.prompt_generators import (
    BidirectionalPromptGenerator,
    PromptGenerator,
)
from context_learner.processes.segmenters import SamDecoder, Segmenter
from context_learner.processes.similarity_matchers import (
    CosineSimilarity,
    SimilarityMatcher,
)
from context_learner.types import Priors, State, Image


class Matcher(Pipeline):
    """
    This is the Matcher pipeline. Its based on the paper "[ICLR'24] Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching"
    https://arxiv.org/abs/2305.13310

    Main novelties:
    - Uses DinoV2 patch encoding instead of SAM for encoding the images, resulting in a more robust feature extractor
    - Uses a bidirectional prompt generator to generate prompts for the segmenter
    - Has a more complex mask postprocessing step to remove and merge masks
    """

    def __init__(self, sam_predictor: SamPredictor, args: argparse.Namespace):
        super().__init__(args)

        self.encoder: Encoder = DinoEncoder(self._state)
        self.feature_selector: FeatureSelector = AllFeaturesSelector(self._state)
        self.similarity_matcher: SimilarityMatcher = CosineSimilarity(self._state)
        self.prompt_generator: PromptGenerator = BidirectionalPromptGenerator(
            self._state, num_background_points=self.args.num_background_points
        )
        self.point_filter: PriorFilter = MaxPointFilter(
            self._state, max_num_points=self.args.num_foreground_points
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
        s.priors = self.prompt_generator(
            s.reference_features, s.processed_reference_masks, s.target_features
        )
        s.priors = self.point_filter(s.priors)
        s.masks, s.used_points = self.segmenter(s.target_images, s.priors)
        s.masks = self.class_overlap_mask_filter(s.masks, s.used_points)
        s.annotations = self.mask_processor(s.masks)
