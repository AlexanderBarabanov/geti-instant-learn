from typing import List

from context_learner.filters.masks.mask_filter_base import MaskFilter
from context_learner.filters.masks.mask_filter_class_overlap import (
    ClassOverlapMaskFilter,
)
from context_learner.filters.priors.max_point_filter import MaxPointFilter
from context_learner.filters.priors.prior_filter_base import PriorFilter
from context_learner.pipelines.pipeline_base import Pipeline
from context_learner.processes.encoders.dino_encoder import DinoEncoder
from context_learner.processes.encoders.encoder_base import Encoder
from context_learner.processes.feature_selectors.feature_selector_base import (
    FeatureSelector,
)
from context_learner.processes.feature_selectors.all_features import AllFeaturesSelector
from context_learner.processes.mask_processors.mask_processor_base import MaskProcessor
from context_learner.processes.mask_processors.mask_to_polygon import MasksToPolygons
from context_learner.processes.prompt_generators.bidirectional_prompt_generator import (
    BidirectionalPromptGenerator,
)
from context_learner.processes.prompt_generators.prompt_generator_base import (
    PromptGenerator,
)
from context_learner.processes.segmenters.sam_decoder import SamDecoder
from context_learner.processes.segmenters.segmenter_base import Segmenter
from context_learner.processes.similarity_matchers.cosine_similarity import (
    CosineSimilarity,
)
from context_learner.processes.similarity_matchers.similarity_matcher_base import (
    SimilarityMatcher,
)
from context_learner.types import Priors, State, Image
from utils.models import load_sam_predictor


class Matcher(Pipeline):
    """
    This is the Matcher pipeline. Its based on the paper "[ICLR'24] Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching"
    https://arxiv.org/abs/2305.13310

    Main novelties:
    - Uses DinoV2 patch encoding instead of SAM for encoding the images
    - Has a more complex mask postprocessing step to remove and merge masks

    """

    def __init__(self):
        super().__init__()
        sam_predictor = load_sam_predictor(sam_name="SAM")

        self.encoder: Encoder = DinoEncoder(self._state)
        self.feature_selector: FeatureSelector = AllFeaturesSelector(self._state)
        self.similarity_matcher: SimilarityMatcher = CosineSimilarity(self._state)
        self.prompt_generator: PromptGenerator = BidirectionalPromptGenerator(
            self._state
        )
        self.point_filter: PriorFilter = MaxPointFilter(self._state, max_num_points=40)
        self.segmenter: Segmenter = SamDecoder(self._state, sam_predictor)
        self.mask_processor: MaskProcessor = MasksToPolygons(self._state)
        self.class_overlap_mask_filter: MaskFilter = ClassOverlapMaskFilter(self._state)

    def learn(self, reference_images: List[Image], reference_priors: List[Priors]):
        s: State = self._state  # More compact name

        # Set input inside the state for convenience
        s.reference_images = reference_images
        s.reference_priors = reference_priors

        # Start running the pipeline
        s.reference_features, s.processed_reference_masks = self.encoder(
            s.reference_images, s.reference_priors
        )
        s.reference_features = self.feature_selector(s.reference_features)

    def infer(self, target_images: List[Image]):
        s: State = self._state  # More compact name

        # Set input inside the state for convenience
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
