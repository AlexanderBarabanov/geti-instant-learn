from typing import List

from context_learner.pipelines.pipeline_base import Pipeline
from context_learner.processes.encoders.encoder_base import Encoder
from context_learner.processes.encoders.sam_encoder import SamEncoder
from context_learner.processes.feature_selectors.average_features import AverageFeatures
from context_learner.processes.feature_selectors.feature_selector_base import FeatureSelector
from context_learner.processes.mask_processors.mask_processor_base import MaskProcessor
from context_learner.processes.prompt_generators.grid_prompt import GridPrompt
from context_learner.processes.prompt_generators.prompt_generator_base import PromptGenerator
from context_learner.processes.segmenters.sam_decoder import SamDecoder
from context_learner.processes.segmenters.segmenter_base import Segmenter
from context_learner.processes.similarity_matchers.cosine_similarity import CosineSimilarity
from context_learner.processes.similarity_matchers.similarity_matcher_base import SimilarityMatcher
from context_learner.types.image import Image
from context_learner.types.annotations import Annotations


class PerSam(Pipeline):

    """
    This is the PerSam algorithm pipeline

    Currently this is a dummy implementation to test the flow

    >>> p = PerSam()
    >>> p.learn([Image()] * 3, [Annotations()] * 3)
    >>> a = p.infer([Image()])
    >>> isinstance(a[0], Annotations)
    True
    """

    def __init__(self):
        super().__init__()
        self.encoder: Encoder = SamEncoder(self._state)
        self.feature_selector: FeatureSelector = AverageFeatures(self._state)
        self.similarity_matcher: SimilarityMatcher = CosineSimilarity(self._state)
        self.prompt_generator: PromptGenerator = GridPrompt(self._state)
        self.segmenter: Segmenter = SamDecoder(self._state)
        self.mask_processor: MaskProcessor = MaskProcessor(self._state)


    def learn(self, reference_images: List[Image], reference_annotations: List[Annotations]):
        # This is an initial dummy implementation of learn
        s = self._state  # More compact name

        # Set input inside the state for convenience
        s.reference_images = reference_images
        s.reference_annotations = reference_annotations

        # Start running the pipeline
        s.reference_features = self.encoder(s.reference_images, s.reference_annotations)
        s.reference_features = self.feature_selector(s.reference_features)

    def infer(self, target_images: List[Image]):
        # This is an initial dummy implementation of infer
        s = self._state  # More compact name

        # Set input inside the state for convenience
        s.target_images = target_images

        # Start running the pipeline
        s.target_features = self.encoder(s.target_images)
        s.similarities = self.similarity_matcher(s.reference_features, s.target_features)
        s.priors = self.prompt_generator(s.similarities)
        s.masks = self.segmenter(s.priors)
        s.annotations = self.mask_processor(s.masks)

        return s.annotations