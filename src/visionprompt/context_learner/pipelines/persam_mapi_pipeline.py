# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import TYPE_CHECKING

from model_api.models.model import Model
from model_api.models.visual_prompting import SAMLearnableVisualPrompter

from visionprompt.context_learner.pipelines.pipeline_base import Pipeline
from visionprompt.context_learner.processes.encoders.sam_mapi_encoder import SamMAPIEncoder
from visionprompt.context_learner.processes.mask_processors.mask_to_polygon import MasksToPolygons
from visionprompt.context_learner.processes.segmenters.sam_mapi_decoder import SamMAPIDecoder
from visionprompt.context_learner.types import Image, Priors, State
from visionprompt.context_learner.types.annotations import Annotations
from visionprompt.utils.constants import MAPI_DECODER_PATH, MAPI_ENCODER_PATH

if TYPE_CHECKING:
    from visionprompt.context_learner.processes.mask_processors.mask_processor_base import MaskProcessor
    from visionprompt.context_learner.processes.segmenters.segmenter_base import Segmenter


class PerSamMAPI(Pipeline):
    """This is the PerSam algorithm pipeline using the ModelAPI implementation.

    >>> p = PerSamMAPI()
    >>> p.learn([Image()] * 3, [Priors()] * 3)
    >>> a = p.infer([Image()])
    >>> isinstance(a[0], Annotations)
    True
    """

    def __init__(self) -> None:
        super().__init__()
        # Initialize SAM backbone
        encoder = Model.create_model(MAPI_ENCODER_PATH)
        decoder = Model.create_model(MAPI_DECODER_PATH)
        model = SAMLearnableVisualPrompter(encoder, decoder)

        # Create pipeline processes
        self.encoder: SamMAPIEncoder = SamMAPIEncoder(self._state, model)
        self.mask_processor: MaskProcessor = MasksToPolygons(self._state)
        self.segmenter: Segmenter = SamMAPIDecoder(self._state, model)
        self.mask_processor: MaskProcessor = MasksToPolygons(self._state)

    def learn(self, reference_images: list[Image], reference_priors: list[Priors]) -> None:
        """Perform learning step on the reference images and priors."""
        if len(reference_images) > 1 or len(reference_priors) > 1:
            msg = "PerSamMAPI does not support multiple references"
            raise RuntimeError(msg)
        s: State = self._state

        # Set input in state for convenience
        s.reference_images = reference_images
        s.reference_priors = reference_priors

        # Extract features
        s.reference_features, s.processed_reference_masks = self.encoder(
            s.reference_images,
            s.reference_priors,
        )

    def infer(self, target_images: list[Image]) -> list[Annotations]:
        """Perform inference step on the target images."""
        s: State = self._state

        # Set input in state for convenience
        s.target_images = target_images
        self._state.masks.clear()
        self._state.annotations.clear()
        self._state.used_points.clear()
        self._state.priors.clear()

        s.masks, s.used_points = self.segmenter(target_images, priors=[])
        s.priors = [Priors(points=p) for p in s.used_points]  # Generate a dummy Priors list
        s.annotations = self.mask_processor(s.masks)

        return s.annotations
