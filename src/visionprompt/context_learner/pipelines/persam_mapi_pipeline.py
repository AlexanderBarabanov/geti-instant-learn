# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import TYPE_CHECKING

from model_api.models.model import Model
from model_api.models.visual_prompting import SAMLearnableVisualPrompter

from visionprompt.context_learner.pipelines.pipeline_base import Pipeline
from visionprompt.context_learner.processes.encoders.sam_mapi_encoder import SamMAPIEncoder
from visionprompt.context_learner.processes.mask_processors.mask_to_polygon import MasksToPolygons
from visionprompt.context_learner.processes.segmenters.sam_mapi_decoder import SamMAPIDecoder
from visionprompt.context_learner.types import Image, Priors, Results
from visionprompt.utils.constants import MAPI_DECODER_PATH, MAPI_ENCODER_PATH
from visionprompt.utils.decorators import track_duration

if TYPE_CHECKING:
    from visionprompt.context_learner.processes.mask_processors.mask_processor_base import MaskProcessor
    from visionprompt.context_learner.processes.segmenters.segmenter_base import Segmenter


class PerSamMAPI(Pipeline):
    """This is the PerSam algorithm pipeline using the ModelAPI implementation."""

    def __init__(self) -> None:
        super().__init__()
        # Initialize SAM backbone
        encoder = Model.create_model(MAPI_ENCODER_PATH)
        decoder = Model.create_model(MAPI_DECODER_PATH)
        model = SAMLearnableVisualPrompter(encoder, decoder)

        # Create pipeline processes
        self.encoder: SamMAPIEncoder = SamMAPIEncoder(model)
        self.mask_processor: MaskProcessor = MasksToPolygons()
        self.segmenter: Segmenter = SamMAPIDecoder(model)
        self.mask_processor: MaskProcessor = MasksToPolygons()
        self.reference_features = None

    @track_duration
    def learn(self, reference_images: list[Image], reference_priors: list[Priors]) -> Results:
        """Perform learning step on the reference images and priors."""
        if len(reference_images) > 1 or len(reference_priors) > 1:
            msg = "PerSamMAPI does not support multiple references"
            raise RuntimeError(msg)

        # Extract features
        self.reference_features, _ = self.encoder(
            reference_images,
            reference_priors,
        )
        return Results()

    @track_duration
    def infer(self, target_images: list[Image]) -> Results:
        """Perform inference step on the target images."""
        masks, _ = self.segmenter(target_images, self.reference_features)
        annotations = self.mask_processor(masks)

        # write output
        results = Results()
        results.masks = masks
        results.annotations = annotations
        return results
