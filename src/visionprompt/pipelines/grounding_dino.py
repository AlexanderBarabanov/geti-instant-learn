"""This is a Pipeline based on grounding Dino with a SAM decoder."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from visionprompt.filters.masks import ClassOverlapMaskFilter, MaskFilter
from visionprompt.models.per_segment_anything import SamPredictor
from visionprompt.pipelines.pipeline_base import Pipeline
from visionprompt.processes.mask_processors import MaskProcessor, MasksToPolygons
from visionprompt.processes.prompt_generators import GroundingDinoBoxGenerator
from visionprompt.processes.segmenters import SamDecoder, Segmenter
from visionprompt.types import Image, Priors, Results, Text
from visionprompt.utils.decorators import track_duration


class GroundingDinoSAM(Pipeline):
    """This Pipeline used GroundingDino to generate boxes for SAM.

    It uses the HuggingFace implementation.
    """

    def __init__(
        self,
        sam_predictor: SamPredictor,
        apply_mask_refinement: bool,
        box_threshold: float = 0.15,
        text_threshold: float = 0.15,
        device: str = "cuda",
    ) -> None:
        """Initialize the pipeline.

        Args:
            sam_predictor: The SAM predictor.
            apply_mask_refinement: Whether to apply mask refinement.
            box_threshold: The box threshold.
            text_threshold: The text threshold.
            device: The device to use.
        """
        super().__init__()
        self.prompt_generator: GroundingDinoBoxGenerator = GroundingDinoBoxGenerator(
            device=device, box_threshold=box_threshold, text_threshold=text_threshold, size="tiny"
        )

        self.segmenter: Segmenter = SamDecoder(
            sam_predictor=sam_predictor,
            apply_mask_refinement=apply_mask_refinement,
            skip_points_in_existing_masks=False,  # not relevant for boxes
        )
        self.mask_processor: MaskProcessor = MasksToPolygons()
        self.class_overlap_mask_filter: MaskFilter = ClassOverlapMaskFilter()
        self.text_priors: Text | None = None

    @track_duration
    def learn(self, reference_images: list[Image], reference_priors: list[Priors]) -> Results:
        """Perform learning step on the reference images and priors."""
        if len(reference_images) != len(reference_priors):
            msg = "Reference images and reference_priors must have same length"
            raise ValueError(msg)
        if not all(p.text is not None for p in reference_priors):
            msg = "reference_priors must have all text types"
            raise ValueError(msg)
        # If all priors are the same use only the first one, else use all.
        if not all(p.text.data for p in reference_priors):
            msg = "Different image-level text priors not supported."
            raise ValueError(msg)
        self.text_priors = reference_priors[0].text

        return Results()

    @track_duration
    def infer(self, target_images: list[Image]) -> Results:
        """Perform inference step on the target images."""
        # Start running the pipeline
        priors = self.prompt_generator(target_images, [self.text_priors] * len(target_images))
        masks, used_points = self.segmenter(target_images, priors)
        annotations = self.mask_processor(masks)

        # write output
        results = Results()
        results.priors = priors
        results.used_points = used_points
        results.masks = masks
        results.annotations = annotations
        return results
