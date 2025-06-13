"""SoftMatcherSpatialSampling pipeline."""
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import torch

from visionprompt.models.per_segment_anything import SamPredictor
from visionprompt.pipelines import Matcher
from visionprompt.processes.prompt_generators.softmatcher_prompt_generator import (
    SoftmatcherPromptGenerator,
)

if TYPE_CHECKING:
    from visionprompt.processes.prompt_generators.prompt_generator_base import (
        PromptGenerator,
    )


class SoftMatcherSpatialSampling(Matcher):
    """This is the SoftMatcherSpatialSampling pipeline.

    This pipeline is the same as the SoftMatcher pipeline, but it uses spatial sampling to generate prompts.

    Examples:
        >>> from visionprompt.pipelines.softmatcher import SoftMatcherSpatialSampling
        >>> from visionprompt.types import Image, Priors
        >>>
        >>> soft_matcher = SoftMatcherSpatialSampling(...)
        >>> soft_matcher.learn([Image()], [Priors()])
        >>> results = soft_matcher.infer([Image()])
    """

    def __init__(
        self,
        sam_predictor: SamPredictor,
        num_foreground_points: int,
        num_background_points: int,
        apply_mask_refinement: bool,
        skip_points_in_existing_masks: bool,
        mask_similarity_threshold: float | None,
        precision: torch.dtype,
        compile_models: bool,
        verbose: bool,
        image_size: int | tuple[int, int] | None = None,
    ) -> None:
        """Initialize the SoftMatcherSpatialSampling pipeline."""
        super().__init__(
            sam_predictor=sam_predictor,
            num_foreground_points=num_foreground_points,
            num_background_points=num_background_points,
            apply_mask_refinement=apply_mask_refinement,
            skip_points_in_existing_masks=skip_points_in_existing_masks,
            mask_similarity_threshold=mask_similarity_threshold,
            precision=precision,
            compile_models=compile_models,
            verbose=verbose,
            image_size=image_size,
        )
        self.prompt_generator: PromptGenerator = SoftmatcherPromptGenerator(
            encoder_input_size=self.encoder.encoder_input_size,
            encoder_patch_size=self.encoder.patch_size,
            encoder_feature_size=self.encoder.feature_size,
            num_background_points=num_background_points,
            num_foreground_points=num_foreground_points,
            use_spatial_sampling=True,
            approximate_matching=False,
            softmatching_bidirectional=False,
        )
