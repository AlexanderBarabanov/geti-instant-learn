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
        >>> from visionprompt.types import Image, Priors, Results
        >>> from visionprompt.models.models import load_sam_model
        >>> import torch
        >>> import numpy as np
        >>>
        >>> sam_predictor = load_sam_model(backbone_name="MobileSAM")
        >>> soft_matcher = SoftMatcherSpatialSampling(sam_predictor=sam_predictor)
        >>> # Create mock inputs
        >>> ref_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        >>> target_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        >>> ref_priors = Priors()
        >>> ref_priors.masks.add(torch.ones(30, 30, dtype=torch.bool), class_id=1)
        >>>
        >>> # Run learn and infer
        >>> learn_results = soft_matcher.learn([Image(ref_image)], [ref_priors])
        >>> infer_results = soft_matcher.infer([Image(target_image)])
        >>>
        >>> isinstance(learn_results, Results) and isinstance(infer_results, Results)
        True
        >>> infer_results.masks is not None
        True
        >>> infer_results.annotations is not None
        True
    """

    def __init__(
        self,
        sam_predictor: SamPredictor,
        num_foreground_points: int = 40,
        num_background_points: int = 2,
        apply_mask_refinement: bool = True,
        skip_points_in_existing_masks: bool = True,
        mask_similarity_threshold: float | None = 0.42,
        precision: torch.dtype = torch.bfloat16,
        compile_models: bool = False,
        verbose: bool = False,
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
