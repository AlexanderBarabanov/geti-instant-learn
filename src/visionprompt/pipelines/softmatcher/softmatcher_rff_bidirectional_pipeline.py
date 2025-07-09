"""SoftMatcherRFFBiDirectional pipeline."""
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from visionprompt.pipelines import Matcher
from visionprompt.processes.prompt_generators.softmatcher_prompt_generator import SoftmatcherPromptGenerator
from visionprompt.utils.constants import SAMModelName

if TYPE_CHECKING:
    from visionprompt.processes.prompt_generators.prompt_generator_base import PromptGenerator


class SoftMatcherRFFBiDirectional(Matcher):
    """This is the SoftMatcherRFFBiDirectional pipeline.

    This pipeline is the same as the SoftMatcherRFF pipeline, but it uses bidirectional soft matching.

    Examples:
        >>> from visionprompt.pipelines.softmatcher import SoftMatcherRFFBiDirectional
        >>> from visionprompt.types import Image, Priors, Results
        >>> import torch
        >>> import numpy as np
        >>>
        >>> soft_matcher = SoftMatcherRFFBiDirectional()
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
        sam_name: SAMModelName = SAMModelName.SAM,
        num_foreground_points: int = 40,
        num_background_points: int = 2,
        apply_mask_refinement: bool = True,
        skip_points_in_existing_masks: bool = True,
        mask_similarity_threshold: float | None = 0.42,
        precision: str = "bf16",
        compile_models: bool = False,
        verbose: bool = False,
        device: str = "cuda",
        image_size: int | tuple[int, int] | None = None,
    ) -> None:
        """Initialize the SoftMatcherRFFBiDirectional pipeline.

        Args:
            sam_name: The name of the SAM model to use.
            num_foreground_points: The number of foreground points to use.
            num_background_points: The number of background points to use.
            apply_mask_refinement: Whether to apply mask refinement.
            skip_points_in_existing_masks: Whether to skip points in existing masks.
            mask_similarity_threshold: The similarity threshold for the mask.
            precision: The precision to use for the model.
            compile_models: Whether to compile the models.
            verbose: Whether to print verbose output of the model optimization process.
            device: The device to use for the model.
            image_size: The size of the image to use, if None, the image will not be resized.
        """
        super().__init__(
            sam_name=sam_name,
            num_foreground_points=num_foreground_points,
            num_background_points=num_background_points,
            apply_mask_refinement=apply_mask_refinement,
            skip_points_in_existing_masks=skip_points_in_existing_masks,
            mask_similarity_threshold=mask_similarity_threshold,
            precision=precision,
            compile_models=compile_models,
            verbose=verbose,
            device=device,
            image_size=image_size,
        )
        self.prompt_generator: PromptGenerator = SoftmatcherPromptGenerator(
            encoder_input_size=self.encoder.encoder_input_size,
            encoder_patch_size=self.encoder.patch_size,
            encoder_feature_size=self.encoder.feature_size,
            num_background_points=num_background_points,
            num_foreground_points=num_foreground_points,
            approximate_matching=True,
            softmatching_bidirectional=True,
        )
