"""Pipeline factory module."""
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from visionprompt.utils.constants import PipelineName, SAMModelName

if TYPE_CHECKING:
    from argparse import Namespace

    from visionprompt.pipelines.pipeline_base import Pipeline

logger = logging.getLogger("Vision Prompt")


def load_pipeline(sam_name: SAMModelName, pipeline_name: PipelineName, args: Namespace) -> Pipeline:  # noqa: C901, PLR0911
    """Instantiate and return the requested pipeline.

    Args:
        sam_name: The name of the SAM model.
        pipeline_name: The name of the pipeline.
        args: The arguments to the pipeline.

    Returns:
        The instantiated pipeline.
    """
    # Lazy import to avoid circular dependencies during module import time.
    from visionprompt.pipelines import (
        GroundingDinoSAM,
        Matcher,
        PerDino,
        PerSam,
        PerSamMAPI,
        SoftMatcher,
        SoftMatcherBiDirectional,
        SoftMatcherBiDirectionalSampling,
        SoftMatcherRFF,
        SoftMatcherRFFBiDirectional,
        SoftMatcherRFFBiDirectionalSampling,
        SoftMatcherRFFSampling,
        SoftMatcherSampling,
    )

    logger.info("Constructing pipeline: %s", pipeline_name.value)

    match pipeline_name:
        case PipelineName.PER_SAM:
            return PerSam(
                sam_name=sam_name,
                num_foreground_points=args.num_foreground_points,
                num_background_points=args.num_background_points,
                apply_mask_refinement=args.apply_mask_refinement,
                skip_points_in_existing_masks=args.skip_points_in_existing_masks,
                num_grid_cells=args.num_grid_cells,
                similarity_threshold=args.similarity_threshold,
                mask_similarity_threshold=args.mask_similarity_threshold,
                precision=args.precision,
                compile_models=args.compile_models,
                verbose=args.verbose,
                image_size=args.image_size,
            )
        case PipelineName.PER_DINO:
            return PerDino(
                sam_name=sam_name,
                num_foreground_points=args.num_foreground_points,
                num_background_points=args.num_background_points,
                apply_mask_refinement=args.apply_mask_refinement,
                skip_points_in_existing_masks=args.skip_points_in_existing_masks,
                num_grid_cells=args.num_grid_cells,
                similarity_threshold=args.similarity_threshold,
                mask_similarity_threshold=args.mask_similarity_threshold,
                precision=args.precision,
                compile_models=args.compile_models,
                verbose=args.verbose,
                image_size=args.image_size,
            )
        case PipelineName.MATCHER:
            return Matcher(
                sam_name=sam_name,
                num_foreground_points=args.num_foreground_points,
                num_background_points=args.num_background_points,
                apply_mask_refinement=args.apply_mask_refinement,
                skip_points_in_existing_masks=args.skip_points_in_existing_masks,
                mask_similarity_threshold=args.mask_similarity_threshold,
                precision=args.precision,
                compile_models=args.compile_models,
                verbose=args.verbose,
                image_size=args.image_size,
            )
        case PipelineName.PER_SAM_MAPI:
            return PerSamMAPI()
        case PipelineName.SOFT_MATCHER:
            return SoftMatcher(
                sam_name=sam_name,
                num_foreground_points=args.num_foreground_points,
                num_background_points=args.num_background_points,
                apply_mask_refinement=args.apply_mask_refinement,
                skip_points_in_existing_masks=args.skip_points_in_existing_masks,
                mask_similarity_threshold=args.mask_similarity_threshold,
                precision=args.precision,
                compile_models=args.compile_models,
                verbose=args.verbose,
                image_size=args.image_size,
            )
        case PipelineName.SOFT_MATCHER_RFF:
            return SoftMatcherRFF(
                sam_name=sam_name,
                num_foreground_points=args.num_foreground_points,
                num_background_points=args.num_background_points,
                apply_mask_refinement=args.apply_mask_refinement,
                skip_points_in_existing_masks=args.skip_points_in_existing_masks,
                mask_similarity_threshold=args.mask_similarity_threshold,
                precision=args.precision,
                compile_models=args.compile_models,
                verbose=args.verbose,
                image_size=args.image_size,
            )
        case PipelineName.SOFT_MATCHER_BIDIRECTIONAL:
            return SoftMatcherBiDirectional(
                sam_name=sam_name,
                num_foreground_points=args.num_foreground_points,
                num_background_points=args.num_background_points,
                apply_mask_refinement=args.apply_mask_refinement,
                skip_points_in_existing_masks=args.skip_points_in_existing_masks,
                mask_similarity_threshold=args.mask_similarity_threshold,
                precision=args.precision,
                compile_models=args.compile_models,
                verbose=args.verbose,
                image_size=args.image_size,
            )
        case PipelineName.SOFT_MATCHER_RFF_BIDIRECTIONAL:
            return SoftMatcherRFFBiDirectional(
                sam_name=sam_name,
                num_foreground_points=args.num_foreground_points,
                num_background_points=args.num_background_points,
                apply_mask_refinement=args.apply_mask_refinement,
                skip_points_in_existing_masks=args.skip_points_in_existing_masks,
                mask_similarity_threshold=args.mask_similarity_threshold,
                precision=args.precision,
                compile_models=args.compile_models,
                verbose=args.verbose,
                image_size=args.image_size,
            )
        case PipelineName.SOFT_MATCHER_SAMPLING:
            return SoftMatcherSampling(
                sam_name=sam_name,
                num_foreground_points=args.num_foreground_points,
                num_background_points=args.num_background_points,
                apply_mask_refinement=args.apply_mask_refinement,
                skip_points_in_existing_masks=args.skip_points_in_existing_masks,
                mask_similarity_threshold=args.mask_similarity_threshold,
                precision=args.precision,
                compile_models=args.compile_models,
                verbose=args.verbose,
                image_size=args.image_size,
            )
        case PipelineName.SOFT_MATCHER_RFF_SAMPLING:
            return SoftMatcherRFFSampling(
                sam_name=sam_name,
                num_foreground_points=args.num_foreground_points,
                num_background_points=args.num_background_points,
                apply_mask_refinement=args.apply_mask_refinement,
                skip_points_in_existing_masks=args.skip_points_in_existing_masks,
                mask_similarity_threshold=args.mask_similarity_threshold,
                precision=args.precision,
                compile_models=args.compile_models,
                verbose=args.verbose,
                image_size=args.image_size,
            )
        case PipelineName.SOFT_MATCHER_BIDIRECTIONAL_SAMPLING:
            return SoftMatcherBiDirectionalSampling(
                sam_name=sam_name,
                num_foreground_points=args.num_foreground_points,
                num_background_points=args.num_background_points,
                apply_mask_refinement=args.apply_mask_refinement,
                skip_points_in_existing_masks=args.skip_points_in_existing_masks,
                mask_similarity_threshold=args.mask_similarity_threshold,
                precision=args.precision,
                compile_models=args.compile_models,
                verbose=args.verbose,
                image_size=args.image_size,
            )
        case PipelineName.SOFT_MATCHER_RFF_BIDIRECTIONAL_SAMPLING:
            return SoftMatcherRFFBiDirectionalSampling(
                sam_name=sam_name,
                num_foreground_points=args.num_foreground_points,
                num_background_points=args.num_background_points,
                apply_mask_refinement=args.apply_mask_refinement,
                skip_points_in_existing_masks=args.skip_points_in_existing_masks,
                mask_similarity_threshold=args.mask_similarity_threshold,
                precision=args.precision,
                compile_models=args.compile_models,
                verbose=args.verbose,
                image_size=args.image_size,
            )
        case PipelineName.GROUNDING_DINO_SAM:
            return GroundingDinoSAM(
                sam_name=sam_name,
                apply_mask_refinement=args.apply_mask_refinement,
                device="cuda:0",
                precision=args.precision,
                compile_models=args.compile_models,
                verbose=args.verbose,
                image_size=args.image_size,
            )
        case _:
            msg = f"Algorithm {pipeline_name.value} not implemented yet"
            raise NotImplementedError(msg)
