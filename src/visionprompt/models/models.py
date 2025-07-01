"""Models and pipelines can be constructed using the methods in this file."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from argparse import Namespace
from logging import getLogger
from typing import TYPE_CHECKING

import torch
from efficientvit.models.efficientvit import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from segment_anything_fast import sam_model_fast_registry
from segment_anything_fast.predictor import SamPredictor as SamFastPredictor
from segment_anything_hq import sam_model_registry as sam_hq_model_registry
from segment_anything_hq.predictor import SamPredictor as SamHQPredictor

from visionprompt.models.model_optimizer import optimize_sam_model
from visionprompt.models.per_segment_anything import (
    SamPredictor,
    sam_model_registry,
)
from visionprompt.pipelines import (
    GroundingDinoSAM,
    Matcher,
    PerDino,
    PerSam,
    PerSamMAPI,
    Pipeline,
    SoftMatcher,
    SoftMatcherBiDirectional,
    SoftMatcherBiDirectionalSampling,
    SoftMatcherRFF,
    SoftMatcherRFFBiDirectional,
    SoftMatcherRFFBiDirectionalSampling,
    SoftMatcherRFFSampling,
    SoftMatcherSampling,
)
from visionprompt.utils import download_file
from visionprompt.utils.constants import DATA_PATH, MODEL_MAP

if TYPE_CHECKING:
    from segment_anything_hq.modeling.sam import Sam as SamHQ

    from visionprompt.models.per_segment_anything.modeling.sam import (
        Sam,
    )

logger = getLogger("Vision Prompt")


def load_sam_model(
    backbone_name: str,
    precision: torch.dtype = torch.float32,
    compile_models: bool = False,
    verbose: bool = False,
) -> SamPredictor | SamHQPredictor | SamFastPredictor | EfficientViTSamPredictor:
    """Load and optimize a SAM model.

    Args:
        backbone_name: The name of the backbone model.
        precision: The precision of the model.
        compile_models: Whether to compile the model.
        verbose: Whether to print verbose output.

    Returns:
        The loaded model.
    """
    if backbone_name not in MODEL_MAP:
        msg = f"Invalid model type: {backbone_name}"
        raise ValueError(msg)

    model_info = MODEL_MAP[backbone_name]
    check_model_weights(backbone_name)

    registry_name = model_info["registry_name"]
    local_filename = model_info["local_filename"]
    checkpoint_path = DATA_PATH.joinpath(local_filename)

    logger.info(f"Loading segmentation model: {backbone_name} from {checkpoint_path}")

    if backbone_name in {"SAM", "MobileSAM"}:
        model: Sam = sam_model_registry[registry_name](checkpoint=str(checkpoint_path)).cuda().eval()
        predictor = SamPredictor(model)
    elif backbone_name in {"SAM-HQ", "SAM-HQ-tiny"}:
        model: SamHQ = sam_hq_model_registry[registry_name](checkpoint=str(checkpoint_path)).cuda().eval()
        predictor = SamHQPredictor(model)
    elif backbone_name == "SAM-Fast":
        model = sam_model_fast_registry[registry_name](checkpoint=str(checkpoint_path)).cuda().eval()
        predictor = SamFastPredictor(model)
    elif backbone_name == "EfficientViT-SAM":
        model = (
            create_efficientvit_sam_model(
                name=registry_name,
                weight_url=str(checkpoint_path),
            )
            .cuda()
            .eval()
        )
        predictor = EfficientViTSamPredictor(model)
    else:
        msg = f"Model {backbone_name} not implemented yet"
        raise NotImplementedError(msg)

    return optimize_sam_model(
        sam_predictor=predictor,
        precision=precision,
        compile_models=compile_models,
        verbose=verbose,
    )


def load_pipeline(  # noqa: C901, PLR0911
    backbone_name: str,
    pipeline_name: str,
    args: Namespace,
) -> Pipeline:
    """Load a pipeline based on the given arguments.

    Args:
        backbone_name: The name of the backbone model.
        pipeline_name: The name of the pipeline.
        args: The arguments to load the model.

    Returns:
        The pipeline.
    """
    sam_model = load_sam_model(backbone_name, args.precision, args.compile_models, args.verbose)

    logger.info(f"Constructing pipeline: {pipeline_name}")
    match pipeline_name:
        case "PerSAMModular":
            return PerSam(
                sam_model,
                num_foreground_points=args.num_foreground_points,
                num_background_points=args.num_background_points,
                apply_mask_refinement=args.apply_mask_refinement,
                skip_points_in_existing_masks=args.skip_points_in_existing_masks,
                num_grid_cells=args.num_grid_cells,
                similarity_threshold=args.similarity_threshold,
                mask_similarity_threshold=args.mask_similarity_threshold,
                image_size=args.image_size,
            )
        case "PerDinoModular":
            return PerDino(
                sam_model,
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
        case "MatcherModular":
            return Matcher(
                sam_model,
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
        case "PerSAMMAPIModular":
            # download model weigths if necessary:
            check_model_weights("EfficientViT-SAM")
            return PerSamMAPI()
        case "SoftMatcherModular":
            return SoftMatcher(
                sam_model,
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
        case "SoftMatcherRFFModular":
            return SoftMatcherRFF(
                sam_model,
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
        case "SoftMatcherBiDirectionalModular":
            return SoftMatcherBiDirectional(
                sam_model,
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
        case "SoftMatcherRFFBiDirectionalModular":
            return SoftMatcherRFFBiDirectional(
                sam_model,
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
        case "SoftMatcherSamplingModular":
            return SoftMatcherSampling(
                sam_model,
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
        case "SoftMatcherRFFSamplingModular":
            return SoftMatcherRFFSampling(
                sam_model,
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
        case "SoftMatcherBiDirectionalSamplingModular":
            return SoftMatcherBiDirectionalSampling(
                sam_model,
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
        case "SoftMatcherRFFBiDirectionalSamplingModular":
            return SoftMatcherRFFBiDirectionalSampling(
                sam_model,
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
        case "GroundingDinoSAM":
            return GroundingDinoSAM(sam_model, apply_mask_refinement=args.apply_mask_refinement, device="cuda:0")
    msg = f"Algorithm {pipeline_name} not implemented yet"
    raise NotImplementedError(msg)


def check_model_weights(model_name: str) -> None:
    """Check if model weights exist locally, download if necessary."""
    if model_name not in MODEL_MAP:
        print(f"Warning: Model '{model_name}' not found in MODEL_MAP for weight checking.")
        return

    model_info = MODEL_MAP[model_name]
    local_filename = model_info["local_filename"]
    download_url = model_info["download_url"]

    if not local_filename or not download_url:
        print(f"Warning: Missing 'local_filename' or 'download_url' for {model_name} in MODEL_MAP.")
        return

    target_path = DATA_PATH.joinpath(local_filename)

    if not target_path.exists():
        print(f"Model weights for {model_name} not found at {target_path}, downloading...")
        download_file(download_url, target_path)
