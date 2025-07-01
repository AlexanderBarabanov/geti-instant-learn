# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import torch

from visionprompt.utils.constants import DATASETS, MODEL_MAP, PIPELINES

# Generate help strings with choices
AVAILABLE_MODELS = ", ".join(MODEL_MAP.keys())
AVAILABLE_PIPELINES = ", ".join(PIPELINES)
AVAILABLE_DATASETS = ", ".join(DATASETS)

HELP_SAM_NAME = (
    f"Backbone segmentation model name or comma-separated list. Use 'all' to run all. Available: [{AVAILABLE_MODELS}]"
)
HELP_PIPELINE = f"Pipeline name or comma-separated list. Use 'all' to run all. Available: [{AVAILABLE_PIPELINES}]"
HELP_DATASET_NAME = f"Dataset name or comma-separated list. Use 'all' to run all. Available: [{AVAILABLE_DATASETS}]"


def get_arguments(arg_list: list[str] | None = None) -> argparse.Namespace:
    """Get arguments.

    Args:
        arg_list: List of arguments

    Returns:
        Arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", type=str, default="INFO", help="Log level")
    parser.add_argument("--sam_name", type=str, default="MobileSAM", help=HELP_SAM_NAME)
    parser.add_argument("--pipeline", type=str, default="MatcherModular", help=HELP_PIPELINE)
    parser.add_argument(
        "--n_shot",
        type=int,
        default=1,
        help="Number of prior images to use as references",
    )
    parser.add_argument("--dataset_name", type=str, default="lvis", help=HELP_DATASET_NAME)
    parser.add_argument(
        "--dataset_filenames",
        type=str,
        nargs="+",
        help="Only perform inference on these files from the dataset. "
        "Filename ambiguity can be solved by including subfolders. "
        "For example: can/01.jpg instead of 01.jpg",
    )
    parser.add_argument("--save", action="store_true", help="Save results to disk")
    parser.add_argument(
        "--apply_mask_refinement",
        action="store_true",
        help="Apply mask refinement",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default=None,
        help="Filter on class name",
    )
    parser.add_argument(
        "--num_grid_cells",
        type=int,
        default=16,
        help="Number of grid cells to use for the grid prompt generator",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Size of the image to use for inference. If not provided, the original size will be used. "
        "If provided, the image will be resized to the given size, maintaining aspect ratio. "
        "Note: images are always resized to 1024x1024 for SAM and to 518x518 for DINO. "
        "This will mainly influence the UI rendering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output data",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="If passed, will save all",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=3,
        help="Number of clusters of features to create, if using the ClusterFeatures module",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.65,
        help="Threshold for segmenting the image",
    )
    parser.add_argument(
        "--mask_similarity_threshold",
        type=float,
        default=0.42,
        help="Threshold for filtering masks based on average similarity",
    )
    parser.add_argument(
        "--skip_points_in_existing_masks",
        type=bool,
        default=True,
        help="Skip foreground points that fall within already generated masks for the same class",
    )
    parser.add_argument(
        "--num_foreground_points",
        type=int,
        default=40,
        help="Maximum number of foreground points to sample, if using the MaxPointFilter module",
    )
    parser.add_argument(
        "--num_background_points",
        type=int,
        default=2,
        help="Number of background points to sample",
    )
    parser.add_argument(
        "--num_priors",
        type=int,
        default=1,
        help="Number of runs to perform, each time using the next image in the dataset as a prior",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="The maximum batch size used during inference.",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        help="The maximum number of batches per class to process. "
        "This can be used to limit the amount images that are processed. "
        "The number of processed images will not exceed num_classes * num_batches * batch_size",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        choices=["float", "float16", "bfloat16"],
        help="The precision to use for the models. Maps to torch.float32, torch.float16, or torch.bfloat16",
    )
    parser.add_argument(
        "--compile_models",
        type=bool,
        default=False,
        help="Whether to compile the models",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to show the inference time of the optimized models",
    )

    args = parser.parse_args(arg_list)

    precision_map = {"float": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    args.precision = precision_map[args.precision]

    return args


def parse_experiment_args(args: argparse.Namespace) -> tuple[list[str], list[str], list[str]]:
    """Parse experiment arguments.

    Args:
        args: Arguments

    Returns:
        tuple containing:
            - datasets_to_run: List of datasets to run
            - pipelines_to_run: List of pipelines to run
            - backbones_to_run: List of backbones to run
    """
    if args.dataset_name == "all":
        valid_datasets = [d for d in DATASETS if d != "all"]
    else:
        datasets_to_run = [d.strip() for d in args.dataset_name.split(",")]
        valid_datasets = [d for d in datasets_to_run if d in DATASETS]

    if args.pipeline == "all":
        valid_pipelines = [p for p in PIPELINES if p != "all"]
    else:
        pipelines_to_run = [p.strip() for p in args.pipeline.split(",")]
        valid_pipelines = [p for p in pipelines_to_run if p in PIPELINES]

    if args.sam_name == "all":
        valid_backbones = [b for b in list(MODEL_MAP.keys()) if b != "all"]
    else:
        backbones_to_run = [b.strip() for b in args.sam_name.split(",")]
        valid_backbones = [b for b in backbones_to_run if b in MODEL_MAP]

    return valid_datasets, valid_pipelines, valid_backbones
