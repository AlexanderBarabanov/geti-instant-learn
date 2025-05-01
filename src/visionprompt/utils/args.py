# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

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
    parser.add_argument("--sam_name", type=str, default="SAM", help=HELP_SAM_NAME)
    parser.add_argument("--pipeline", type=str, default="MatcherModular", help=HELP_PIPELINE)
    parser.add_argument(
        "--n_shot",
        type=int,
        default=1,
        help="Number of prior images to use as references",
    )
    parser.add_argument("--dataset_name", type=str, default="lvis", help=HELP_DATASET_NAME)
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
        default=0.45,
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

    return parser.parse_args(arg_list)
