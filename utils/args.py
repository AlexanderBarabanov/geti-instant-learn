# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
from utils.constants import MODEL_MAP, DATASETS, PIPELINES


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--sam_name", type=str, default="SAM", choices=MODEL_MAP.keys())
    parser.add_argument(
        "--pipeline", type=str, default="MatcherModular", choices=PIPELINES
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=1,
        help="Number of prior images to use as references",
    )
    parser.add_argument("--dataset_name", type=str, default="lvis", choices=DATASETS)
    parser.add_argument("--save", action="store_true", help="Save results to disk")
    parser.add_argument(
        "--apply_mask_refinement", action="store_true", help="Apply mask refinement"
    )
    parser.add_argument(
        "--class_name", type=str, default=None, help="Filter on class name"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output data"
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

    args = parser.parse_args(arg_list)
    return args
