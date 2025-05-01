# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from argparse import Namespace
from pathlib import Path

import requests
from efficientvit.models.efficientvit import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_efficientvit_sam_model

from visionprompt.context_learner.pipelines.matcher_pipeline import Matcher
from visionprompt.context_learner.pipelines.perdino_pipeline import PerDino
from visionprompt.context_learner.pipelines.persam_mapi_pipeline import PerSamMAPI
from visionprompt.context_learner.pipelines.persam_pipeline import PerSam
from visionprompt.context_learner.pipelines.pipeline_base import Pipeline
from visionprompt.third_party.PersonalizeSAM.per_segment_anything import SamPredictor, sam_model_registry
from visionprompt.utils.constants import DATA_PATH, MODEL_MAP


def load_pipeline(args: Namespace) -> Pipeline:
    """Load a pipeline based on the given arguments.

    Args:
        args: The arguments to load the model.

    Returns:
        The loaded model.
    """
    if args.sam_name not in MODEL_MAP:
        msg = f"Invalid model type: {args.sam_name}"
        raise ValueError(msg)

    model_info = MODEL_MAP[args.sam_name]
    _check_model_weights(args.sam_name)

    registry_name = model_info["registry_name"]
    local_filename = model_info["local_filename"]
    checkpoint_path = DATA_PATH.joinpath(local_filename)

    logging.info(f"Loading segmentation model: {args.sam_name} from {checkpoint_path}")

    if args.sam_name in {"SAM", "MobileSAM"}:
        sam_model = sam_model_registry[registry_name](checkpoint=str(checkpoint_path)).cuda()
        sam_model.eval()
        sam_model = SamPredictor(sam_model)
    elif args.sam_name == "EfficientViT-SAM":
        sam_model = create_efficientvit_sam_model(
            name=registry_name,
            weight_url=str(checkpoint_path),
        ).cuda()
        sam_model.eval()
        sam_model = EfficientViTSamPredictor(sam_model)
    else:
        msg = f"Model {args.sam_name} not implemented yet"
        raise NotImplementedError(msg)

    logging.info(f"Constructing pipeline: {args.pipeline}")
    # Construct pipeline
    if args.pipeline == "PerSAMModular":
        return PerSam(sam_model, args)
    if args.pipeline == "PerDinoModular":
        return PerDino(sam_model, args)
    if args.pipeline == "MatcherModular":
        return Matcher(sam_model, args)
    if args.pipeline == "PerSAMMAPIModular":
        return PerSamMAPI()
    msg = f"Algorithm {args.pipeline} not implemented yet"
    raise NotImplementedError(msg)


def _check_model_weights(sam_name: str) -> None:
    """Check if model weights exist locally, download if necessary."""
    if sam_name not in MODEL_MAP:
        print(f"Warning: Model '{sam_name}' not found in MODEL_MAP for weight checking.")
        return

    model_info = MODEL_MAP[sam_name]
    local_filename = model_info["local_filename"]
    download_url = model_info["download_url"]

    if not local_filename or not download_url:
        print(f"Warning: Missing 'local_filename' or 'download_url' for {sam_name} in MODEL_MAP.")
        return

    target_path = DATA_PATH.joinpath(local_filename)

    if not target_path.exists():
        print(f"Model weights for {sam_name} not found at {target_path}, downloading...")
        _download_file(download_url, target_path)


def _download_file(url: str, target_path: Path) -> None:
    """Download a file from a URL to a target path."""
    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with target_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded model weights to {target_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading model weights from {url}: {e}")
        if target_path.exists():
            try:
                target_path.unlink()
            except OSError as unlink_e:
                print(f"Error removing partially downloaded file {target_path}: {unlink_e}")
