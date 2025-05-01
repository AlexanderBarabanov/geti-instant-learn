# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from argparse import Namespace
from pathlib import Path

import requests
from efficientvit.models.efficientvit import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from segment_anything_hq import sam_model_registry as sam_hq_model_registry
from segment_anything_hq.predictor import SamPredictor as SamHQPredictor

from visionprompt.context_learner.pipelines.matcher_pipeline import Matcher
from visionprompt.context_learner.pipelines.perdino_pipeline import PerDino
from visionprompt.context_learner.pipelines.persam_mapi_pipeline import PerSamMAPI
from visionprompt.context_learner.pipelines.persam_pipeline import PerSam
from visionprompt.context_learner.pipelines.pipeline_base import Pipeline
from visionprompt.third_party.PersonalizeSAM.per_segment_anything import (
    SamPredictor,
    sam_model_registry,
)
from visionprompt.utils.constants import DATA_PATH, MODEL_MAP


def load_pipeline(backbone_name: str, pipeline_name: str, args: Namespace) -> Pipeline:
    """Load a pipeline based on the given arguments.

    Args:
        backbone_name: The name of the backbone model.
        pipeline_name: The name of the pipeline.
        args: The arguments to load the model.

    Returns:
        The loaded model.
    """
    if backbone_name not in MODEL_MAP:
        msg = f"Invalid model type: {backbone_name}"
        raise ValueError(msg)

    model_info = MODEL_MAP[backbone_name]
    _check_model_weights(backbone_name)

    registry_name = model_info["registry_name"]
    local_filename = model_info["local_filename"]
    checkpoint_path = DATA_PATH.joinpath(local_filename)

    logging.info(f"Loading segmentation model: {backbone_name} from {checkpoint_path}")

    if backbone_name in {"SAM", "MobileSAM"}:
        sam_model = sam_model_registry[registry_name](checkpoint=str(checkpoint_path)).cuda()
        sam_model.eval()
        sam_model = SamPredictor(sam_model)
    elif backbone_name in {"SAM-HQ", "SAM-HQ-tiny"}:
        sam_model = sam_hq_model_registry[registry_name](checkpoint=str(checkpoint_path)).cuda()
        sam_model.eval()
        sam_model = SamHQPredictor(sam_model)
    elif backbone_name == "EfficientViT-SAM":
        sam_model = create_efficientvit_sam_model(
            name=registry_name,
            weight_url=str(checkpoint_path),
        ).cuda()
        sam_model.eval()
        sam_model = EfficientViTSamPredictor(sam_model)
    else:
        msg = f"Model {backbone_name} not implemented yet"
        raise NotImplementedError(msg)

    logging.info(f"Constructing pipeline: {pipeline_name}")
    # Construct pipeline
    if pipeline_name == "PerSAMModular":
        return PerSam(sam_model, args)
    if pipeline_name == "PerDinoModular":
        return PerDino(sam_model, args)
    if pipeline_name == "MatcherModular":
        return Matcher(sam_model, args)
    if pipeline_name == "PerSAMMAPIModular":
        return PerSamMAPI()
    msg = f"Algorithm {pipeline_name} not implemented yet"
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

    disable_progress = not sys.stderr.isatty()
    progress = Progress(
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        " • ",
        DownloadColumn(),
        " • ",
        TransferSpeedColumn(),
        " • ",
        TimeRemainingColumn(),
        transient=True,
        disable=disable_progress,
    )

    try:  # noqa: PLR1702
        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            print(f"Downloading {target_path.name} ({total_size / (1024 * 1024):.2f} MB) from {url}...")

            with progress:
                task_id = progress.add_task("download", total=total_size, filename=target_path.name)
                with target_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            progress.update(task_id, advance=len(chunk))

            if not disable_progress and total_size > 0:
                progress.update(task_id, completed=total_size)

        print(f"Downloaded model weights successfully to {target_path}")
    except Exception as e:
        # Catch other potential errors (e.g., file writing issues)
        print(f"\nAn unexpected error occurred during download: {e}")
        if target_path.exists():
            try:
                target_path.unlink()
            except OSError as unlink_e:
                print(f"Error removing file {target_path} after error: {unlink_e}")
        raise  # Re-raise
