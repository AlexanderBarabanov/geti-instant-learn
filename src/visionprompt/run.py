"""This module contains functionality for running the pipeline on custom data."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import ast
import pathlib
import re
from logging import getLogger
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from visionprompt.pipelines import Matcher, Pipeline
from visionprompt.processes.visualizations import ExportMaskVisualization
from visionprompt.types import Image, Priors, Text
from visionprompt.utils.constants import IMAGE_EXTENSIONS
from visionprompt.utils.utils import setup_logger

logger = getLogger("Vision Prompt")
setup_logger()


def run_pipeline(
    pipeline: Pipeline,
    target_image_dir: str,
    reference_image_dir: str | None = None,
    reference_prompt_dir: str | None = None,
    reference_points_str: str | None = None,
    reference_text_prompt: str | None = None,
    output_location: str | None = None,
    chunk_size: int = 10,
) -> None:
    """Loads reference data (images and prompts) and target images and runs the pipeline.

    User can provide a reference image directory with or without a reference prompt directory.
    Reference prompt directory can contain multiple directories, each containing the prompts for a single class.
    Prompt files are expected to have the same filename as that of the reference image file.
    Prompts can also be passed in the form of a point coordinate list passed as a string.
        in the form of a list of coordinates for each image:
            Image1, Image2, Image3, ... with ImageX: [class_id:[coord_x, coord_y], ...]
            class_id: -1 is for background, 0 and up are for foreground classes.
            coord_x and coord_y are the x and y coordinates of the point.
            the order of the images is the order of the images in the reference image directory.
        For coordinates of two classes for multiple images:
            "[-1:[x1,y1], 0:[x1, y1], 0:[x2, y2], 1:[x3, y3], ...], [0:[x1, y1], 1:[x3, y3], ...], ..."
        For coordinates of one class for multiple images:
            "[-1:[x1,y1], 0:[x1, y1], 0:[x2, y2],...], [-1:[x1,y1], 0:[x1, y1], 0:[x2, y2],...], ..."

    Example:
        reference_image_dir: "~/data/reference"
        reference_prompt_dir: "~/data/reference/prompts"
            containing class directories with prompt files for a mask e.g. "class_1/<reference_image_filename>.png"
            or point files e.g. "class_1/<reference_image_filename>.txt"

    Example:
        reference_image_dir: "~/data/reference"
        reference_points_str: "[-1:[50, 50], 0:[200, 320], 0:[100, 100], ...], [-1:[100, 100], 1:[100, 100], ...], ..."

    Args:
        pipeline: The pipeline to run.
        target_image_dir: The directory containing all target images.
        reference_image_dir: The directory containing all reference images.
        reference_prompt_dir: The directory containing all reference prompt such as mask files or point files.
        reference_points_str: The string containing all reference points.
        reference_text_prompt: The string containing the text prompt.
        output_location: Custom location for the output data. If not provided, the output will be saved in the
            root of the target image directory.
        chunk_size: The number of images to process in each chunk.
    """
    reference_image_directory = Path(reference_image_dir).expanduser() if reference_image_dir else None
    reference_prompt_directory = Path(reference_prompt_dir).expanduser() if reference_prompt_dir else None
    target_image_directory = Path(target_image_dir).expanduser()
    output_location = (
        Path(output_location).expanduser()
        if output_location
        else pathlib.Path(target_image_directory).parent.parent / "output"
    )

    reference_images, reference_priors, num_classes = parse_reference_data(
        reference_image_directory, reference_prompt_directory, reference_points_str, reference_text_prompt
    )
    target_images, _ = parse_image_files(target_image_directory)

    pipeline.learn(reference_images, reference_priors)
    if reference_image_directory:
        ExportMaskVisualization(str(output_location / "reference"))(
            images=reference_images,
            masks=[p.masks for p in reference_priors],
            names=[image.image_path.name for image in reference_images],
            points=[p.points for p in reference_priors],
            num_classes=num_classes,
        )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    )
    with progress:
        task = progress.add_task("[cyan]Inference", total=len(target_images))
        for i in range(0, len(target_images), chunk_size):
            chunk = target_images[i : i + chunk_size]
            results = pipeline.infer(chunk)
            ExportMaskVisualization(str(output_location / "target"))(
                images=chunk,
                masks=results.masks,
                names=[image.image_path.name for image in chunk],
                points=results.used_points,
                boxes=[p.boxes for p in results.priors],
                num_classes=num_classes,
            )
            progress.update(task, advance=len(chunk))
    logger.info(f"Ouput saved in {output_location}")


def parse_reference_prompt_from_directory(reference_prompt_directory: Path) -> dict[str, Priors]:
    """Parse the reference prompt from a directory.

    This function can parse a directory with class sub-directories, or a single directory with masks.
    The function returns a dictionary of image filenames to Priors.

    Args:
        reference_prompt_directory: The root directory containing all
            reference prompt such as mask files or point files.

    Returns:
        A dictionary of image filenames to Priors.
    """
    priors_map: dict[str, Priors] = {}
    image_suffixes = {f".{ext.strip('*.')}" for ext in IMAGE_EXTENSIONS}
    prompt_globs = [*IMAGE_EXTENSIONS, "*.txt", "*.json"]

    dirs_to_scan = [d for d in reference_prompt_directory.iterdir() if d.is_dir()]
    if not dirs_to_scan:
        # No subdirectories found, so scan the root directory.
        dirs_to_scan = [reference_prompt_directory]

    for class_id, directory in enumerate(dirs_to_scan):
        prompt_files = [p for ext in prompt_globs for p in directory.glob(ext)]

        for prompt_file in prompt_files:
            image_filename = prompt_file.stem
            if image_filename not in priors_map:
                priors_map[image_filename] = Priors()

            priors = priors_map[image_filename]

            if prompt_file.suffix.lower() in image_suffixes:
                pil_image = PILImage.open(prompt_file).convert("L")
                mask_data = np.array(pil_image)
                priors.masks.add(mask=mask_data, class_id=class_id)
            # TODO(Daankrol): Handle other prompt types like .txt or .json  # noqa: TD003

    return priors_map


def parse_reference_str_prompt(reference_points_str: str) -> list[Priors]:
    """Parse the reference prompt string into a list of Priors.

    This function can parse a string with multiple images and multiple classes.
    class_id -1 is for background, 0 and up are for foreground classes.
    coord_x and coord_y are the x and y coordinates of the point.
    the order of the images is the order of the images in the reference image directory.

    The string is expected to be in the form of:
        "[-1:[x1,y1], 0:[x1, y1], 0:[x2, y2], 1:[x3, y3], ...], [0:[x1, y1], 1:[x3, y3], ...], ..."

    Args:
        reference_points_str: The string containing all reference points.

    Returns:
        A list of Priors.
    """
    if not reference_points_str:
        return []

    priors_list = []
    wrapped_str = f"[{reference_points_str}]"
    tuple_str = re.sub(r"(-?\d+):(\[[^\]]+\])", r"(\1, \2)", wrapped_str)
    image_prompts = ast.literal_eval(tuple_str)

    if image_prompts and isinstance(image_prompts[0], tuple):
        image_prompts = [image_prompts]

    for image_prompt_list in image_prompts:
        priors = Priors()
        points_by_class = {}
        for class_id, coord in image_prompt_list:
            points_by_class.setdefault(class_id, []).append(coord)

        background_points = [[p[0], p[1], 0, 0] for p in points_by_class.get(-1, [])]

        foreground_classes = sorted([k for k in points_by_class if k >= 0])

        for class_id in foreground_classes:
            class_points = [[p[0], p[1], 1, 1] for p in points_by_class[class_id]]

            # Add background points to each foreground class
            class_points.extend(background_points)

            if class_points:
                priors.points.add(data=np.array(class_points), class_id=class_id)

        priors_list.append(priors)
    return priors_list


def parse_reference_data(
    reference_image_root: str | None = None,
    reference_prompt_root: str | None = None,
    reference_points_str: str | None = None,
    reference_text_prompt: str | None = None,
) -> tuple[list[Image], list[Priors], int]:
    """Parse the reference data.

    Args:
        reference_image_root: The root directory of the reference data.
        reference_prompt_root: The root directory containing all reference prompt such as mask files or point files.
        reference_points_str: The string containing all reference points.
        reference_text_prompt: The string containing the text prompt.

    Returns:
        A tuple of lists of images and prompts, and the number of classes.
    """
    num_classes = 1
    reference_images: list[Image] = []
    if reference_image_root:
        reference_images, num_classes = parse_image_files(reference_image_root)

    reference_prompts: list[Priors] = []
    if reference_prompt_root is not None:
        if not reference_image_root:
            msg = "reference_image_dir must be provided with reference_prompt_dir"
            raise ValueError(msg)
        prior_map = parse_reference_prompt_from_directory(reference_prompt_root)
        # sort the prompts by the reference image filenames
        reference_prompts = [prior_map[image.image_path.stem] for image in reference_images]

    if reference_points_str is not None:
        reference_prompts.extend(parse_reference_str_prompt(reference_points_str))

    if reference_text_prompt is not None:
        text_prior = Text()
        # split text based on commas and add to text_prior
        split_text = reference_text_prompt.split(",")
        for class_id, text in enumerate(split_text):
            text_prior.add(text, class_id=class_id)
        reference_prompts = [Priors(text=text_prior)]
        num_classes = len(split_text)

    return reference_images, reference_prompts, num_classes


def parse_image_files(root_dir: str) -> list[Image]:
    """Parse the image files from a directory.

    Args:
        root_dir: The root directory of the image files.

    Returns:
        A list of images.
        Number of classes.
    """
    root_dir = pathlib.Path(root_dir)
    class_dirs = [d for d in root_dir.iterdir() if d.is_dir()]

    image_files = []
    if class_dirs:
        # Root directory contains class directories
        for class_dir in class_dirs:
            for ext in IMAGE_EXTENSIONS:
                image_files.extend(class_dir.glob(ext))
    else:
        # Root directory contains images
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(root_dir.glob(ext))

    return [Image(image_path=f) for f in image_files], len(class_dirs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--reference_image_dir", type=str)
    parser.add_argument("--target_image_dir", type=str)
    parser.add_argument("--reference_prompt_dir", type=str)
    parser.add_argument("--points", type=str)
    parser.add_argument("--reference_text_prompt", type=str)
    parser.add_argument("--output_location", type=str)
    args = parser.parse_args()

    run_pipeline(
        pipeline=Matcher(sam_name="MobileSAM"),
        target_image_dir=args.target_image_dir,
        reference_image_dir=args.reference_image_dir,
        reference_prompt_dir=args.reference_prompt_dir,
        reference_points_str=args.points,
        reference_text_prompt=args.reference_text_prompt,
        output_location=args.output_location,
    )
