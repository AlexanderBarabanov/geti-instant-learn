# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from visionprompt.context_learner.pipelines import Pipeline
from visionprompt.context_learner.processes.calculators import SegmentationMetrics
from visionprompt.context_learner.processes.visualizations import ExportMaskVisualization
from visionprompt.context_learner.types import Image, Masks, Priors
from visionprompt.datasets import BatchedSingleCategoryIter, Dataset
from visionprompt.utils.args import get_arguments
from visionprompt.utils.data import load_dataset
from visionprompt.utils.models import load_pipeline
from visionprompt.utils.utils import parse_experiment_args


def handle_output_path(output_path: str, overwrite: bool) -> str:
    """Handle output path to avoid overwriting existing data.

    Args:
        output_path: The path to the output data
        overwrite: Whether to overwrite existing data
    Returns:
        The path to the output data
    """
    output_path_obj = Path(output_path)
    if output_path_obj.exists():
        if overwrite:
            shutil.rmtree(output_path_obj)
        else:
            choice = input(
                f"Output path {output_path_obj} already exists. Do you want to overwrite it? (y/n)\n",
            )
            if choice == "y":
                print("Removing existing output data")
                shutil.rmtree(output_path_obj)
            else:
                output_path_obj = Path(f"{output_path}_1")
                print(f"Output path set to {output_path_obj}")
    output_path_obj.mkdir(parents=True, exist_ok=True)
    return str(output_path_obj)


def get_all_instances(images: list[np.ndarray], masks: list[np.ndarray], count: int) -> tuple[list[Image], list[Masks]]:
    """This method returns priors including masks.

    Args:
        images: The list of images of a certain category
        masks: The list of masks of a certain category
        count: The number of image masks to return

    Returns:
        List of images and masks
    """
    # Load all prior images and masks
    prior_images = []
    prior_masks = []
    for i, (image, mask) in enumerate(zip(images, masks, strict=False)):
        prior_images.append(Image(image))
        mask[mask > 1] = 1  # Keep all instances
        mask = mask[:, :, None]
        masks = Masks()
        masks.add(mask)
        prior_masks.append(masks)
        if i >= count - 1:
            break
    return prior_images, prior_masks


def save_priors(prior_images: list[Image], prior_masks: list[Masks], output_path: str) -> None:
    """This method saves the priors to disk.

    Args:
        prior_images: The list of prior images
        prior_masks: The list of prior masks
        output_path: The path to save the priors
    """
    output_path_obj = Path(output_path)
    output_path_obj.mkdir(parents=True, exist_ok=True)
    for i, (image, mask) in enumerate(zip(prior_images, prior_masks, strict=False)):
        mask_np = mask.to_numpy()[0]  # Mask is CHW
        image_np = image.to_numpy()
        mask_np[mask_np > 0] = 255
        mask_np = cv2.cvtColor(mask_np.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(image_np, 0.7, mask_np, 0.3, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path_obj / f"prior_{i}.png"), overlay)


def predict_on_dataset(
    args: argparse.Namespace,
    pipeline: Pipeline,
    priors_dataset: Dataset,
    dataset: Dataset,
    unique_output: str,
    dataset_name: str,
    pipeline_name: str,
    backbone_name: str,
    number_of_priors_tests: int,
    number_of_batches: int | None,
) -> pd.DataFrame:
    """This runs predictions on the dataset and evaluates them.

    Args:
        args: Args from the argparser.
        pipeline: The pipeline to use.
        priors_dataset: The training set that is used for priors
        dataset: The validation set that is processed
        unique_output: Unique output name
        dataset_name: The dataset name
        pipeline_name: The algorithm namen
        backbone_name: The model name
        number_of_priors_tests: The number of priors to try
        number_of_batches: The number of batches per class to process (limited testing)
            pass None to process all data
    Returns:

    """
    unique_output_str = handle_output_path(unique_output, args.overwrite)
    unique_output_path = Path(unique_output_str)
    print(f"Output path: {unique_output_path}")

    visualizer = ExportMaskVisualization(
        state=pipeline._state,
        output_folder=str(unique_output_path),
    )
    metrics_calculators: dict[int, SegmentationMetrics] = {}  # keep metrics per prior

    time_sum = 0
    time_count = 0

    # Setup Rich Progress
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    )

    with progress:
        # Main task for categories (persistent)
        categories_task = progress.add_task(f"[cyan]Processing {dataset_name}", total=dataset.get_category_count())

        # Iterate over all categories in the dataset
        for category_index, batches in enumerate(dataset):
            category_name = dataset.category_index_to_name(category_index)

            priors_cat_index = priors_dataset.category_name_to_index(category_name)

            # Task for priors for the current category (transient)
            priors_iter = BatchedSingleCategoryIter(priors_dataset, args.n_shot, priors_cat_index)
            priors_task = progress.add_task(f"[green]Learn step: {category_name}", total=1, transient=True)

            # Iterate over all priors in the batch (break after number_of_prior_test iterations)
            for priors_batch_index, (priors_images, priors_masks) in enumerate(priors_iter):
                # Add a new metrics calculator if needed
                if priors_batch_index not in metrics_calculators:
                    metrics_calculators[priors_batch_index] = SegmentationMetrics(
                        state=pipeline._state,
                        categories=dataset.get_categories(),
                    )

                # Select priors
                priors_images, priors_masks = get_all_instances(
                    priors_images,
                    priors_masks,
                    args.n_shot,
                )

                # Learn using the priors (currently only use the first masks)
                priors = [Priors(masks=priors_masks[0])]
                pipeline.learn(reference_images=priors_images, reference_priors=priors)
                progress.update(priors_task, advance=1)

                # Save priors
                priors_export_paths = [
                    str(
                        Path("priors")
                        / f"priors_batch_{priors_batch_index}"
                        / category_name
                        / f"prior_{image_index}.png"
                    )
                    for image_index in range(len(priors_images))
                ]
                masks_priors = visualizer.masks_from_priors(
                    pipeline._state.reference_priors,
                )
                visualizer(
                    images=pipeline._state.reference_images,
                    masks=masks_priors,
                    names=priors_export_paths,
                )

                # Task for batches for the current category and prior (transient)
                batches.reset()  # reset batch iterator because it was consumed
                num_batches_to_process = (
                    len(batches) if number_of_batches is None else min(len(batches), number_of_batches + 1)
                )
                batches_task = progress.add_task(
                    f"[magenta]Infer step: {category_name}", total=num_batches_to_process, transient=True
                )

                # Iterate over all batches
                for batch_index, (images, masks) in enumerate(batches):
                    target_images = [Image(image) for image in images]
                    start_time = time.time()
                    pipeline.infer(target_images=target_images)
                    time_sum += time.time() - start_time
                    time_count += len(images)

                    # Generate names for exported files and export them
                    export_paths = [
                        str(
                            Path("predictions")
                            / f"priors_batch_{priors_batch_index}"
                            / category_name
                            / Path(batches.get_image_filename(batch_index, image_index)).name
                        )
                        for image_index in range(len(images))
                    ]
                    export_paths_all_points = [
                        str(
                            Path("predictions_all_points")
                            / f"priors_batch_{priors_batch_index}"
                            / category_name
                            / Path(batches.get_image_filename(batch_index, image_index)).name
                        )
                        for image_index in range(len(images))
                    ]
                    export_paths_gt = [
                        str(
                            Path("ground_truth")
                            / f"priors_batch_{priors_batch_index}"
                            / category_name
                            / Path(batches.get_image_filename(batch_index, image_index)).name
                        )
                        for image_index in range(len(images))
                    ]
                    visualizer(
                        images=pipeline._state.target_images,
                        masks=pipeline._state.masks,
                        names=export_paths,
                    )
                    visualizer(
                        images=pipeline._state.target_images,
                        masks=pipeline._state.masks,
                        names=export_paths_all_points,
                        points=visualizer.points_from_priors(visualizer._state.priors),
                    )
                    gt_masks = visualizer.arrays_to_masks(masks)
                    visualizer(
                        images=pipeline._state.target_images,
                        masks=gt_masks,
                        names=export_paths_gt,
                    )
                    metrics_calculators[priors_batch_index](
                        predictions=pipeline._state.masks,
                        references=gt_masks,
                        mapping={0: category_name},
                    )
                    progress.update(batches_task, advance=1)
                    if number_of_batches is not None and batch_index >= number_of_batches:  # Adjusted condition
                        break
                progress.remove_task(batches_task)

                if priors_batch_index >= number_of_priors_tests - 1:
                    break  # Do not proceed with the next batch of priors
            progress.remove_task(priors_task)
            progress.update(categories_task, advance=1)

    # Construct the output metrics file from the calculated metrics
    all_metrics = None
    for prior_index, calculator in metrics_calculators.items():
        metrics = calculator.get_metrics()
        ln = len(metrics["category"])
        metrics["prior_index"] = [prior_index] * ln
        metrics["inference_time"] = [time_count / time_sum] * ln
        metrics["images_per_category"] = [
            dataset.get_image_count_per_category(cat_name) for cat_name in metrics["category"]
        ]
        metrics["instances_per_category"] = [
            dataset.get_instance_count_per_category(cat_name) for cat_name in metrics["category"]
        ]
        metrics["dataset_name"] = [dataset_name] * ln
        metrics["pipeline_name"] = [pipeline_name] * ln
        metrics["backbone_name"] = [backbone_name] * ln
        if all_metrics is None:
            all_metrics = metrics
        else:
            for key in all_metrics.keys():
                all_metrics[key].extend(metrics[key])

    # Create DataFrame for output
    all_metrics_df = pd.DataFrame.from_dict(all_metrics)

    # Print stats
    print(
        f"\nDataset: {dataset_name}, Backbone: {backbone_name}, Pipeline: {pipeline_name}",
    )
    print("nmIoU:      %.2f" % (100 * all_metrics_df.iou.mean()))
    print("mPrecision: %.2f" % (100 * all_metrics_df.recall.mean()))
    print("mRecall:    %.2f" % (100 * all_metrics_df.precision.mean()))
    print(
        f"Inference time: {all_metrics_df.inference_time.mean():.2f} seconds/target image",
    )
    return all_metrics_df


def main() -> None:
    """Main function to run the experiments.

    This function initializes the arguments, determines which models, datasets, and pipelines to process,
    and then iterates over all combinations to run the predictions and evaluate them.
    """
    # Initialize
    args = get_arguments()
    logging.info(f"Using arguments: {args}\n")
    output_path = Path("~").expanduser() / "outputs"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create data frame with results
    all_results = []
    avg_result_dataframe = None
    datasets_to_run, pipelines_to_run, backbones_to_run = parse_experiment_args(args)
    datasets_str = "-".join(datasets_to_run)
    pipelines_str = "-".join(pipelines_to_run)
    backbones_str = "-".join(backbones_to_run)

    for backbone_name in backbones_to_run:
        for dataset_name in datasets_to_run:
            dataset = load_dataset(dataset_name, whitelist=args.class_name)
            for pidx, pipeline_name in enumerate(pipelines_to_run):
                if pipeline_name == "PerSAMModular" and backbone_name == "EfficientViT-SAM":
                    print(f"Skipping {backbone_name} {pipeline_name} because it is not supported")
                    continue
                if pipeline_name == "PerSAMMAPIModular" and pidx > 0:
                    print("Skipping because PerSAMMAPIModular is independent of the backbone")
                    continue

                pipeline = load_pipeline(backbone_name, pipeline_name, args)

                if args.experiment_name:
                    unique_output_path = (
                        output_path / args.experiment_name / f"{dataset_name}_{backbone_name}_{pipeline_name}"
                    )
                else:
                    unique_output_path = output_path / f"{dataset_name}_{backbone_name}_{pipeline_name}"

                all_metrics_df = predict_on_dataset(
                    args,
                    pipeline,
                    priors_dataset=dataset,
                    dataset=dataset,
                    unique_output=str(unique_output_path),
                    dataset_name=dataset_name,
                    pipeline_name=pipeline_name,
                    backbone_name=backbone_name,
                    number_of_priors_tests=1,
                    number_of_batches=None,
                )
                all_results.append(all_metrics_df)

    all_result_dataframe = pd.concat(all_results, ignore_index=True)
    if args.experiment_name:
        output_path = output_path / args.experiment_name
    all_results_dataframe_filename = (
        output_path / f"models-{backbones_str}_datasets-{datasets_str}_algorithms-{pipelines_str}_all_results.csv"
    )
    all_results_dataframe_filename.parent.mkdir(parents=True, exist_ok=True)
    all_result_dataframe.to_csv(str(all_results_dataframe_filename))
    print(f"Saved all results to: {all_results_dataframe_filename}")

    avg_results_dataframe_filename = (
        output_path / f"models-{backbones_str}_datasets-{datasets_str}_algorithms-{pipelines_str}_avg_results.csv"
    )
    avg_results_dataframe_filename.parent.mkdir(parents=True, exist_ok=True)
    avg_result_dataframe = all_result_dataframe.groupby(
        ["dataset_name", "pipeline_name", "backbone_name"],
    ).mean(numeric_only=True)
    avg_result_dataframe.to_csv(str(avg_results_dataframe_filename))
    print(f"Saved average results to: {avg_results_dataframe_filename}")
    print(f"\n\n Final Average Results:\n {avg_result_dataframe}")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger("dinov2").setLevel(logging.WARNING)
    main()
