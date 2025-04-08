# ruff: noqa: E402
import warnings
from typing import Tuple, List, Dict

import sys

from context_learner.pipelines.pipeline_base import Pipeline
from context_learner.processes.calculators.segmentation_metrics import SegmentationMetrics
from context_learner.processes.visualizations.export_visualization import ExportMaskVisualization
from context_learner.types.image import Image
from context_learner.types.masks import Masks
from context_learner.types.priors import Priors
from datasets.dataset_base import Dataset, DatasetIter
from datasets.dataset_iterators import BatchedCategoryIter, BatchedSingleCategoryIter
from utils.args import get_arguments

warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import os
import shutil
import time
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from algorithms import PerDinoPredictor, PerSamPredictor
from utils.constants import MODEL_MAP, DATASETS, ALGORITHMS
from P2SAM.eval_utils import AverageMeter, intersectionAndUnion
from model_api.models.visual_prompting import Prompt, SAMLearnableVisualPrompter
from utils.models import load_model
from utils.data import load_dataset
import logging


def handle_output_path(output_path: str, overwrite: bool):
    if os.path.exists(output_path):
        if overwrite:
            shutil.rmtree(output_path)
        else:
            choice = input(f"Output path {output_path} already exists. Do you want to overwrite it? (y/n)\n")
            if choice == "y":
                print("Removing existing output data")
                shutil.rmtree(output_path)
            else:
                output_path += "_1"
                print(f"Output path set to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    return output_path


def get_all_instances(images: List[np.ndarray], masks: List[np.ndarray], count: int):
    """
    This method returns priors including masks. It only all first instances.

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
    for i, (image, mask) in enumerate(zip(images, masks)):
        prior_images.append(Image(image))
        mask[mask > 1] = 1  # Keep all instances
        mask = mask[:, :, None]
        masks = Masks()
        masks.add(mask)
        prior_masks.append(masks)
        if i >= count - 1:
            break
    return prior_images, prior_masks


def save_priors(prior_images: List[Image], prior_masks: List[Masks], output_path):
    # save prior images and masks to disk (merging instances)
    os.makedirs(output_path, exist_ok=True)
    for i, (image, mask) in enumerate(zip(prior_images, prior_masks)):
        mask_np = mask.to_numpy()[0]  # Mask is CHW
        image_np = image.to_numpy()
        mask_np[mask_np > 0] = 255
        mask_np = cv2.cvtColor(mask_np.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(image_np, 0.7, mask_np, 0.3, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_path, f"prior_{i}.png"), overlay)


def predict_on_dataset2(
    args: argparse.Namespace,
    pipeline: Pipeline,
    priors_dataset: Dataset,
    dataset: Dataset,
    unique_output: str,
    dataset_name: str,
    pipeline_name: str,
    backbone_name: str,
    number_of_priors_tests: int,
):
    """
    This runs predictions on the dataset and evaluates them
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

    Returns:

    """
    unique_output = handle_output_path(unique_output, args.overwrite)
    print(f"Output path: {unique_output}")

    visualizer = ExportMaskVisualization(state=pipeline._state, output_folder=unique_output)
    metrics_calculators: Dict[int, SegmentationMetrics] = dict()  # keep metrics per prior

    time_sum = 0
    time_count = 0
    # Iterate over all categories in the dataset
    for category_index, batches in tqdm(enumerate(dataset), total=dataset.get_category_count(), desc="categories"):
        category_name = dataset.category_index_to_name(category_index)

        # Find the index of the current category in the priors_dataset.
        priors_cat_index = priors_dataset.category_name_to_index(category_name)

        # Iterate over all priors in the batch (break after number_of_prior_test iterations)
        for priors_batch_index, (priors_images, priors_masks) in tqdm(enumerate(BatchedSingleCategoryIter(priors_dataset, args.n_shot, priors_cat_index)), desc=f"priors {category_name}"):

            # Add a new metrics calculator if needed
            if priors_batch_index not in metrics_calculators.keys():
                metrics_calculators[priors_batch_index] = SegmentationMetrics(state=pipeline._state, categories=dataset.get_categories())

            # Select  priors
            priors_images, priors_masks = get_all_instances(priors_images, priors_masks, args.n_shot)

            # Learn using the priors (currently only use the first masks)
            priors = [Priors(masks=priors_masks[0])]
            pipeline.learn(reference_images=priors_images, reference_priors=priors)

            # Save priors
            priors_export_paths = [os.path.join("priors", f"priors_batch_{priors_batch_index}", category_name, f"prior_{image_index}.png") for image_index in range(len(priors_images))]
            masks_priors = visualizer.priors_to_masks(pipeline._state.reference_priors)
            visualizer(images=pipeline._state.reference_images, masks=masks_priors, names=priors_export_paths)

            # Iterate over all batches
            batches.reset()  # reset batch iterator because it was consumed
            for batch_index, (images, masks) in enumerate(tqdm(batches, total=len(batches), desc=f"batches {category_name}")):
                target_images = [Image(image) for image in images]
                start_time = time.time()
                pipeline.infer(target_images=target_images)
                time_sum += (time.time() - start_time)
                time_count += len(images)

                # Generate names for exported files and export them
                export_paths = [os.path.join("predictions", f"priors_batch_{priors_batch_index}", category_name, os.path.basename(batches.get_image_filename(batch_index, image_index))) for image_index in range(len(images))]
                export_paths_all_points = [os.path.join("predictions_all_points", f"priors_batch_{priors_batch_index}", category_name, os.path.basename(batches.get_image_filename(batch_index, image_index))) for image_index in range(len(images))]
                export_paths_gt = [os.path.join("ground_truth", f"priors_batch_{priors_batch_index}",category_name, os.path.basename(batches.get_image_filename(batch_index, image_index))) for image_index in range(len(images))]
                visualizer(images=pipeline._state.target_images, masks=pipeline._state.masks, names=export_paths)
                visualizer(images=pipeline._state.target_images, masks=pipeline._state.masks, names=export_paths_all_points, points=visualizer.priors_to_points(visualizer._state.priors))
                gt_masks = visualizer.arrays_to_masks(masks)
                visualizer(images=pipeline._state.target_images, masks=gt_masks, names=export_paths_gt)
                metrics_calculators[priors_batch_index](predictions=pipeline._state.masks, references=gt_masks, mapping={0: category_name})

            if priors_batch_index >= number_of_priors_tests - 1:
                break  # Do not proceed with the next batch of priors

    # Construct the output metrics file from the calculated metrics
    all_metrics = None
    for prior_index, calculator in metrics_calculators.items():
        metrics = calculator.get_metrics()
        ln = len(metrics["category"])
        metrics["prior_index"] = [prior_index] * ln
        metrics["inference_time"] = [time_count / time_sum] * ln
        metrics["images_per_category"] = [dataset.get_image_count_per_category(cat_name) for cat_name in metrics["category"]]
        metrics["instances_per_category"] = [dataset.get_instance_count_per_category(cat_name) for cat_name in metrics["category"]]
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
    print(f"\nDataset: {dataset_name}, Backbone: {backbone_name}, Pipeline: {pipeline_name}")
    print("nmIoU:      %.2f" % (100 * all_metrics_df.iou.mean()))
    print("mPrecision: %.2f" % (100 * all_metrics_df.recall.mean()))
    print("mRecall:    %.2f" % (100 * all_metrics_df.precision.mean()))
    print("Inference time: %.2f seconds/target image" % all_metrics_df.inference_time.mean())
    return all_metrics_df


def main2():
    # Initialize
    args = get_arguments()
    output_path = os.path.expanduser(os.path.join("~", "outputs"))
    os.makedirs(output_path, exist_ok=True)

    # Determine which models, datasets and algorithms to process
    datasets_to_run = DATASETS if args.dataset_name == "all" else [args.dataset_name]
    pipelines_to_run = ALGORITHMS if args.algo == "all" else [args.algo]
    backbones_to_run = MODEL_MAP.keys() if args.sam_name == "all" else [args.sam_name]

    # Create data frame with results
    all_results = []
    avg_result_dataframe = None
    datasets_str = "-".join(datasets_to_run)
    pipelines_str = "-".join(pipelines_to_run)
    backbones_str = "-".join(backbones_to_run)

    # Process each combination
    for backbone_name in backbones_to_run:
        for dataset_name in datasets_to_run:
            dataset = load_dataset(dataset_name, whitelist=args.class_name)
            for pipeline_name in pipelines_to_run:
                pipeline = load_model(backbone_name, pipeline_name)
                unique_output = os.path.join(output_path, f"{dataset_name}_{backbone_name}_{pipeline_name}")
                all_metrics_df = predict_on_dataset2(args, pipeline,
                                                     priors_dataset=dataset,
                                                      dataset=dataset,
                                                      unique_output=unique_output,
                                                      dataset_name=dataset_name,
                                                      pipeline_name=pipeline_name,
                                                      backbone_name=backbone_name,
                                                      number_of_priors_tests=1)
                all_results.append(all_metrics_df)

                # Save and print (intermediate) results
                all_result_dataframe = pd.concat(all_results, ignore_index=True)
                all_results_dataframe_filename = os.path.join(output_path, f"models-{backbones_str}_datasets-{datasets_str}_algorithms-{pipelines_str}_all_results.csv")
                all_result_dataframe.to_csv(all_results_dataframe_filename)

                avg_results_dataframe_filename = os.path.join(output_path, f"models-{backbones_str}_datasets-{datasets_str}_algorithms-{pipelines_str}_avg_results.csv")
                avg_result_dataframe = all_result_dataframe.groupby(["dataset_name", "pipeline_name", "backbone_name"]).mean(numeric_only=True)
                avg_result_dataframe.to_csv(avg_results_dataframe_filename)

    print(f"\n\n Results: {avg_result_dataframe}")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main2()
