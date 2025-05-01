"""This is the main file for the web UI.

It is a Flask application that allows you to run several Visual Prompting pipelines and see the results.
The web UI is served at http://localhost:5050

The web UI can be started by running:
python -m web_ui.app
"""
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import json

import cv2
import numpy as np
import torch
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    stream_with_context,
)

from visionprompt.context_learner.types import Image, Masks, Points, Priors, Similarities
from visionprompt.utils.args import get_arguments
from visionprompt.utils.constants import DATASETS, MODEL_MAP, PIPELINES
from visionprompt.utils.data import load_dataset
from visionprompt.utils.models import load_pipeline

app = Flask(__name__, static_folder="static", template_folder="templates")
default_args = get_arguments([])
print("Loading initial predictor and pipeline...")
current_pipeline_instance = load_pipeline(default_args)
current_pipeline_name = default_args.pipeline


def prepare_image_for_web(image_np):
    """Encodes an image (assumed RGB) as Base64 PNG data URI."""
    if not isinstance(image_np, np.ndarray):
        app.logger.error(
            f"prepare_image_for_web: Input is not a numpy array! Type: {type(image_np)}",
        )
        raise TypeError("Input must be a numpy array")

    # Check shape and convert assumed RGB input to BGR for encoding
    if image_np.ndim == 3 and image_np.shape[2] == 3:
        # Input is 3-channel, assume RGB, convert to BGR
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(
            f"Input array must be RGB (H, W, 3), got shape {image_np.shape}",
        )

    # Ensure dtype is uint8 for imencode
    if image_bgr.dtype != np.uint8:
        # Ensure values are in valid range before converting
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)

    try:
        is_success, buffer = cv2.imencode(".png", image_bgr)
        if not is_success:
            app.logger.error("prepare_image_for_web: cv2.imencode failed!")
            raise ValueError("Could not encode image to PNG")
    except Exception:
        raise

    png_as_text = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{png_as_text}"


def prepare_mask_image_for_web(mask_np):
    """Encodes a single-channel mask as a transparent Base64 PNG data URI."""
    if mask_np.dtype != np.uint8:
        mask_np = mask_np.astype(np.uint8)

    h, w = mask_np.shape
    bgra_mask = np.zeros((h, w, 4), dtype=np.uint8)

    # Set RED color for mask pixels (B=0, G=0, R=255)
    bgra_mask[mask_np > 0, 0] = 0  # Blue
    bgra_mask[mask_np > 0, 1] = 0  # Green
    bgra_mask[mask_np > 0, 2] = 255  # Red
    # Set alpha channel (A=255 for mask, A=0 for background)
    bgra_mask[mask_np > 0, 3] = 255

    is_success, buffer = cv2.imencode(".png", bgra_mask)
    if not is_success:
        raise ValueError("Could not encode mask image to PNG")

    png_as_text = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{png_as_text}"


def prepare_gt_mask_image_for_web(mask_np):
    """Encodes a single-channel mask as a transparent Base64 PNG data URI (Green)."""
    if mask_np.dtype != np.uint8:
        mask_np = mask_np.astype(np.uint8)

    h, w = mask_np.shape
    bgra_mask = np.zeros((h, w, 4), dtype=np.uint8)

    # Set GREEN color for mask pixels (B=0, G=255, R=0)
    bgra_mask[mask_np > 0, 0] = 0  # Blue
    bgra_mask[mask_np > 0, 1] = 255  # Green
    bgra_mask[mask_np > 0, 2] = 0  # Red
    # Set alpha channel (A=255 for mask, A=0 for background)
    bgra_mask[mask_np > 0, 3] = 255

    is_success, buffer = cv2.imencode(".png", bgra_mask)
    if not is_success:
        raise ValueError("Could not encode ground truth mask image to PNG")

    png_as_text = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{png_as_text}"


def process_points_for_web(points_obj: Points):
    """Converts Points object to a JSON-serializable list."""
    processed_points = []
    for class_id, list_of_tensors in points_obj.data.items():
        for tensor in list_of_tensors:
            points_list = tensor.cpu().tolist()
            for point_data in points_list:
                x = point_data[0]
                y = point_data[1]
                score = point_data[2]
                label = int(point_data[3])
                processed_points.append(
                    {
                        "class_id": class_id,
                        "x": x,
                        "y": y,
                        "score": score,
                        "label": label,
                    },
                )
    return processed_points


def process_similarity_maps_for_web(similarities_obj: Similarities):
    """Converts Similarity object maps to JSON-serializable list of data URIs."""
    processed_maps = []
    if not similarities_obj or not hasattr(similarities_obj, "data") or not similarities_obj.data:
        return processed_maps

    for class_id, sim_map_tensor in similarities_obj.data.items():
        try:
            sim_map_tensor_cpu = sim_map_tensor.cpu()

            if sim_map_tensor_cpu.ndim == 3:  # Shape [N, H, W]
                num_instances = sim_map_tensor_cpu.shape[0]
                tensor_to_process = sim_map_tensor_cpu
            elif sim_map_tensor_cpu.ndim == 2:  # Shape [H, W]
                num_instances = 1
                tensor_to_process = sim_map_tensor_cpu.unsqueeze(0)
            else:
                # Try squeezing first in case of extra dims like [1, 1, H, W]
                squeezed_tensor = sim_map_tensor_cpu.squeeze()
                if squeezed_tensor.ndim == 3:  # Shape [N, H, W]
                    num_instances = squeezed_tensor.shape[0]
                    tensor_to_process = squeezed_tensor
                elif squeezed_tensor.ndim == 2:  # Shape [H, W]
                    num_instances = 1
                    tensor_to_process = squeezed_tensor.unsqueeze(0)
                else:
                    app.logger.warning(
                        f"Unexpected sim map shape {sim_map_tensor_cpu.shape} after squeeze for class {class_id}, skipping.",
                    )
                    continue

            for idx in range(num_instances):
                sim_map_np = tensor_to_process[idx].numpy()
                if sim_map_np.size == 0:
                    app.logger.warning(
                        f"Sim map for class {class_id}, instance {idx} is empty, skipping.",
                    )
                    continue

                # Assume sim_map_np is already in [0, 1] range, scale to [0, 255]
                normalized_map = (sim_map_np * 255).astype(np.uint8)
                # Invert the map so high similarity (high value) -> red in JET
                inverted_normalized_map = 255 - normalized_map
                colored_map = cv2.applyColorMap(
                    inverted_normalized_map,
                    cv2.COLORMAP_JET,
                )
                map_uri = prepare_image_for_web(colored_map)

                processed_maps.append(
                    {
                        "point_index": idx,
                        "map_data_uri": map_uri,
                    },
                )
        except Exception as e:
            app.logger.error(
                f"Error processing similarity map for class {class_id}, instance {idx}: {e}",
                exc_info=True,
            )

    return processed_maps


@app.route("/")
def index():
    """Serves the main HTML page."""
    ui_pipelines = [p for p in PIPELINES if p.lower() != "all"]
    ui_datasets = [d for d in DATASETS if d.lower() != "all"]
    return render_template(
        "index.html",
        sam_names=list(MODEL_MAP.keys()),
        pipelines=ui_pipelines,
        datasets=ui_datasets,
    )


@app.route("/api/classes")
def get_classes():
    """Returns a list of unique class names for a given dataset."""
    dataset_name = request.args.get("dataset", "PerSeg")
    try:
        full_dataset = load_dataset(dataset_name)
        unique_classes = full_dataset.get_categories()
        return jsonify({"classes": unique_classes})
    except FileNotFoundError:
        app.logger.error(f"Dataset '{dataset_name}' files not found.", exc_info=True)
        return jsonify({"error": f"Dataset '{dataset_name}' files not found."}), 404
    except Exception as e:
        app.logger.error(
            f"Error getting classes for {dataset_name}: {e}",
            exc_info=True,
        )
        return jsonify({"error": "Could not retrieve class list."}), 500


@app.route("/api/class_info")
def get_class_info():
    """Returns the total number of images for a given class in a dataset."""
    dataset_name = request.args.get("dataset")
    class_name = request.args.get("class_name")

    if not dataset_name or not class_name:
        return jsonify({"error": "Missing dataset or class_name parameter"}), 400

    try:
        full_dataset = load_dataset(dataset_name)
        # Assuming dataset object has a method to get image count per class
        count = full_dataset.get_instance_count_per_category(class_name)
        return jsonify({"total_images": count})

    except Exception as e:
        app.logger.error(
            f"Error getting image count for {class_name} in {dataset_name}: {e}",
            exc_info=True,
        )
        return jsonify({"error": "Could not retrieve image count."}), 500


def _parse_request_and_check_reload(request_data, current_args, current_pipeline_name):
    """Parses relevant values from request data and determines if a pipeline reload is needed.

    Args:
        request_data (dict): The JSON data from the incoming request.
        current_args (SimpleNamespace): The current arguments namespace used by the pipeline.
        current_pipeline_name (str): The name of the currently loaded pipeline.

    Returns:
        tuple: A tuple containing:
            - bool: reload_needed - True if any relevant parameter changed, False otherwise.
            - dict: requested_values - A dictionary containing the values parsed from the request.
    """
    requested_values = {
        "pipeline": request_data.get("pipeline", current_args.pipeline),
        "num_background_points": int(
            request_data.get(
                "num_background_points",
                current_args.num_background_points,
            ),
        ),
        "sam_name": request_data.get("sam_name", current_args.sam_name),
        "similarity_threshold": float(
            request_data.get("similarity_threshold", current_args.similarity_threshold),
        ),
        "mask_similarity_threshold": float(
            request_data.get(
                "mask_similarity_threshold",
                current_args.mask_similarity_threshold,
            ),
        ),
        "skip_points_in_existing_masks": bool(
            request_data.get(
                "skip_points_in_existing_masks",
                getattr(current_args, "skip_points_in_existing_masks", False),
            ),
        ),
        "num_target_images": request_data.get("num_target_images", None),
    }

    reload_needed = False
    if requested_values["pipeline"] != current_pipeline_name:
        reload_needed = True
        print(
            f"Pipeline name changed: {current_pipeline_name} -> {requested_values['pipeline']}",
        )
    if requested_values["sam_name"] != current_args.sam_name:
        reload_needed = True
        print(
            f"Backbone (sam_name) changed: {current_args.sam_name} -> {requested_values['sam_name']}",
        )
    if requested_values["num_background_points"] != current_args.num_background_points:
        reload_needed = True
        print(
            f"Background points changed: {current_args.num_background_points} -> {requested_values['num_background_points']}",
        )
    if requested_values["similarity_threshold"] != current_args.similarity_threshold:
        reload_needed = True
        print(
            f"Similarity threshold changed: {current_args.similarity_threshold} -> {requested_values['similarity_threshold']}",
        )
    if requested_values["mask_similarity_threshold"] != current_args.mask_similarity_threshold:
        reload_needed = True
        print(
            f"Mask similarity threshold changed: {current_args.mask_similarity_threshold} -> {requested_values['mask_similarity_threshold']}",
        )
    current_skip_val = getattr(current_args, "skip_points_in_existing_masks", False)
    if requested_values["skip_points_in_existing_masks"] != current_skip_val:
        reload_needed = True
        print(
            f"Skip covered points changed: {current_skip_val} -> {requested_values['skip_points_in_existing_masks']}",
        )

    return reload_needed, requested_values


def _reload_pipeline_if_needed(reload_needed, requested_values):
    """Reloads the pipeline if necessary based on the requested values.

    Updates the global `current_pipeline_instance`, `current_pipeline_name`, and `default_args`.
    Handles exceptions during reload and reverts `default_args` if loading fails.

    Args:
        reload_needed (bool): Whether a reload is required.
        requested_values (dict): Dictionary of parameters parsed from the request.

    Raises:
        Exception: Propagates exceptions from `load_model` if the reload fails.
    """
    global default_args, current_pipeline_instance, current_pipeline_name

    if reload_needed:
        print("Reloading pipeline due to parameter changes...")
        default_args.pipeline = requested_values["pipeline"]
        default_args.sam_name = requested_values["sam_name"]
        default_args.num_background_points = requested_values["num_background_points"]
        default_args.similarity_threshold = requested_values["similarity_threshold"]
        default_args.mask_similarity_threshold = requested_values["mask_similarity_threshold"]
        default_args.skip_points_in_existing_masks = requested_values["skip_points_in_existing_masks"]
        default_args.num_target_images = requested_values["num_target_images"]

        print(f"Attempting to reload with updated args: {vars(default_args)}")
        reloaded_pipeline_instance = load_pipeline(default_args)

        current_pipeline_instance = reloaded_pipeline_instance
        current_pipeline_name = requested_values["pipeline"]
        print(
            f"Pipeline reloaded successfully to: {current_pipeline_name} with backbone: {default_args.sam_name}",
        )

    return current_pipeline_instance, default_args, current_pipeline_name


def _load_and_prepare_data(
    dataset_name,
    class_name_filter,
    n_shot,
    num_target_images=None,
):
    """Loads the specified dataset, validates parameters, and prepares reference data.

    Args:
        dataset_name (str): Name of the dataset to load.
        class_name_filter (str): The class name to filter data by.
        n_shot (int): The number of reference shots.
        num_target_images (int, optional): The number of target images to limit.

    Returns:
        tuple: A tuple containing:
            - list[Image]: reference_images
            - list[Priors]: reference_priors
            - range: target_indices
            - Dataset: full_dataset instance

    Raises:
        FileNotFoundError: If dataset files are not found.
        KeyError: If the class name is not found in the dataset.
        ValueError: If inputs are invalid (e.g., empty class name, not enough samples).
        Exception: For other dataset loading or processing errors.
    """
    if not class_name_filter:
        raise ValueError("Class name filter cannot be empty")

    try:
        full_dataset = load_dataset(dataset_name)
    except FileNotFoundError:
        app.logger.error(
            f"Dataset '{dataset_name}' files not found during processing.",
            exc_info=True,
        )
        raise  # Propagate the error
    except Exception as e:
        app.logger.error(f"Failed to load dataset '{dataset_name}': {e}", exc_info=True)
        raise ValueError(f"Could not load dataset '{dataset_name}'.") from e

    try:
        image_count_for_category = full_dataset.get_image_count_per_category(
            class_name_filter,
        )
    except KeyError:
        raise KeyError(
            f"Class name '{class_name_filter}' not found in dataset '{dataset_name}'",
        )
    except Exception as e:
        app.logger.error(
            f"Error accessing category data for '{class_name_filter}': {e}",
            exc_info=True,
        )
        raise RuntimeError("Error retrieving category information.") from e

    if image_count_for_category == 0:
        raise ValueError(
            f"No data found for class '{class_name_filter}' in dataset '{dataset_name}'",
        )

    if image_count_for_category <= n_shot:
        raise ValueError(
            f"Not enough samples ({image_count_for_category}) for class '{class_name_filter}' to provide {n_shot} reference shots and at least one target.",
        )

    reference_indices = range(n_shot)
    target_indices = range(n_shot, image_count_for_category)

    reference_images = []
    reference_priors = []

    for i in reference_indices:
        try:
            image_list = full_dataset.get_images_by_category(
                class_name_filter,
                start=i,
                end=i + 1,
            )
            masks_list = full_dataset.get_masks_by_category(
                class_name_filter,
                start=i,
                end=i + 1,
            )
            if not image_list or not masks_list:
                raise ValueError(f"Missing image or mask for reference index {i}")
            image_np = image_list[0]
            mask_np = masks_list[0]
            if mask_np.dtype != np.uint8:
                mask_np = (mask_np > 0).astype(np.uint8)
            ref_image_obj = Image(image_np)
            reference_images.append(ref_image_obj)
            masks = Masks()
            masks.add(mask_np, class_id=0)
            reference_priors.append(Priors(masks=masks))
        except Exception as e:
            app.logger.error(
                f"Error processing reference {i} for class {class_name_filter}: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Error processing reference images for '{class_name_filter}'.",
            ) from e

    # Limit target images based on input
    if num_target_images is not None and num_target_images >= 1:
        num_target_images = int(num_target_images)
        if len(target_indices) > num_target_images:
            app.logger.info(
                f"Limiting target images from {len(target_indices)} to {num_target_images}",
            )
            target_indices = target_indices[:num_target_images]
        elif num_target_images > len(target_indices):
            app.logger.warning(
                f"Requested num_target_images ({num_target_images}) is greater than available ({len(target_indices)}). Using all available.",
            )

    if not target_indices:
        app.logger.warning(
            f"No target images found for class '{class_name_filter}' in dataset '{dataset_name}'",
        )

    return reference_images, reference_priors, target_indices, full_dataset


@app.route("/api/process", methods=["POST"])
def run_processing():
    """Main endpoint to run the visual prompting pipeline and stream results.

    Handles request parsing, pipeline reloading, data preparation, learning,
    and streaming inference results.
    """
    global default_args, current_pipeline_instance, current_pipeline_name

    try:  # noqa: PLR1702
        request_data = request.json
        reload_needed, requested_values = _parse_request_and_check_reload(
            request_data,
            default_args,
            current_pipeline_name,
        )

        try:
            current_pipeline_instance, default_args, current_pipeline_name = _reload_pipeline_if_needed(
                reload_needed,
                requested_values,
            )
        except Exception as e:
            return jsonify({"error": f"Failed to apply settings: {e}"}), 500

        if current_pipeline_instance is None:
            app.logger.error(
                f"Pipeline instance for '{current_pipeline_name}' is not loaded.",
            )
            return jsonify(
                {"error": f"Pipeline '{current_pipeline_name}' could not be loaded."},
            ), 500

        selected_pipeline = current_pipeline_instance
        dataset_name = request_data.get("dataset", "PerSeg")
        class_name_filter = request_data.get("class_name", "can")
        n_shot = int(request_data.get("n_shot", 1))
        try:
            (
                reference_images,
                reference_priors,
                target_indices,
                full_dataset,
            ) = _load_and_prepare_data(
                dataset_name,
                class_name_filter,
                n_shot,
                requested_values["num_target_images"],
            )
        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as e:
            app.logger.error(f"Data preparation error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 400  # Use 400 for client-related errors
        except Exception as e:
            app.logger.error(f"Unexpected data preparation error: {e}", exc_info=True)
            return jsonify({"error": "Failed to prepare data."}), 500

        try:
            selected_pipeline.reset_state()
            selected_pipeline.learn(
                reference_images=reference_images,
                reference_priors=reference_priors,
            )
        except Exception as e:
            app.logger.error(f"Error during pipeline learn step: {e}", exc_info=True)
            return jsonify({"error": "Pipeline learn step failed."}), 500

        # --- Prepare Reference Data for Frontend ---
        prepared_reference_data = []
        try:
            for i, (ref_img, ref_prior) in enumerate(zip(reference_images, reference_priors, strict=False)):
                ref_img_uri = prepare_image_for_web(ref_img.data)
                ref_mask_uri = None
                mask_tensor_list = next(iter(ref_prior.masks.data.values()), [])
                mask_tensor = mask_tensor_list[0]
                mask_np = mask_tensor.cpu().numpy()
                if mask_np.dtype != np.uint8:
                    mask_np = (mask_np > 0).astype(np.uint8)
                # Use the green mask function for reference GT masks
                ref_mask_uri = prepare_gt_mask_image_for_web(mask_np)

                prepared_reference_data.append({
                    "image_data_uri": ref_img_uri,
                    "mask_data_uri": ref_mask_uri,
                })
        except Exception as e:
            app.logger.error(f"Error preparing reference data for frontend: {e}", exc_info=True)
            prepared_reference_data = []
        # --- End Prepare Reference Data ---

        def stream_inference_and_results(
            target_indices_stream,
            full_dataset_stream,
            pipeline_stream,
            class_name_filter_stream,
            reference_data_to_send,  # Add reference data as argument
        ):
            """Generator function to process targets in chunks and yield results."""
            CHUNK_SIZE = 5
            target_indices_list = list(target_indices_stream)
            total_targets = len(target_indices_list)

            try:
                # Send initial message with total count AND reference data
                initial_message = {"total_targets": total_targets, "reference_data": reference_data_to_send}
                yield json.dumps(initial_message) + "\n"
            except Exception as e:
                app.logger.error(f"Error yielding initial message: {e}", exc_info=True)
                yield (json.dumps({"error": "Failed to initiate result streaming."}) + "\n")
                return

            for chunk_start_idx in range(0, total_targets, CHUNK_SIZE):
                chunk_end_idx = min(chunk_start_idx + CHUNK_SIZE, total_targets)
                current_chunk_indices = target_indices_list[chunk_start_idx:chunk_end_idx]

                chunk_target_image_objects = []
                try:
                    for i in current_chunk_indices:
                        image_list = full_dataset_stream.get_images_by_category(
                            class_name_filter_stream,
                            start=i,
                            end=i + 1,
                        )
                        if not image_list:
                            raise ValueError(f"No image found for target index {i}")
                        img_np = image_list[0]
                        chunk_target_image_objects.append(Image(img_np))
                except Exception as e:
                    app.logger.error(
                        f"Error preparing target chunk {chunk_start_idx}-{chunk_end_idx}: {e}",
                        exc_info=True,
                    )
                    yield (
                        json.dumps(
                            {
                                "error": f"Error preparing images for chunk starting at index {target_indices_list[chunk_start_idx]}.",
                            },
                        )
                        + "\n"
                    )
                    continue

                if not chunk_target_image_objects:
                    continue

                try:
                    pipeline_stream.infer(chunk_target_image_objects)
                except Exception as e:
                    app.logger.error(
                        f"Error during pipeline inference for chunk {chunk_start_idx}-{chunk_end_idx}: {e}",
                        exc_info=True,
                    )
                    yield (
                        json.dumps(
                            {
                                "error": f"Pipeline inference failed for chunk starting at index {target_indices_list[chunk_start_idx]}.",
                            },
                        )
                        + "\n"
                    )
                    continue

                results_chunk = []
                try:
                    num_results_in_chunk = len(chunk_target_image_objects)
                    state_masks = pipeline_stream._state.masks
                    state_used_points = pipeline_stream._state.used_points
                    state_priors = pipeline_stream._state.priors
                    state_similarities = getattr(
                        pipeline_stream._state,
                        "similarities",
                        None,
                    )

                    if (
                        len(state_masks) < num_results_in_chunk
                        or len(state_used_points) < num_results_in_chunk
                        or len(state_priors) < num_results_in_chunk
                    ):
                        raise ValueError(
                            f"Pipeline state length mismatch after inference for chunk {chunk_start_idx}-{chunk_end_idx}.",
                        )

                    for j in range(num_results_in_chunk):
                        chunk_idx = j
                        state_idx = -num_results_in_chunk + j
                        original_target_index = current_chunk_indices[chunk_idx]  # Get original index

                        target_img_obj = chunk_target_image_objects[chunk_idx]
                        masks_obj = state_masks[state_idx]
                        used_points_obj = state_used_points[state_idx]
                        prior_obj = state_priors[state_idx]

                        img_data_uri = prepare_image_for_web(target_img_obj.data)

                        processed_mask_data_uris = []
                        instance_counter = 0
                        target_img_h, target_img_w = target_img_obj.data.shape[:2]
                        if masks_obj and hasattr(masks_obj, "data"):
                            for class_id, list_of_tensors in masks_obj.data.items():
                                for mask_tensor in list_of_tensors:
                                    mask_np = mask_tensor.cpu().numpy()
                                    if mask_np.ndim > 2:
                                        mask_np = np.squeeze(mask_np)
                                    if mask_np.dtype == bool or mask_np.dtype == np.bool_:
                                        mask_np = mask_np.astype(np.uint8)
                                    elif mask_np.dtype in [
                                        torch.float32,
                                        torch.float16,
                                    ]:
                                        mask_np = (mask_np > 0.5).astype(np.uint8)
                                    else:
                                        mask_np = (mask_np > 0).astype(np.uint8)

                                    if mask_np.shape != (target_img_h, target_img_w):
                                        mask_np = cv2.resize(
                                            mask_np,
                                            (target_img_w, target_img_h),
                                            interpolation=cv2.INTER_NEAREST,
                                        )
                                    instance_id = f"mask_{instance_counter}"
                                    mask_data_uri = prepare_mask_image_for_web(mask_np)
                                    processed_mask_data_uris.append(
                                        {
                                            "class_id": class_id,
                                            "instance_id": instance_id,
                                            "mask_data_uri": mask_data_uri,
                                        },
                                    )
                                    instance_counter += 1

                        web_used_points = process_points_for_web(used_points_obj)
                        web_prior_points = process_points_for_web(prior_obj.points)

                        web_similarity_maps = []
                        if state_similarities is not None and len(state_similarities) > 0:
                            similarities_for_target = state_similarities[state_idx]
                            web_similarity_maps = process_similarity_maps_for_web(
                                similarities_for_target,
                            )

                        # Fetch corresponding ground truth masks for this chunk
                        chunk_gt_masks = []
                        for i in current_chunk_indices:
                            gt_masks_list = full_dataset_stream.get_masks_by_category(
                                class_name_filter_stream,
                                start=i,
                                end=i + 1,
                            )
                            if not gt_masks_list:
                                app.logger.warning(f"No ground truth mask found for target index {i}")
                                chunk_gt_masks.append(None)  # Add placeholder if not found
                            else:
                                gt_mask_np = gt_masks_list[0]
                                if gt_mask_np.dtype != np.uint8:
                                    gt_mask_np = (gt_mask_np > 0).astype(np.uint8)
                                chunk_gt_masks.append(gt_mask_np)

                        gt_mask_uri = None
                        if chunk_gt_masks[chunk_idx] is not None:
                            gt_mask_uri = prepare_gt_mask_image_for_web(chunk_gt_masks[chunk_idx])

                        results_chunk.append(
                            {
                                "image_data_uri": img_data_uri,
                                "masks": processed_mask_data_uris,
                                "used_points": web_used_points,
                                "prior_points": web_prior_points,
                                "similarity_maps": web_similarity_maps,
                                "gt_mask_uri": gt_mask_uri,
                            },
                        )

                except Exception as e:
                    app.logger.error(
                        f"Error processing results for chunk {chunk_start_idx}-{chunk_end_idx}: {e}",
                        exc_info=True,
                    )
                    yield (
                        json.dumps(
                            {
                                "error": f"Error processing results for chunk starting at index {target_indices_list[chunk_start_idx]}.",
                            },
                        )
                        + "\n"
                    )
                    continue

                if results_chunk:
                    yield json.dumps({"target_results": results_chunk}) + "\n"

        return Response(
            stream_with_context(
                stream_inference_and_results(
                    target_indices,
                    full_dataset,
                    selected_pipeline,
                    class_name_filter,
                    prepared_reference_data,  # Pass prepared data here
                ),
            ),
            mimetype="application/json",
        )

    except FileNotFoundError as e:
        app.logger.error(f"File not found error: {e}", exc_info=True)
        return jsonify({"error": f"Server error: File not found - {e.filename}"}), 500
    except ImportError as e:
        app.logger.error(f"Import error: {e}", exc_info=True)
        return jsonify({"error": f"Server configuration error: {e.name}"}), 500
    except Exception as e:
        app.logger.error(
            f"An unexpected error occurred in run_processing: {e}",
            exc_info=True,
        )
        return jsonify(
            {"error": f"An unexpected server error occurred: {type(e).__name__}"},
        ), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5050)
