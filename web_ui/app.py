import cv2
import numpy as np
import torch
import base64
from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    Response,
    stream_with_context,
)
import sys
import json

from context_learner.types import Image, Masks, Priors, Points, Similarities
from utils.args import get_arguments
from utils.data import load_dataset
from utils.models import load_model
from utils.constants import MODEL_MAP, PIPELINES, DATASETS


"""
This is the main file for the web UI.
It is a Flask application that allows you to run several Visual Prompting pipelines and see the results.

The web UI is served at http://localhost:5050

The web UI can be started by running:
python -m web_ui.app
"""


app = Flask(__name__, static_folder="static", template_folder="templates")
default_args = get_arguments([])
print("Loading initial predictor and pipeline...")
try:
    current_pipeline_instance = load_model(default_args)
    current_pipeline_name = default_args.pipeline
    print(
        f"Initialized with default pipeline: {current_pipeline_name} and backbone: {default_args.sam_name}"
    )
except ValueError as e:
    print(
        f"ERROR: Could not initialize default pipeline '{default_args.pipeline}' with backbone '{default_args.sam_name}': {e}. Check utils.models.load_model, utils.constants",
        file=sys.stderr,
    )
    current_pipeline_instance = None
    current_pipeline_name = None
except Exception as e:
    print(
        f"FATAL: Unexpected error initializing default pipeline {default_args.pipeline}: {e}",
        file=sys.stderr,
    )
    raise


def prepare_image_for_web(image_np):
    """Encodes an image (assumed RGB) as Base64 PNG data URI."""

    if not isinstance(image_np, np.ndarray):
        app.logger.error(
            f"prepare_image_for_web: Input is not a numpy array! Type: {type(image_np)}"
        )
        raise TypeError("Input must be a numpy array")

    # Check shape and convert assumed RGB input to BGR for encoding
    if image_np.ndim == 3 and image_np.shape[2] == 3:
        # Input is 3-channel, assume RGB, convert to BGR
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(
            f"Input array must be RGB (H, W, 3), got shape {image_np.shape}"
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
    except Exception as e:
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


def process_points_for_web(points_obj: Points):
    """Converts Points object to a JSON-serializable list."""
    processed_points = []
    if not points_obj or not hasattr(points_obj, "data"):
        return processed_points

    for class_id, list_of_tensors in points_obj.data.items():
        for tensor in list_of_tensors:
            points_list = tensor.cpu().tolist()
            for point_data in points_list:
                # Ensure we have at least x, y, score, label
                if len(point_data) >= 4:
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
                        }
                    )
                elif len(point_data) >= 2:
                    x = point_data[0]
                    y = point_data[1]
                    score = point_data[2] if len(point_data) > 2 else None
                    processed_points.append(
                        {
                            "class_id": class_id,
                            "x": x,
                            "y": y,
                            "score": score,
                            "label": 1,
                        }
                    )
    return processed_points


def process_similarity_maps_for_web(similarities_obj: Similarities):
    """Converts Similarity object maps to JSON-serializable list of data URIs."""
    processed_maps = []
    # Revert to checking original structure: hasattr data and data is not None
    if (
        not similarities_obj
        or not hasattr(similarities_obj, "data")
        or not similarities_obj.data
    ):
        return processed_maps

    # Revert to iterating over items, assuming data is a dict {class_id: tensor}
    if not isinstance(similarities_obj.data, dict):
        app.logger.error(
            f"process_similarity_maps_for_web: Expected similarities_obj.data to be a dict, got {type(similarities_obj.data)}"
        )
        return processed_maps

    for class_id, sim_map_tensor in similarities_obj.data.items():
        try:
            if not isinstance(sim_map_tensor, torch.Tensor):
                app.logger.warning(
                    f"Item for class {class_id} is not a tensor, skipping."
                )
                continue

            sim_map_tensor_cpu = sim_map_tensor.cpu()

            # Original logic to handle tensor dimensions and potential multiple instances
            if sim_map_tensor_cpu.ndim == 3:  # Shape [N, H, W]
                num_instances = sim_map_tensor_cpu.shape[0]
                tensor_to_process = sim_map_tensor_cpu
            elif sim_map_tensor_cpu.ndim == 2:  # Shape [H, W]
                num_instances = 1
                tensor_to_process = sim_map_tensor_cpu.unsqueeze(
                    0
                )  # Add batch dim for loop consistency
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
                        f"Unexpected sim map shape {sim_map_tensor_cpu.shape} after squeeze for class {class_id}, skipping."
                    )
                    continue

            for idx in range(num_instances):
                sim_map_np = tensor_to_process[idx].numpy()

                # Check for empty tensor again after indexing
                if sim_map_np.size == 0:
                    app.logger.warning(
                        f"Sim map for class {class_id}, instance {idx} is empty, skipping."
                    )
                    continue

                # Normalize map (handle potential non 0-1 ranges)
                # Assume sim_map_np is already in [0, 1] range, scale to [0, 255]
                normalized_map = (sim_map_np * 255).astype(np.uint8)

                # Invert the map so high similarity (high value) -> red in JET
                inverted_normalized_map = 255 - normalized_map

                # Apply colormap
                colored_map = cv2.applyColorMap(
                    inverted_normalized_map, cv2.COLORMAP_JET
                )
                # Pass to the validated prepare_image_for_web function
                map_uri = prepare_image_for_web(colored_map)

                processed_maps.append(
                    {
                        # "class_id": class_id,
                        "point_index": idx,  # Use instance index as point index
                        "map_data_uri": map_uri,  # Use 'map_data_uri' key for JS
                    }
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
    # Pass model, pipeline, and dataset names to the template
    # Exclude 'all' from lists intended for UI selection
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
        # Get unique class names using the new method
        unique_classes = full_dataset.get_categories()
        return jsonify({"classes": unique_classes})
    except FileNotFoundError:
        app.logger.error(f"Dataset '{dataset_name}' files not found.", exc_info=True)
        return jsonify({"error": f"Dataset '{dataset_name}' files not found."}), 404
    except Exception as e:
        app.logger.error(
            f"Error getting classes for {dataset_name}: {e}", exc_info=True
        )
        return jsonify({"error": "Could not retrieve class list."}), 500


@app.route("/api/process", methods=["POST"])
def run_processing():
    """Runs the pipeline and streams results back chunk by chunk."""

    # --- Setup and Pipeline Loading (mostly unchanged) ---
    try:
        global default_args
        data = request.json
        requested_pipeline_name = data.get("pipeline", default_args.pipeline)
        num_background_points = int(
            data.get("num_background_points", default_args.num_background_points)
        )
        requested_sam_name = data.get("sam_name", default_args.sam_name)

        global current_pipeline_instance, current_pipeline_name

        # --- Determine if pipeline needs reload ---
        reload_needed = False
        if requested_pipeline_name != current_pipeline_name:
            reload_needed = True
            print(
                f"Pipeline name changed: {current_pipeline_name} -> {requested_pipeline_name}"
            )
        if requested_sam_name != default_args.sam_name:
            reload_needed = True
            print(
                f"Backbone (sam_name) changed: {default_args.sam_name} -> {requested_sam_name}"
            )
        if num_background_points != default_args.num_background_points:
            reload_needed = True
            print(
                f"Background points changed: {default_args.num_background_points} -> {num_background_points}"
            )

        if reload_needed:
            print(
                f"Reloading pipeline with args: sam_name={requested_sam_name}, pipeline={requested_pipeline_name}, bg_points={num_background_points}"
            )
            default_args.sam_name = requested_sam_name
            default_args.num_background_points = num_background_points
            default_args.pipeline = requested_pipeline_name
            try:
                current_pipeline_instance = load_model(default_args)
                current_pipeline_name = default_args.pipeline
                print(
                    f"Pipeline reloaded to: {current_pipeline_name} with backbone: {default_args.sam_name}"
                )
            except ValueError as e:
                error_msg = str(e)
                app.logger.error(error_msg)
                # Return error immediately if reload fails
                return jsonify({"error": error_msg}), 400
            except Exception as e:
                error_msg = (
                    f"Failed to switch pipeline to {requested_pipeline_name}: {str(e)}"
                )
                app.logger.error(error_msg, exc_info=True)
                # Return error immediately if reload fails
                return jsonify({"error": error_msg}), 500

        if current_pipeline_instance is None:
            app.logger.error(
                f"Pipeline instance for '{current_pipeline_name}' is not loaded."
            )
            return jsonify(
                {"error": f"Pipeline '{current_pipeline_name}' could not be loaded."}
            ), 500

        selected_pipeline = current_pipeline_instance

        # --- Dataset Loading and Input Prep (mostly unchanged) ---
        dataset_name = data.get("dataset", "PerSeg")
        class_name_filter = data.get("class_name", "can")
        n_shot = int(data.get("n_shot", 1))

        try:
            full_dataset = load_dataset(dataset_name)
        except FileNotFoundError:
            app.logger.error(
                f"Dataset '{dataset_name}' files not found during processing.",
                exc_info=True,
            )
            return jsonify({"error": f"Dataset '{dataset_name}' files not found."}), 404
        except Exception as e:
            app.logger.error(
                f"Failed to load dataset '{dataset_name}': {e}", exc_info=True
            )
            return jsonify({"error": f"Could not load dataset '{dataset_name}'."}), 500

        if not class_name_filter:
            return jsonify({"error": "Class name filter cannot be empty"}), 400

        try:
            # Use class_name_filter (string) instead of category_id for consistency
            image_count_for_category = full_dataset.get_image_count_per_category(
                class_name_filter
            )
        except KeyError:
            return jsonify(
                {
                    "error": f"Class name '{class_name_filter}' not found in dataset '{dataset_name}'"
                }
            ), 404
        except Exception as e:
            app.logger.error(
                f"Error accessing category data for '{class_name_filter}': {e}",
                exc_info=True,
            )
            return jsonify({"error": "Error retrieving category information."}), 500

        if image_count_for_category == 0:
            return jsonify(
                {
                    "error": f"No data found for class '{class_name_filter}' in dataset '{dataset_name}'"
                }
            ), 404

        if image_count_for_category <= n_shot:
            return jsonify(
                {
                    "error": f"Not enough samples ({image_count_for_category}) for class '{class_name_filter}' to provide {n_shot} reference shots and at least one target."
                }
            ), 400

        reference_indices = range(n_shot)
        target_indices = range(n_shot, image_count_for_category)

        reference_images = []
        reference_priors = []
        selected_pipeline.reset_state()

        # --- Process References (unchanged error handling needed) ---
        for i in reference_indices:
            try:
                image_list = full_dataset.get_images_by_category(
                    class_name_filter, start=i, end=i + 1
                )
                masks_list = full_dataset.get_masks_by_category(
                    class_name_filter, start=i, end=i + 1
                )
                if not image_list or not masks_list:
                    raise ValueError("Missing image or mask")
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
                return jsonify(
                    {
                        "error": f"Error processing reference images for '{class_name_filter}'."
                    }
                ), 500

        # --- Run Pipeline Inference (single call) ---
        try:
            selected_pipeline.learn(
                reference_images=reference_images, reference_priors=reference_priors
            )
        except Exception as e:
            app.logger.error(f"Error during pipeline execution: {e}", exc_info=True)
            # Return error immediately if pipeline fails
            return jsonify({"error": "Pipeline execution failed."}), 500

        # --- Stream Inference and Result Processing ---
        def stream_inference_and_results():
            nonlocal target_indices, full_dataset, selected_pipeline, class_name_filter  # Ensure access
            CHUNK_SIZE = 5
            target_indices_list = list(
                target_indices
            )  # Convert range to list for slicing
            total_targets = len(target_indices_list)

            # --- Yield total count first ---
            try:
                yield json.dumps({"total_targets": total_targets}) + "\n"
            except Exception as e:
                app.logger.error(f"Error yielding total count: {e}", exc_info=True)
                # If we can't even yield the total, something is wrong
                yield (
                    json.dumps({"error": "Failed to initiate result streaming."}) + "\n"
                )
                return

            # Process targets in chunks
            for chunk_start_idx in range(0, total_targets, CHUNK_SIZE):
                chunk_end_idx = min(chunk_start_idx + CHUNK_SIZE, total_targets)
                current_chunk_indices = target_indices_list[
                    chunk_start_idx:chunk_end_idx
                ]

                # Prepare target images for the current chunk
                chunk_target_image_objects = []
                try:
                    for i in current_chunk_indices:
                        image_list = full_dataset.get_images_by_category(
                            class_name_filter, start=i, end=i + 1
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
                                "error": f"Error preparing images for chunk starting at index {target_indices_list[chunk_start_idx]}."
                            }
                        )
                        + "\n"
                    )
                    continue  # Skip to next chunk

                if not chunk_target_image_objects:
                    continue  # Skip if chunk is empty for some reason

                # --- Run Inference for the current chunk ---
                try:
                    selected_pipeline.infer(chunk_target_image_objects)
                except Exception as e:
                    app.logger.error(
                        f"Error during pipeline inference for chunk {chunk_start_idx}-{chunk_end_idx}: {e}",
                        exc_info=True,
                    )
                    yield (
                        json.dumps(
                            {
                                "error": f"Pipeline inference failed for chunk starting at index {target_indices_list[chunk_start_idx]}."
                            }
                        )
                        + "\n"
                    )
                    continue  # Skip to next chunk

                # --- Process and Yield Results for the current chunk ---
                results_chunk = []
                try:
                    # Assuming infer updates state, and we need results for the *last* `len(chunk_target_image_objects)` items
                    num_results_in_chunk = len(chunk_target_image_objects)
                    state_masks = selected_pipeline._state.masks
                    state_used_points = selected_pipeline._state.used_points
                    state_priors = selected_pipeline._state.priors
                    state_similarities = getattr(
                        selected_pipeline._state, "similarities", None
                    )

                    # Validate state length - simple check assumes state reflects *only* last inference
                    # More complex logic needed if state accumulates across infer calls.
                    if (
                        len(state_masks) < num_results_in_chunk
                        or len(state_used_points) < num_results_in_chunk
                        or len(state_priors) < num_results_in_chunk
                    ):
                        raise ValueError(
                            f"Pipeline state length mismatch after inference for chunk {chunk_start_idx}-{chunk_end_idx}. Expected {num_results_in_chunk}, State: masks={len(state_masks)}, points={len(state_used_points)}"
                        )

                    # Process the results for this chunk (assuming they are the last N in state)
                    for j in range(num_results_in_chunk):
                        # Index into the chunk objects and the *end* of the state lists
                        chunk_idx = j
                        state_idx = (
                            -num_results_in_chunk + j
                        )  # Index from the end of state lists

                        target_img_obj = chunk_target_image_objects[chunk_idx]
                        masks_obj = state_masks[state_idx]
                        used_points_obj = state_used_points[state_idx]
                        prior_obj = state_priors[state_idx]

                        img_data_uri = prepare_image_for_web(target_img_obj.data)

                        # Process masks (same logic as before)
                        processed_mask_data_uris = []
                        instance_counter = 0
                        target_img_h, target_img_w = target_img_obj.data.shape[:2]
                        if masks_obj and hasattr(masks_obj, "data"):
                            for class_id, list_of_tensors in masks_obj.data.items():
                                for mask_tensor in list_of_tensors:
                                    mask_np = mask_tensor.cpu().numpy()
                                    if mask_np.ndim > 2:
                                        mask_np = np.squeeze(mask_np)
                                    if (
                                        mask_np.dtype == bool
                                        or mask_np.dtype == np.bool_
                                    ):
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
                                        }
                                    )
                                    instance_counter += 1

                        web_used_points = process_points_for_web(used_points_obj)
                        web_prior_points = process_points_for_web(prior_obj.points)

                        web_similarity_maps = []
                        if (
                            state_similarities is not None
                            and len(state_similarities) > 0
                        ):
                            similarities_for_target = state_similarities[state_idx]
                            web_similarity_maps = process_similarity_maps_for_web(
                                similarities_for_target
                            )

                        results_chunk.append(
                            {
                                "image_data_uri": img_data_uri,
                                "masks": processed_mask_data_uris,
                                "used_points": web_used_points,
                                "prior_points": web_prior_points,
                                "similarity_maps": web_similarity_maps,
                            }
                        )

                except Exception as e:
                    app.logger.error(
                        f"Error processing results for chunk {chunk_start_idx}-{chunk_end_idx}: {e}",
                        exc_info=True,
                    )
                    yield (
                        json.dumps(
                            {
                                "error": f"Error processing results for chunk starting at index {target_indices_list[chunk_start_idx]}."
                            }
                        )
                        + "\n"
                    )
                    continue  # Skip to next chunk

                # Yield the processed chunk
                if results_chunk:
                    yield json.dumps({"target_results": results_chunk}) + "\n"

        # Return the streaming response using the new generator
        return Response(
            stream_with_context(stream_inference_and_results()),
            mimetype="application/json",
        )

    except FileNotFoundError as e:  # Catch errors before streaming starts
        app.logger.error(f"File not found error before streaming: {e}", exc_info=True)
        return jsonify({"error": f"Server error: File not found - {e.filename}"}), 500
    except ImportError as e:
        app.logger.error(f"Import error before streaming: {e}", exc_info=True)
        return jsonify({"error": f"Server configuration error: {e.name}"}), 500
    except Exception as e:
        app.logger.error(
            f"An unexpected error occurred before streaming: {e}", exc_info=True
        )
        # This catches errors during setup, loading, learn phase etc.
        return jsonify(
            {"error": f"An unexpected error occurred: {type(e).__name__}"}
        ), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5051)
