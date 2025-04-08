# app.py
import os
import cv2
import numpy as np
import torch
import base64
import warnings  # Import the warnings module
from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
from werkzeug.utils import secure_filename

# Filter the specific UserWarning from PersonalizeSAM
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="PersonalizeSAM.per_segment_anything.modeling.tiny_vit_sam",
)

# Assume your project structure allows these imports
from context_learner.pipelines.matcher_pipeline import Matcher
from context_learner.pipelines.persam_pipeline import PerSam
from context_learner.types import Image, Masks, Priors, Points
from utils.data import load_dataset

# --- Configuration ---
STATIC_FOLDER = "static"
# TEMP_IMAGE_FOLDER = os.path.join(STATIC_FOLDER, "temp_images") # No longer needed for images
# os.makedirs(TEMP_IMAGE_FOLDER, exist_ok=True)

# GENERATED_FILES = set() # No longer needed

app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder="templates")
# app.config["TEMP_IMAGE_FOLDER"] = TEMP_IMAGE_FOLDER # No longer needed

# Initialize *all* supported pipeline instances
pipelines = {"Matcher": Matcher(), "PerSam": PerSam()}

# Initialize pipeline components
pipeline = PerSam()


# --- Helper Functions ---
def prepare_image_for_web(image_np):
    """Encodes BGR image as Base64 PNG data URI."""
    # Ensure BGR format for encoding
    if image_np.ndim == 3 and image_np.shape[2] == 3:
        # Assuming input is RGB, convert to BGR for OpenCV encoding
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    elif image_np.ndim == 2:
        # Convert grayscale to BGR
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image_np  # Assume already BGR

    # Encode image as PNG in memory
    is_success, buffer = cv2.imencode(".png", image_bgr)
    if not is_success:
        raise ValueError("Could not encode image to PNG")

    # Encode buffer to Base64 string
    png_as_text = base64.b64encode(buffer).decode("utf-8")

    # Return Base64 data URI
    return f"data:image/png;base64,{png_as_text}"


def prepare_mask_image_for_web(mask_np):
    """Encodes a single-channel mask as a transparent Base64 PNG data URI."""
    # Ensure mask is uint8 (0 or 1/True)
    if mask_np.dtype != np.uint8:
        mask_np = mask_np.astype(np.uint8)

    # Create 4-channel BGRA image
    h, w = mask_np.shape
    bgra_mask = np.zeros((h, w, 4), dtype=np.uint8)

    # Set RED color for mask pixels (B=0, G=0, R=255)
    bgra_mask[mask_np > 0, 0] = 0  # Blue
    bgra_mask[mask_np > 0, 1] = 0  # Green
    bgra_mask[mask_np > 0, 2] = 255  # Red
    # Set alpha channel (A=255 for mask, A=0 for background)
    bgra_mask[mask_np > 0, 3] = 255

    # Encode BGRA image as PNG in memory
    is_success, buffer = cv2.imencode(".png", bgra_mask)
    if not is_success:
        raise ValueError("Could not encode mask image to PNG")

    # Encode buffer to Base64 string
    png_as_text = base64.b64encode(buffer).decode("utf-8")

    # Return Base64 data URI
    return f"data:image/png;base64,{png_as_text}"


def process_points_for_web(points_obj: Points):
    """Converts Points object to a JSON-serializable list."""
    processed_points = []
    if not points_obj or not hasattr(points_obj, "data"):
        return processed_points

    for class_id, list_of_tensors in points_obj.data.items():
        for tensor in list_of_tensors:
            # Convert tensor to list of lists/tuples [x, y, score]
            points_list = tensor.cpu().tolist()
            for point_data in points_list:
                # Ensure we have at least x, y, score, label
                if len(point_data) >= 4:
                    x = point_data[0]
                    y = point_data[1]
                    score = point_data[2]
                    label = int(point_data[3])  # Get label (0=bg, >0=fg)
                    processed_points.append(
                        {
                            "class_id": class_id,  # Keep class_id for color grouping
                            "x": x,
                            "y": y,
                            "score": score,
                            "label": label,
                        }
                    )
                elif (
                    len(point_data) >= 2
                ):  # Fallback if label is missing (shouldn't happen often)
                    x = point_data[0]
                    y = point_data[1]
                    score = point_data[2] if len(point_data) > 2 else None
                    processed_points.append(
                        {
                            "class_id": class_id,
                            "x": x,
                            "y": y,
                            "score": score,
                            "label": 1,  # Assume foreground if label missing
                        }
                    )
    return processed_points


# --- Routes ---
@app.route("/")
def index():
    """Serves the main HTML page."""
    return render_template("index.html")


@app.route("/api/classes")
def get_classes():
    """Returns a list of unique class names for a given dataset."""
    dataset_name = request.args.get(
        "dataset", "PerSeg"
    )  # Get dataset name from query param
    try:
        full_dataset = load_dataset(dataset_name)
        if full_dataset.empty:
            return jsonify(
                {"error": f"Dataset '{dataset_name}' not found or is empty."}
            ), 404

        # Get unique class names and convert to list for JSON
        unique_classes = full_dataset["class_name"].unique().tolist()
        return jsonify({"classes": unique_classes})
    except Exception as e:
        app.logger.error(
            f"Error getting classes for {dataset_name}: {e}", exc_info=True
        )
        return jsonify({"error": "Could not retrieve class list."}), 500


@app.route("/api/process", methods=["POST"])
def run_processing():
    """Runs the pipeline and returns results."""
    try:
        data = request.json
        # Get selected pipeline name from request, default to Matcher
        pipeline_name = data.get("pipeline", "Matcher")
        if pipeline_name not in pipelines:
            return jsonify({"error": f"Unsupported pipeline: {pipeline_name}"}), 400

        # Select the pipeline instance
        selected_pipeline = pipelines[pipeline_name]

        dataset_name = data.get("dataset", "PerSeg")
        class_name_filter = data.get("class_name", "can")
        n_shot = int(data.get("n_shot", 1))

        # --- 1. Load Data ---
        full_dataset = load_dataset(dataset_name)
        if not class_name_filter:
            return jsonify({"error": "Class name filter cannot be empty"}), 400

        dataset = full_dataset[full_dataset["class_name"] == class_name_filter]
        if dataset.empty:
            return jsonify(
                {
                    "error": f"No data found for class '{class_name_filter}' in dataset '{dataset_name}'"
                }
            ), 404

        if len(dataset) <= n_shot:
            return jsonify(
                {
                    "error": f"Not enough samples ({len(dataset)}) for class '{class_name_filter}' to provide {n_shot} reference shots and at least one target."
                }
            ), 400

        references = dataset.head(n_shot)
        targets = dataset.iloc[n_shot:]  # Select subsequent rows as targets

        # --- 2. Prepare Pipeline Input ---
        reference_images = []
        reference_priors = []
        # Use the selected pipeline's state
        selected_pipeline.reset_state()

        for _, ref in references.iterrows():
            image_np = cv2.cvtColor(cv2.imread(ref.image), cv2.COLOR_BGR2RGB)
            # Read mask as default (BGR)
            mask_np = cv2.imread(ref.mask_image)

            ref_image_obj = Image(image_np)
            reference_images.append(ref_image_obj)

            masks = Masks()
            # Pass the BGR mask directly to add method
            masks.add(mask_np, class_id=0)  # Assuming single class 0 for now
            reference_priors.append(Priors(masks=masks))

        target_image_objects = []
        original_target_paths = []
        for _, target in targets.iterrows():
            img_np = cv2.cvtColor(cv2.imread(target.image), cv2.COLOR_BGR2RGB)
            target_image_objects.append(Image(img_np))
            original_target_paths.append(target.image)  # Keep track of original path

        # --- 3. Run Selected Pipeline ---
        selected_pipeline.learn(
            reference_images=reference_images, reference_priors=reference_priors
        )
        selected_pipeline.infer(target_image_objects)

        # --- 4. Process Results from selected pipeline's state ---
        results = []
        # Check lengths using selected_pipeline._state
        if len(target_image_objects) != len(selected_pipeline._state.masks):
            raise ValueError(
                "Mismatch between number of target images and masks in state."
            )
        if len(target_image_objects) != len(selected_pipeline._state.used_points):
            raise ValueError(
                "Mismatch between number of target images and used points in state."
            )

        for i, (target_img_obj, masks_obj, points_obj) in enumerate(
            zip(
                target_image_objects,
                selected_pipeline._state.masks,
                selected_pipeline._state.used_points,
            )
        ):
            # Encode target image directly
            img_data_uri = prepare_image_for_web(target_img_obj.data)

            # Process masks into data URIs
            processed_mask_data_uris = []
            instance_counter = 0
            target_img_h, target_img_w = target_img_obj.data.shape[:2]

            if masks_obj and hasattr(masks_obj, "data"):
                for class_id, list_of_tensors in masks_obj.data.items():
                    for mask_tensor in list_of_tensors:
                        # Ensure tensor is on CPU and convert to numpy
                        mask_np = mask_tensor.cpu().numpy()

                        # Handle potential extra dimensions (like batch dim)
                        if mask_np.ndim > 2:
                            mask_np = np.squeeze(mask_np)

                        # Convert boolean or float mask to 0/1 uint8
                        if mask_np.dtype == bool or mask_np.dtype == np.bool_:
                            mask_np = mask_np.astype(np.uint8)
                        elif (
                            mask_np.dtype == torch.float32
                            or mask_np.dtype == torch.float16
                        ):
                            mask_np = (mask_np > 0.5).astype(
                                np.uint8
                            )  # Threshold float masks
                        else:
                            # Assume already uint8-like if not bool/float
                            mask_np = (mask_np > 0).astype(np.uint8)

                        # Resize mask if necessary (should match target image)
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

            # Process points
            web_points = process_points_for_web(points_obj)

            results.append(
                {
                    "image_data_uri": img_data_uri,
                    "masks": processed_mask_data_uris,
                    "points": web_points,
                }
            )

        return jsonify({"target_results": results})

    except FileNotFoundError as e:
        app.logger.error(f"File not found: {e}")
        return jsonify({"error": f"Server error: File not found - {e.filename}"}), 500
    except ImportError as e:
        app.logger.error(f"Import error: {e}")
        return jsonify({"error": f"Server configuration error: {e.name}"}), 500
    except Exception as e:
        app.logger.error(f"An error occurred: {e}", exc_info=True)  # Log stack trace
        return jsonify(
            {"error": f"An unexpected error occurred: {type(e).__name__}"}
        ), 500


if __name__ == "__main__":
    # Make sure templates and static directories exist
    if not os.path.exists("templates"):
        os.makedirs("templates")
    if not os.path.exists("static"):
        os.makedirs("static")

    # Create a dummy index.html if it doesn't exist
    if not os.path.exists("templates/index.html"):
        with open("templates/index.html", "w") as f:
            f.write(
                "<html><head><title>Processing App</title></head><body><h1>Processing...</h1></body></html>"
            )

    app.run(debug=True)  # debug=True for development, False for production
