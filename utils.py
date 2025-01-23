import colorsys
import os
from typing import List, Dict

import numpy as np
import ot
import pandas as pd
import cv2
import torch
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE

from model_api.models import Prompt
from constants import DATA_PATH
from model_api.models.result_types.visual_prompting import PredictedMask
from model_api.models.visual_prompting import VisualPromptingFeatures, _draw_points

DATAFRAME_COLUMNS = ["class_name", "file_name", "image", "mask_image", "frame"]


def load_dataset(dataset_name: str) -> pd.DataFrame:
    if dataset_name == "PerSeg":
        return load_perseg_data()
    elif dataset_name == "DAVIS":
        return load_davis_data()


def load_perseg_data() -> pd.DataFrame:
    images_path = os.path.join(DATA_PATH, "PerSeg", "Images")
    annotations_path = os.path.join(DATA_PATH, "PerSeg", "Annotations")

    data = pd.DataFrame(columns=DATAFRAME_COLUMNS)

    for class_name in os.listdir(images_path):
        if ".DS" in class_name:
            continue

        for file_name in os.listdir(os.path.join(images_path, class_name)):
            if ".DS" in file_name:
                continue

            frame = int(file_name[:-4])  # Remove .jpg and convert to int

            data = pd.concat(
                [
                    data,
                    pd.DataFrame(
                        [
                            {
                                "class_name": class_name,
                                "file_name": file_name,
                                "image": os.path.join(
                                    images_path, class_name, file_name
                                ),
                                "mask_image": os.path.join(
                                    annotations_path,
                                    class_name,
                                    file_name[:-4] + ".png",
                                ),
                                "frame": frame,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    # sort on class_name and frame
    data.sort_values(by=["class_name", "frame"], inplace=True)
    return data


def load_davis_data() -> pd.DataFrame:
    """Load DAVIS dataset into a pandas DataFrame.
    Returns DataFrame with columns: class_name, file_name, image, mask_image, frame
    """
    images_path = os.path.join(DATA_PATH, "DAVIS" "JPEGImages", "480p")
    annotations_path = os.path.join(DATA_PATH, "DAVIS", "Annotations", "480p")
    imagesets_path = os.path.join(DATA_PATH, "DAVIS", "ImageSets", "2017", "val.txt")

    data = pd.DataFrame(columns=DATAFRAME_COLUMNS)

    with open(imagesets_path, "r") as f:
        sequences = [x.strip() for x in f.readlines()]

    for sequence in sequences:
        frames = sorted(os.listdir(os.path.join(images_path, sequence)))

        for frame in frames:
            if frame.endswith(".jpg"):
                frame_id = frame[:-4]  # Remove .jpg extension
                mask_file = frame_id + ".png"

                frame_number = int(frame_id)

                data = pd.concat(
                    [
                        data,
                        pd.DataFrame(
                            [
                                {
                                    "class_name": sequence,
                                    "file_name": frame,
                                    "image": os.path.join(images_path, sequence, frame),
                                    "mask_image": os.path.join(
                                        annotations_path, sequence, mask_file
                                    ),
                                    "frame": frame_number,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

    # Sort by class_name and frame
    data.sort_values(by=["class_name", "frame"], inplace=True)
    return data


def mask_image_to_polygon_prompts(mask_image: np.array) -> List[Prompt]:
    mask = mask_image.astype(np.uint8)
    mask = np.stack((mask, np.zeros_like(mask), np.zeros_like(mask)), axis=-1)


def get_colors(n: int):
    HSV_tuples = [(x / n, 0.5, 0.5) for x in range(n)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return (np.array(list(RGB_tuples)) * 255).astype(np.uint8)


def transform_point_prompts_to_dict(
    prompts: List[Prompt],
) -> Dict[int, List[List[int]]]:
    result = {}
    for prompt in prompts:
        label = prompt.label
        coords = prompt.data.tolist()
        if label not in result:
            result[label] = []
        result[label].append(coords)
    return result


def transform_mask_prompts_to_dict(prompts: List[Prompt]) -> Dict[int, np.array]:
    """
    Transform masks into a dictionary of masks per class.

    Args:
        prompts: List[Prompt]
    Returns:
        Dict[int, np.array]  where the key is the class index and the value is the mask
    """
    result = {}
    for prompt in prompts:
        label = prompt.label
        mask = prompt.data
        result[label] = mask
    return result


def similarity_maps_to_visualization(
    similarity_maps: np.ndarray | torch.Tensor,
    points: np.ndarray | None = None,
    bg_points: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert similarity maps to visualization with optional point overlays.

    Args:
        similarity_maps: Either a 2D array (H,W) or 3D array (N,H,W) of similarity maps
        points: Optional foreground points to visualize as white dots (N,2)
        bg_points: Optional background points to visualize as red dots (N,2)
    """
    if isinstance(similarity_maps, torch.Tensor):
        similarity_maps = similarity_maps.cpu().numpy()

    if len(similarity_maps.shape) == 2:
        # Handle single 2D similarity map
        similarity = np.zeros((*similarity_maps.shape, 3), dtype=np.uint8)
        similarity[:, :] = np.expand_dims(similarity_maps * 255, axis=2)

        # Draw points if provided
        if points is not None:
            _draw_points(
                similarity, points[:, 0], points[:, 1], size=15, color=(255, 255, 255)
            )
        if bg_points is not None:
            _draw_points(
                similarity, bg_points[:, 0], bg_points[:, 1], size=15, color=(0, 0, 255)
            )
    else:
        # Handle stack of similarity maps - arrange horizontally
        n_maps, height, width = similarity_maps.shape
        # Create a wide image to hold all maps side by side
        similarity = np.zeros((height, width * n_maps, 3), dtype=np.uint8)

        for i in range(n_maps):
            # Create individual map visualization
            single_map = np.zeros((height, width, 3), dtype=np.uint8)
            single_map[:, :] = np.expand_dims(similarity_maps[i] * 255, axis=2)

            # Draw points on individual map if provided
            if points is not None:
                _draw_points(
                    single_map,
                    points[:, 0],
                    points[:, 1],
                    size=15,
                    color=(255, 255, 255),
                )
            if bg_points is not None:
                _draw_points(
                    single_map,
                    bg_points[:, 0],
                    bg_points[:, 1],
                    size=15,
                    color=(0, 0, 255),
                )

            # Place the map with points in its position
            start_x = i * width
            end_x = (i + 1) * width
            similarity[:, start_x:end_x] = single_map

    return similarity


def save_visualization(
    image: np.ndarray,
    masks_result: PredictedMask,
    visual_outputs: dict[str, dict[int, np.ndarray]],
    output_path: str,
    points=None,
    scores=None,
) -> None:
    """
    Save a visualization of the segmentation mask overlaid on the image.

    Args:
        image: RGB image as numpy array
        mask: Segmentation mask object with mask.mask containing instance masks
        visual_output: Extra images representing the intermediate results of the algorithm
        output_path: Path where to save the visualization
        points: Optional points to visualize
        scores: Optional confidence scores for the points
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get unique colors for each instance mask
    mask_colors = get_colors(len(masks_result.mask))
    image_vis = image.copy()

    # Create visualization output
    base = os.path.splitext(output_path)[0]
    for name, data in visual_outputs.items():
        fn = f"{base}_{name}.png"
        cv2.imwrite(fn, data[0])

    # Draw each instance mask with a different color
    for i, instance in enumerate(masks_result.mask):
        masked_img = np.where(instance[..., None], mask_colors[i], image_vis)
        image_vis = cv2.addWeighted(image_vis, 0.2, masked_img, 0.8, 0)

    # Draw points and confidence scores if provided
    if points is not None and scores is not None:
        for i, point in enumerate(points):
            # Draw star marker
            x, y = int(point[0]), int(point[1])
            size = int(image.shape[0] / 50)  # Scale marker size with image
            cv2.drawMarker(image_vis, (x, y), (255, 255, 255), cv2.MARKER_STAR, size)

            # Add confidence score text
            confidence = float(scores[i])
            cv2.putText(
                image_vis,
                f"{confidence:.2f}",
                (x + 5, y - 5),  # Offset text slightly from point
                cv2.FONT_HERSHEY_SIMPLEX,
                image.shape[0] / 1500,  # Font scale relative to image height
                (255, 255, 255),  # White text
                1,
            )  # Line thickness

    # Save visualization
    cv2.imwrite(output_path, cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))


def show_cosine_distance(reference_features: dict[int, torch.Tensor]) -> None:
    """
    Show pair wise cosine distance between reference features for each class in a formatted table.

    Args:
        reference_features: Dictionary mapping class labels to their reference features tensors
    """
    if isinstance(reference_features, VisualPromptingFeatures):
        # modelAPI does not support multiple reference features
        return

    for label, features in reference_features.items():
        if features.shape[0] > 1:
            n_features = features.shape[0]
            high_similarity_found = False

            # Print header with class label
            print(f"\nCosine Similarity Matrix for class {label} reference features:")
            print("-" * 50)
            header = "    " + "".join(f"  [{i}]  " for i in range(n_features))
            print(header)
            print("-" * 50)

            # Print similarity matrix
            for i in range(n_features):
                row = f"[{i}]"
                for j in range(n_features):
                    if j < i:
                        sim = torch.nn.functional.cosine_similarity(
                            features[i].squeeze(),
                            features[j].squeeze(),
                            dim=0,
                        ).item()
                        if sim > 0.9:
                            high_similarity_found = True
                        row += f" {sim:6.3f}"
                    elif j == i:
                        row += "  1.000"
                    else:
                        row += "      -"
                print(row)
            print("-" * 50)

            if high_similarity_found:
                print("\nWarning: Some reference features have similarity > 0.9")
                print(
                    "This indicates very similar semantic information between points,"
                )
                print("which might not contribute to better performance.")


def _compute_wasserstein_distance(a, b, weights=None) -> float:
    """
    Computes the Wasserstein distance between two distributions a and b.
    Lower distance is better match.
    :param a:
    :param b:
    :param weights:
    :return:
    """
    n_a = a.shape[0]
    a_hist = ot.unif(n_a)
    if weights is not None:
        b_hist = weights / np.sum(weights)
    else:
        n_b = b.shape[0]
        b_hist = ot.unif(n_b)

    M = ot.dist(a, b)
    wasserstein_distance = ot.emd2(a_hist, b_hist, M, numItermax=10000000)
    return wasserstein_distance


def visualize_feature_clusters(
    features: np.ndarray, cluster_labels: np.ndarray, cluster_centers: np.ndarray
) -> plt.Figure:
    """
    Create UMAP and t-SNE visualizations of feature clusters side by side.

    Args:
        features: Feature vectors to visualize [N, D]
        cluster_labels: Cluster assignment for each feature [N]
        cluster_centers: Cluster centroids [K, D]

    Returns:
        matplotlib Figure with UMAP and t-SNE visualizations
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # UMAP embedding
    reducer_umap = umap.UMAP(random_state=42)
    embedding_umap = reducer_umap.fit_transform(features)
    centroids_embedding_umap = reducer_umap.transform(cluster_centers)

    # t-SNE embedding with adjusted perplexity
    n_centers = len(cluster_centers)
    perplexity = min(30, n_centers - 1)  # Ensure perplexity is less than n_samples
    reducer_tsne = TSNE(random_state=42, perplexity=perplexity)
    embedding_tsne = reducer_tsne.fit_transform(features)
    centroids_embedding_tsne = reducer_tsne.fit_transform(cluster_centers)

    # Plot UMAP
    scatter1 = ax1.scatter(
        embedding_umap[:, 0],
        embedding_umap[:, 1],
        c=cluster_labels,
        cmap="tab10",
        alpha=0.6,
        s=100,
    )
    ax1.scatter(
        centroids_embedding_umap[:, 0],
        centroids_embedding_umap[:, 1],
        marker="*",
        s=300,
        c=range(len(cluster_centers)),
        cmap="tab10",
        edgecolor="black",
        linewidth=1.5,
    )
    ax1.set_title("UMAP projection")

    # Plot t-SNE
    scatter2 = ax2.scatter(
        embedding_tsne[:, 0],
        embedding_tsne[:, 1],
        c=cluster_labels,
        cmap="tab10",
        alpha=0.6,
        s=100,
    )
    ax2.scatter(
        centroids_embedding_tsne[:, 0],
        centroids_embedding_tsne[:, 1],
        marker="*",
        s=300,
        c=range(len(cluster_centers)),
        cmap="tab10",
        edgecolor="black",
        linewidth=1.5,
    )
    ax2.set_title("t-SNE projection")

    # Add colorbar
    plt.colorbar(scatter1, ax=ax1, label="Cluster")
    plt.colorbar(scatter2, ax=ax2, label="Cluster")

    fig.suptitle(
        "Dimensionality reduction projections of reference features\nPoints colored by cluster, centroids shown as stars",
        y=1.02,
    )

    return fig
