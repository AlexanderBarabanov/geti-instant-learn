import colorsys
from typing import List, Dict

import numpy as np
import ot
from matplotlib import pyplot as plt
from typing import Union
from torch.nn import functional as F
from sklearn.cluster import KMeans
import cv2
import torch
import umap
from sklearn.manifold import TSNE
import random

from model_api.models import Prompt
from model_api.models.visual_prompting import VisualPromptingFeatures
from context_learner.types import Image, Masks, Points

# Define some colors for visualization
COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (0, 128, 128),
    (128, 0, 128),
]

def _draw_points(img, x, y, size=1, color=(255, 255, 255)):
    if size == 1:  # faster
        img[y.astype(int), x.astype(int)] = color
    else:  # slower
        for x, y in zip(x, y):
            cv2.drawMarker(img, (int(x), int(y)), color, cv2.MARKER_STAR, size)
    return img

def overlay_masks_and_points(
    image: Image, masks: Masks, points: Points, alpha: float = 0.3
) -> np.ndarray:
    """
    Overlays masks onto an image using fixed blending weights (0.7 image, 0.3 mask),
    similar to the provided main.py snippet. Creates a combined mask first.
    Draws points on top with distinct colors per class.

    Args:
        image: The input Image object.
        masks: A Masks object containing masks per class.
        points: A Points object containing points per class.
        alpha: Transparency level (currently ignored for mask blending, uses 0.3).

    Returns:
        The image (as a numpy array) with mask overlays and points.
    """
    img_np = image.data.copy()  # Work on a copy

    # Ensure image is 3-channel BGR
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 4:  # RGBA
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)

    # Create a combined mask from all classes
    combined_mask_np = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
    for class_id, mask_tensor in masks.data.items():
        if mask_tensor.numel() == 0:
            continue
        class_combined_mask_tensor = torch.any(mask_tensor, dim=0)
        mask_np = class_combined_mask_tensor.cpu().numpy().astype(np.uint8)

        if mask_np.ndim == 3:
            mask_np = mask_np.squeeze()

        if mask_np.shape != (img_np.shape[0], img_np.shape[1]):
            mask_np = cv2.resize(
                mask_np,
                (img_np.shape[1], img_np.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Combine with the overall mask using logical OR
        combined_mask_np = np.logical_or(combined_mask_np, mask_np).astype(np.uint8)

    # Create a 3-channel BGR image from the final combined mask (white where mask is > 0)
    final_mask_img = np.zeros_like(img_np, dtype=np.uint8)
    final_mask_img[combined_mask_np > 0] = (255, 255, 255)

    # Blend the original image with the final 3-channel mask image using fixed weights
    overlay_image = cv2.addWeighted(img_np, 0.7, final_mask_img, 0.3, 0)

    # Draw points on top of the blended image
    color_idx = 0
    for class_id, points_per_map_tensor in points.data.items():
        color = COLORS[color_idx % len(COLORS)]
        color_idx += 1
        for points_tensor in points_per_map_tensor:
            for point in points_tensor:
                # Use .item() to get scalar value from tensor
                x, y = int(point[0].item()), int(point[1].item())
                cv2.circle(
                    overlay_image, (x, y), radius=5, color=color, thickness=-1
                )  # Filled circle
                # Optional: Add a border for better visibility
                cv2.circle(
                    overlay_image, (x, y), radius=5, color=(0, 0, 0), thickness=1
                )

    return overlay_image


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


def prepare_target_guided_prompting(
    sim: torch.Tensor, reference_features: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare target guided prompting for the decoder. Produces attention similarity and target embedding for the decoder.
    This technique is used in Per-Segment-Anything and can improve the performance of the decoder by providing additional information.
    Note that not all backbones support this technique.

    Args:
        sim: similarity tensor
        reference_features: reference features tensor

    Returns:
        attention_similarity: attention similarity tensor
        reference_features: reference features tensor
    """
    # For multiple similarity masks (e.g. Part-level features), we take the mean of the similarity maps
    if len(sim.shape) == 3:
        sim = sim.mean(dim=0)

    sim = (sim - sim.mean()) / torch.std(sim)
    sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
    attention_similarity = sim.sigmoid_().unsqueeze(0).flatten(3)

    # For multiple reference features (e.g. Part-level features), we take the mean of the reference features
    if len(reference_features.shape) == 2:
        reference_features = reference_features.mean(0).unsqueeze(0)
    return attention_similarity, reference_features


def cluster_features(
    reference_features: torch.Tensor, n_clusters: int = 8, visualize: bool = False
) -> Union[torch.Tensor, tuple[torch.Tensor, plt.Figure]]:
    """Create part-level features from reference features.
    This performs a k-means++ clustering on the reference features and takes the centroid as prototype.
    Resulting part-level features are normalized to unit length.
    If n_clusters is 1, the mean of the reference features is taken as prototype.

    Args:
        reference_features: Reference features tensor [X, 256]
        n_clusters: Number of clusters to create (e.g. number of part-level-features)
        visualize: Whether to return UMAP visualization of features and clusters

    Returns:
        Part-level features tensor [n_clusters, 256] and optionally a matplotlib figure
    """
    if n_clusters == 1:
        part_level_features = reference_features.mean(0).unsqueeze(0)
        part_level_features = part_level_features / part_level_features.norm(
            dim=-1, keepdim=True
        )
        return part_level_features.unsqueeze(0), None  # 1, 256

    features_np = reference_features.cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0)
    cluster = kmeans.fit_predict(features_np)
    part_level_features = []

    for c in range(n_clusters):
        # use centroid of cluster as prototype
        part_level_feature = features_np[cluster == c].mean(axis=0)
        part_level_feature = part_level_feature / np.linalg.norm(
            part_level_feature, axis=-1, keepdims=True
        )
        part_level_features.append(torch.from_numpy(part_level_feature))

    part_level_features = torch.stack(
        part_level_features, dim=0
    ).cuda()  # [n_clusters, 256]

    if visualize:
        features_np = reference_features.cpu().numpy()
        fig = visualize_feature_clusters(
            features=features_np,
            cluster_labels=cluster,
            cluster_centers=kmeans.cluster_centers_,
        )
        return part_level_features, fig

    return part_level_features, None


def cluster_points(points: np.ndarray, n_clusters: int = 8) -> np.ndarray:
    """Cluster points using k-means++."""
    if len(points) < n_clusters:
        return points
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0)
    cluster = kmeans.fit_predict(points)
    # use centroid of cluster as prototype
    prototypes = []
    for c in range(n_clusters):
        prototype = points[cluster == c].mean(axis=0)
        prototypes.append(prototype)
    return np.array(prototypes).astype(np.int64)


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


def gen_colors(n: int):
    hsv_tuples = [(x / n, 0.5, 0.5) for x in range(n)]
    rgb_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)
    colors = [
        (0, 0, 0),
    ] + list(rgb_tuples)
    return (np.array(colors) * 255).astype(np.uint8)


def color_overlay(image: np.ndarray, mask: np.ndarray):
    mask_colors = gen_colors(np.max(mask))
    color_mask = mask_colors[mask]  # create color map
    color_mask[mask == 0] = image[mask == 0]  # set background to original color
    image_vis = cv2.addWeighted(image, 0.2, color_mask, 0.8, 0)
    return image_vis[:, :, ::-1]  # BGR2RGB


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
        if len(features) > 10:
            print(f"Skipping class {label} with {len(features)} features")
            continue
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


def generate_combinations(n: int, k: int) -> list[list[int]]:
    """
    Generate all possible k-combinations from n elements.

    This function recursively generates all possible combinations of k elements
    chosen from a set of n elements (0 to n-1).

    Args:
        n: The total number of elements to choose from (0 to n-1)
        k: The number of elements to include in each combination

    Returns:
        list[list[int]]: A list of all possible k-combinations, where each combination
            is represented as a list of integers

    Examples:
        >>> combinations(3, 2)
        [[0, 1], [0, 2], [1, 2]]
        >>> combinations(2, 0)
        [[]]
        >>> combinations(2, 3)
        []
    """
    if k > n:
        return []
    if k == 0:
        return [[]]
    if k == n:
        return [[i for i in range(n)]]
    res = []
    for i in range(n):
        for j in generate_combinations(i, k - 1):
            res.append(j + [i])
    return res


def is_in_mask(point, mask):
    """
    Check if a point is in a mask.
    """
    h, w = mask.shape
    point = point.astype(np.int64)
    point = point[:, ::-1]  # y,x
    point = np.clip(point, 0, [h - 1, w - 1])
    return mask[point[:, 0], point[:, 1]]


def sample_points(
    points: np.ndarray, sample_range: tuple[int, int] = (4, 6), max_iterations: int = 30
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Sample points by generating point sets of different sizes. Point sets can contain duplicates.
    Point sets have equal length so they can be batched. Note that each point sets has increased amount of
    points. e.g. subset0.shape = X, 1, 2 and subset1.shape = X, 2, 2, subset3.shape = X, 3, 2 etc.
    This is used to generate prompts with different granularity.

    For small point sets (≤8 points), generates all possible combinations.
    For larger sets (>8 points), uses random sampling to generate max_iterations samples.

    Args:
        points: Input points array of shape (N, 2) where N is number of points
        sample_range: Tuple of (min_points, max_points) to sample
        max_iterations: Maximum number of random sampling iterations for large point sets

    Returns:
        tuple containing:
            - sample_list: List of arrays, where each array has shape:
                - (max_iterations, i, 2) for large point sets (random sampling)
                - (n_combinations, i, 2) for small point sets (all combinations)
                where i ranges from sample_range[0] to sample_range[1], and 2 represents x,y coordinates
            - label_list: List of arrays, where each array has shape:
                - (max_iterations, i) for large point sets
                - (n_combinations, i) for small point sets
                containing ones for each sampled point

    """
    sample_list = []
    label_list = []
    for i in range(
        min(sample_range[0], len(points)),
        min(sample_range[1], len(points)) + 1,
    ):
        if len(points) > 8:
            index = [
                random.sample(range(len(points)), i) for j in range(max_iterations)
            ]
            sample = np.take(points, index, axis=0)  # (max_iterations * i) * 2
        else:
            index = generate_combinations(len(points), i)
            sample = np.take(points, index, axis=0)  # i * n * 2

        # generate label  max_iterations * i
        label = np.ones((sample.shape[0], i))
        sample_list.append(sample)
        label_list.append(label)
    return sample_list, label_list
