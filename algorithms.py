import os
from collections import defaultdict

from pandas import Series
from sklearn.cluster import KMeans

from constants import MAPI_DECODER_PATH, MAPI_ENCODER_PATH, MODEL_MAP
from efficientvit.models.efficientvit import EfficientViTSamPredictor, SamResize, SamPad
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from P2SAM.eval_utils import intersectionAndUnion
import PersonalizeSAM.persam
import PersonalizeSAM.show
from PersonalizeSAM.per_segment_anything import SamPredictor
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
import cv2
import numpy as np

from PersonalizeSAM.per_segment_anything import sam_model_registry
from model_api.models import Prompt, PredictedMask, ZSLVisualPromptingResult
from model_api.models.model import Model
from model_api.models.visual_prompting import (
    SAMLearnableVisualPrompter,
    VisualPromptingFeatures,
    _inspect_overlapping_areas,
    _point_selection,
)
from utils import (
    similarity_maps_to_visualization,
    transform_point_prompts_to_dict,
    transform_mask_prompts_to_dict,
)


class PerSamPredictor:
    """Wrapper on top of the SAM model to make it inline with ModelAPI SAM Interface"""

    def __init__(self, model: SamPredictor, algo_name: str = "Personalized SAM"):
        self.grid_size = 64
        self.num_bg_points = 1
        self.threshold = 0.65
        self.model = model
        self.algo_name = algo_name
        self.reference_features = {}
        self.reference_masks = {}
        self.highest_class_idx = 0  # num_classes  - 1

    def learn(
        self,
        image: np.array,
        boxes: list[Prompt] | None = None,
        points: list[Prompt] | None = None,
        polygons: list[Prompt] | None = None,
        masks: list[Prompt] | None = None,
        show: bool = False,
        num_clusters: int = 1,
    ) -> tuple[VisualPromptingFeatures, np.array]:
        mask_per_class = self.prepare_input(image, masks, points, boxes, polygons)

        for class_idx, mask in mask_per_class.items():
            if show:
                fig = plt.figure(figsize=(10, 10))
                plt.imshow(mask)
                plt.show()
                cv2.imwrite(f"reference_mask_{class_idx}.jpg", mask)

            image_embedding, reference_features = self.extract_reference_features(
                mask, image
            )
            if reference_features is None:
                continue

            # PerSAM: use average of all features.
            # P2SAM/PartAware: use k-means++ clustering to create num_clusters part-level features
            reference_features = cluster_features(reference_features, num_clusters)

            # Accumulate features to allow for few-shot learning
            if class_idx in self.reference_features:
                self.reference_features[class_idx] = torch.cat(
                    [self.reference_features[class_idx], reference_features], dim=0
                )
                self.reference_masks[class_idx] = np.concatenate(
                    [self.reference_masks[class_idx], mask[:, :, 0]], axis=0
                )
            else:
                self.reference_features[class_idx] = reference_features
                self.reference_masks[class_idx] = mask[:, :, 0]

        if not self.reference_features:
            print("No reference features found. Please provide a larger reference mask")
            return None, None

        return VisualPromptingFeatures(
            feature_vectors=np.stack(
                [v.cpu().numpy() for v in self.reference_features.values()]
            ),
            used_indices=np.array(self.reference_features.keys()),
        ), np.stack(list(self.reference_masks.values()))

    def reset_reference_features(self):
        """Reset all accumulated reference features and masks."""
        self.reference_features = {}
        self.reference_masks = {}
        self.highest_class_idx = 0

    def few_shot_learn(
        self,
        images: list[np.array],
        boxes: list[list[Prompt]] | None = None,
        points: list[list[Prompt]] | None = None,
        polygons: list[list[Prompt]] | None = None,
        masks: list[list[Prompt]] | None = None,
        show: bool = False,
        num_clusters: int = 1,
    ) -> tuple[VisualPromptingFeatures, np.array]:
        """Learn from multiple support images and their corresponding prompts.

        Args:
            images: List of support images
            boxes: List of box prompts per image
            points: List of point prompts per image
            polygons: List of polygon prompts per image
            masks: List of mask prompts per image
            show: Whether to show debug visualizations
            num_clusters: Number of clusters for feature clustering

        Returns:
            features: Combined features from all support images
            masks: Stack of reference masks
        """
        self.reset_reference_features()

        for i in range(len(images)):
            current_boxes = boxes[i] if boxes is not None else None
            current_points = points[i] if points is not None else None
            current_polygons = polygons[i] if polygons is not None else None
            current_masks = masks[i] if masks is not None else None

            self.learn(
                image=images[i],
                boxes=current_boxes,
                points=current_points,
                polygons=current_polygons,
                masks=current_masks,
                show=show,
                num_clusters=num_clusters,
            )

        return VisualPromptingFeatures(
            feature_vectors=np.stack(
                [v.cpu().numpy() for v in self.reference_features.values()]
            ),
            used_indices=np.array(self.reference_features.keys()),
        ), np.stack(list(self.reference_masks.values()))

    def infer(
        self,
        image: np.array,
        reference_features=None,
        apply_masks_refinement: bool = True,
        target_guided_attention: bool = False,
        mask_generation_method: str = "point-by-point",
        selection_on_similarity_maps: str = "per-map",
    ) -> ZSLVisualPromptingResult:
        prediction: dict[int, PredictedMask] = {}
        final_point_prompts: dict[int, list] = defaultdict(list)
        all_point_prompt_candidates: dict[int, np.ndarray] = defaultdict(list)
        all_masks: dict[int, list] = defaultdict(list)
        all_scores: dict[int, list] = defaultdict(list)
        all_bg_prompts: dict[int, list] = defaultdict(list)
        sim_masks_per_class = {}
        attn_sim = None
        target_guided_embedding = None
        visual_outputs: dict[str, dict[int, np.ndarray]] = defaultdict(dict)

        # Image feature encoding
        test_feat = self.get_image_embedding_sam(image)
        c, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(c, h * w)

        for class_idx, reference_features in self.reference_features.items():
            # Cosine similarity
            sim = reference_features @ test_feat
            sim = sim.reshape(reference_features.shape[0], 1, h, w)

            sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
            sim = self.model.model.postprocess_masks(
                sim,
                input_size=self.model.input_size,
                original_size=self.model.original_size,
            ).squeeze()
            sim_masks_per_class[class_idx] = sim

            # Point selection
            if len(sim_masks_per_class[class_idx].shape) == 2:
                # allow single and multiple similarity maps
                sim_masks_per_class[class_idx] = sim_masks_per_class[
                    class_idx
                ].unsqueeze(0)

            if selection_on_similarity_maps == "per-map":
                for idx, sim in enumerate(sim_masks_per_class[class_idx]):
                    point_prompt_candidates, bg_points = _point_selection(
                        mask_sim=sim.cpu().numpy(),  # numpy  H W  720 1280
                        original_shape=np.array(
                            self.model.original_size
                        ),  # [ 720  1280]
                        threshold=self.threshold,
                    )
                    if point_prompt_candidates is None:
                        continue
                    all_point_prompt_candidates[class_idx].extend(
                        point_prompt_candidates
                    )
                    all_bg_prompts[class_idx].extend(bg_points)

                    visual_outputs[f"similarity_{idx}"][class_idx] = (
                        similarity_maps_to_visualization(
                            sim,
                            points=point_prompt_candidates,
                            bg_points=bg_points,
                        )
                    )
            elif selection_on_similarity_maps == "stacked-maps":
                visual_outputs["similarity"][class_idx] = (
                    similarity_maps_to_visualization(
                        sim_masks_per_class[class_idx],
                    )
                )
                sim_masks_per_class[class_idx] = (
                    sim_masks_per_class[class_idx].mean(dim=0).squeeze().cpu().numpy()
                )
                point_prompt_candidates, bg_points = _point_selection(
                    mask_sim=sim_masks_per_class[class_idx],
                    original_shape=np.array(self.model.original_size),
                    threshold=self.threshold,
                )
                if point_prompt_candidates is not None:
                    all_point_prompt_candidates[class_idx].extend(
                        point_prompt_candidates
                    )
                    all_bg_prompts[class_idx].extend(bg_points)
                visual_outputs["similarity_stacked"][class_idx] = (
                    similarity_maps_to_visualization(
                        sim_masks_per_class[class_idx],
                        points=point_prompt_candidates,
                        bg_points=bg_points,
                    )
                )

            # Predict masks
            point_prompt_candidates = all_point_prompt_candidates[class_idx]
            bg_points = all_bg_prompts[class_idx]

            # if no points are found, we do not return points and return one empty mask
            if len(point_prompt_candidates) == 0:
                final_point_prompts[class_idx] = []
                all_masks[class_idx] = [np.zeros_like(sim_masks_per_class[class_idx])]
                all_scores[class_idx] = [0.0]
                continue

            # Obtain the target guidance for cross-attention layers
            if target_guided_attention:
                attn_sim, target_guided_embedding = prepare_target_guided_prompting(
                    sim_masks_per_class[class_idx], self.reference_features[class_idx]
                )

            if mask_generation_method == "one-go":
                all_masks[class_idx], final_point_prompts[class_idx] = (
                    self.predict_masks(
                        point_prompt_candidates,
                        bg_points,
                        apply_masks_refinement=apply_masks_refinement,
                        attn_sim=attn_sim,
                        target_guided_embedding=target_guided_embedding,
                        class_idx=class_idx,
                    )
                )
            elif mask_generation_method == "point-by-point":
                all_masks[class_idx], final_point_prompts[class_idx] = (
                    self.predict_masks_point_by_point(
                        point_prompt_candidates,
                        bg_points,
                        apply_masks_refinement=apply_masks_refinement,
                        attn_sim=attn_sim,
                        target_guided_embedding=target_guided_embedding,
                    )
                )
        # Refine overlapping class masks
        _inspect_overlapping_areas(all_masks, final_point_prompts)

        # Create result object
        for label in final_point_prompts:
            final_point_prompts[label] = np.array(final_point_prompts[label])
            prediction[label] = PredictedMask(
                mask=all_masks[label],
                points=final_point_prompts[label][:, :2]
                if len(final_point_prompts[label]) > 0
                else np.array([]),
                scores=final_point_prompts[label][:, 2]
                if len(final_point_prompts[label]) > 0
                else np.array([]),
            )

        return ZSLVisualPromptingResult(prediction), visual_outputs

    def prepare_input(
        self,
        image: np.array,
        masks: list[Prompt] = None,
        points: list[Prompt] = None,
        boxes: list[Prompt] = None,
        polygons: list[Prompt] = None,
    ) -> dict[int, np.array]:
        """
        Create mask per class based on the input masks or points

        Args:
            image: input image
            masks: input masks
            points: input points
            boxes: input boxes
            polygons: input polygons

        Returns:
            mask_per_class: dictionary of masks per class
        """
        if points is not None:
            mask_per_class = {}
            points_per_class = transform_point_prompts_to_dict(points)
            for class_idx, points in points_per_class.items():
                self.model.set_image(image)
                masks, scores, logits, *_ = self.model.predict(
                    point_coords=np.array(points),
                    point_labels=np.array([class_idx] * len(points)),
                    multimask_output=False,
                )
                best_idx = np.argmax(scores)
                if not masks[best_idx].any():
                    print(f"No mask found for reference class {class_idx}")
                    continue
                best_mask = masks[best_idx].astype(np.uint8) * 128
                mask_per_class[class_idx] = np.stack(
                    (best_mask, np.zeros_like(best_mask), np.zeros_like(best_mask)),
                    axis=-1,
                )
            if len(mask_per_class) == 0:
                print("No masks found for any class given the input points")
                return None, None
        elif masks is not None:
            mask_per_class = transform_mask_prompts_to_dict(masks)
        else:
            raise ValueError("Either points or masks are required for learning")

        return mask_per_class

    def get_image_embedding_sam(self, image: np.array) -> torch.Tensor:
        """
        Get the image embedding for the current image.

        :param image: input image

        Returns:
            image_embedding: 256, 64, 64
        """
        self.model.set_image(image)
        return self.model.features.squeeze()

    def extract_reference_features(
        self, mask: np.array, image: np.array
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract reference features using the provided mask.

        Args:
            mask: input mask
            image: input image

        Returns:
            image_embedding: 64, 64, 256
            reference_features: X, 256
        """
        if isinstance(self.model, SamPredictor):
            # set_image resizes and pads to square input
            # TODO set_image also computes image embedding which is now performed for every mask and not just once
            # TODO this is not efficient and should only resize the mask. However, in practice we have only mask per class
            reference_mask = self.model.set_image(image, mask)  # 1, 3 ,1024, 1024
            image_embedding = self.model.features.squeeze().permute(
                1, 2, 0
            )  # 64, 64, 256
            # transform reference mask to a 64 64 mask
            reference_mask = F.interpolate(
                reference_mask, size=image_embedding.shape[0:2], mode="bilinear"
            )
            reference_mask = reference_mask.squeeze()[0]  # 64, 64
        elif isinstance(self.model, EfficientViTSamPredictor):
            self.model.set_image(image)
            image_embedding = self.model.features.squeeze().permute(
                1, 2, 0
            )  # 64, 64, 256
            # resize relative to longest size
            reference_mask = SamResize(self.model.model.image_size[0])(
                mask
            )  # 576 1024 3
            reference_mask = torch.as_tensor(reference_mask, device=self.model.device)
            reference_mask = reference_mask.permute(2, 0, 1).contiguous()[
                None, :, :, :
            ]  # 1, 3, 576, 1024
            reference_mask = SamPad(self.model.model.image_size[0])(
                reference_mask
            )  # 1, 3, 1024, 1024
            reference_mask = F.interpolate(
                reference_mask.float(),
                size=image_embedding.shape[0:2],
                mode="bilinear",
            )
            reference_mask = reference_mask.squeeze()[0]  # 64 64

        # local reference feature extraction
        reference_features = image_embedding[reference_mask > 0]
        if reference_features.shape[0] == 0:
            print(f"The reference mask is too small to detect any features")
            return None, None
        return image_embedding, reference_features

    def post_refinement(
        self,
        logits: torch.Tensor,
        point_coordinates: np.ndarray,
        point_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Refines the prediced mask by reapplying the decoder on step wise increase of input information.

        Args:
            logits: logits from the decoder
            point_coordinates: point coordinates (x, y)
            point_labels: point labels (1 for foreground, 0 for background)

        Returns:
            final_mask: refined mask
            masks: all masks
            final_score: score of the refined mask
        """
        best_idx = 0
        # Cascaded Post-refinement-1
        masks, scores, logits, *_ = self.model.predict(
            point_coords=point_coordinates,
            point_labels=point_labels,
            mask_input=logits[best_idx : best_idx + 1, :, :],
            multimask_output=True,
        )
        best_idx = np.argmax(scores)

        # Cascaded Post-refinement-2
        y, x = np.nonzero(masks[best_idx])
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        input_box = np.array([x_min, y_min, x_max, y_max])
        masks, scores, logits, *_ = self.model.predict(
            point_coords=point_coordinates,
            point_labels=point_labels,
            box=input_box[None, :],
            mask_input=logits[best_idx : best_idx + 1, :, :],
            multimask_output=True,
        )
        best_idx = np.argmax(scores)
        final_mask = masks[best_idx]
        final_score = scores[best_idx]
        return final_mask, masks, final_score

    def predict_masks(
        self,
        point_prompt_candidates: np.array,
        bg_points: np.array,
        apply_masks_refinement: bool = True,
        attn_sim: torch.Tensor | None = None,
        target_guided_embedding: torch.Tensor | None = None,
        class_idx: int = 0,
    ):
        """
        Generate masks based on the provided point prompts in one go
        """
        # first remove all points that have a score of -1.0 or 0.0
        point_prompt_candidates = [
            p for p in point_prompt_candidates if p[2] not in [-1.0, 0.0]
        ]
        # combine the background points with the foreground points, use label 1 for foreground and 0 for background
        point_coords = np.array([p[:2] for p in point_prompt_candidates] + bg_points)
        point_labels = np.array(
            [1] * len(point_prompt_candidates) + [0] * len(bg_points), dtype=np.float32
        )

        if isinstance(self.model, SamPredictor):
            masks, scores, low_res_logits, *_ = self.model.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
                attn_sim=attn_sim,
                target_embedding=target_guided_embedding,
            )
        elif isinstance(self.model, EfficientViTSamPredictor):
            masks, scores, low_res_logits = self.model.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
        else:
            raise NotImplementedError("Model not supported")

        if apply_masks_refinement:
            final_mask, masks, _ = self.post_refinement(
                low_res_logits, point_coords, point_labels
            )
        else:
            best_idx = np.argmax(scores)
            final_mask = masks[best_idx]

        return [final_mask], point_prompt_candidates

    def predict_masks_point_by_point(
        self,
        point_prompt_candidates: np.array,
        bg_points: np.array,
        apply_masks_refinement: bool = True,
        attn_sim: torch.Tensor | None = None,
        target_guided_embedding: torch.Tensor | None = None,
    ):
        """
        Generate masks based on the provided point prompts. This method predicts a mask for each point but filters out points that lie in a previously found mask.
        It does not supply the decoder with multiple points at the same time. This is the implementation in ModelAPI.

        Args:
            point_prompt_candidates: list of point prompts (x, y, score)
            bg_points: list of background points (x, y, score)
            apply_masks_refinement: whether to apply masks refinement
            attn_sim: similarity tensor for target-guided attention
            target_guided_embedding: target embedding for target-guided attention

        Returns:
            all_masks: list of masks
            final_point_prompts: list of used point prompts (x, y, score)
        """
        final_point_prompts = []
        all_masks = []

        for i, (x, y, score) in enumerate(point_prompt_candidates):
            # remove points with very low confidence
            if score in [-1.0, 0.0]:
                continue
            # filter out points that lie inside a previously found mask
            is_done = False
            for predicted_mask in all_masks:
                if predicted_mask[int(y), int(x)] > 0:
                    is_done = True
                    break
            if is_done:
                continue

            point_coords = np.concatenate(
                (np.array([[x, y]]), bg_points),
                axis=0,
                dtype=np.float32,
            )
            point_labels = np.array([1] + [0] * len(bg_points), dtype=np.float32)

            # predict the masks based on the points
            if isinstance(self.model, SamPredictor):
                masks, scores, low_res_logits, *_ = self.model.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False,
                    attn_sim=attn_sim,  # Target-guided Attention (not in model api)
                    target_embedding=target_guided_embedding,  # Target-semantic Prompting (not in model api)
                )
            elif isinstance(self.model, EfficientViTSamPredictor):
                masks, scores, low_res_logits = self.model.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False,
                )
            else:
                raise NotImplementedError("Model not supported")

            # Cascaded post refinement of the masks (disabled by default in Geti VPS)
            if not apply_masks_refinement:
                best_idx = np.argmax(scores)
                final_mask = masks[best_idx]
            else:
                final_mask, masks, scores = self.post_refinement(
                    low_res_logits, point_coords, point_labels
                )

            all_masks.append(final_mask)
            final_point_prompts.append([x, y, score])

        return all_masks, final_point_prompts


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
    reference_features: torch.Tensor, n_clusters: int = 8
) -> torch.Tensor:
    """Create part-level features from reference features.
    This performs a k-means++ clustering on the reference features and takes the centroid as prototype.
    Resulting part-level features are normalized to unit length.
    If n_clusters is 1, the mean of the reference features is taken as prototype.

    Args:
        reference_features: Reference features tensor [X, 256]
        n_clusters: Number of clusters to create (e.g. number of part-level-features)

    Returns:
        Part-level features tensor [n_clusters, 256]
    """
    if n_clusters == 1:
        part_level_features = reference_features.mean(0).unsqueeze(0)
        part_level_features = part_level_features / part_level_features.norm(
            dim=-1, keepdim=True
        )
        return part_level_features.unsqueeze(0)  # 1, 256

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
    return part_level_features


def load_model(
    sam_name="SAM", algo_name="Personalized SAM"
) -> PerSamPredictor | SAMLearnableVisualPrompter:
    if sam_name not in MODEL_MAP:
        raise ValueError(f"Invalid model type: {sam_name}")

    name, checkpoint_path = MODEL_MAP[sam_name]
    if sam_name in ["SAM", "MobileSAM"]:
        backbone = sam_model_registry[name](checkpoint=checkpoint_path).cuda()
        backbone.eval()
        backbone = SamPredictor(backbone)
        return PerSamPredictor(backbone, algo_name)
    elif sam_name == "EfficientViT-SAM":
        backbone = create_efficientvit_sam_model(
            name=name, weight_url=checkpoint_path
        ).cuda()
        backbone.eval()
        backbone = EfficientViTSamPredictor(backbone)
        return PerSamPredictor(backbone, algo_name)
    elif sam_name == "MobileSAM-MAPI":
        encoder = Model.create_model(MAPI_ENCODER_PATH)
        decoder = Model.create_model(MAPI_DECODER_PATH)
        return SAMLearnableVisualPrompter(encoder, decoder)
    else:
        raise NotImplementedError(f"Model {sam_name} not implemented yet")
