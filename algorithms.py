from collections import defaultdict
import random
from typing import Optional

import cv2
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import ot

from Matcher.segment_anything.automatic_mask_generator import (
    SamAutomaticMaskGenerator as MatcherSamAutomaticMaskGenerator,
)
from dinov2.data.transforms import MaybeToTensor, make_normalize_transform
from dinov2.models.vision_transformer import DinoVisionTransformer
from efficientvit.models.efficientvit import EfficientViTSamPredictor, SamResize, SamPad
from PersonalizeSAM.per_segment_anything import SamPredictor
import torch
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
from scipy.optimize import linear_sum_assignment

from model_api.models import Prompt, PredictedMask, ZSLVisualPromptingResult
from model_api.models.visual_prompting import (
    VisualPromptingFeatures,
    _inspect_overlapping_areas,
    _point_selection,
)
from utils.utils import (
    cluster_features,
    cluster_points,
    is_in_mask,
    prepare_target_guided_prompting,
    sample_points,
    similarity_maps_to_visualization,
    transform_point_prompts_to_dict,
    transform_mask_prompts_to_dict,
)


class PerSamPredictor:
    """Wrapper on top of the SAM model to make it inline with ModelAPI SAM Interface"""

    def __init__(self, sam_model: SamPredictor):
        self.grid_size = 64
        self.num_bg_points = 1
        self.threshold = 0.65
        self.sam_model: SamPredictor = sam_model
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
    ) -> tuple[VisualPromptingFeatures, np.array, Optional[Figure]]:
        mask_per_class = self.prepare_input(image, masks, points, boxes, polygons)

        for class_idx, mask in mask_per_class.items():
            image_embedding, reference_features = self.extract_reference_features(
                mask, image
            )
            if reference_features is None:
                continue

            # PerSAM: use average of all features.
            # P2SAM/PartAware: use k-means++ clustering to create num_clusters part-level features
            reference_features, umap_fig = cluster_features(
                reference_features, num_clusters, visualize=show
            )

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

        return (
            VisualPromptingFeatures(
                feature_vectors=np.stack(
                    [v.cpu().numpy() for v in self.reference_features.values()]
                ),
                used_indices=np.array(self.reference_features.keys()),
            ),
            np.stack(list(self.reference_masks.values())),
            umap_fig,
        )

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
        # tesst_feat has shape 256 64 64
        test_feat = test_feat.reshape(c, h * w)

        for class_idx, reference_features in self.reference_features.items():
            # Cosine similarity
            sim = (
                reference_features @ test_feat
            )  # (1,1,256) @ (256, 64*64)  ->  (1,1,64*64)
            sim = sim.reshape(reference_features.shape[0], 1, h, w)  # (1,1,64,64)

            sim = F.interpolate(sim, scale_factor=4, mode="bilinear")  # 1, 1, 256, 256
            sim = self.sam_model.model.postprocess_masks(
                sim,
                input_size=self.sam_model.input_size,
                original_size=self.sam_model.original_size,
            ).squeeze()  # 1280, 1280
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
                            self.sam_model.original_size
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
                    original_shape=np.array(self.sam_model.original_size),
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
                self.sam_model.set_image(image)
                masks, scores, logits, *_ = self.sam_model.predict(
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
        self.sam_model.set_image(image)
        return self.sam_model.features.squeeze()

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
        if isinstance(self.sam_model, SamPredictor):
            # set_image resizes and pads to square input
            # TODO set_image also computes image embedding which is now performed for every mask and not just once
            # TODO this is not efficient and should only resize the mask. However, in practice we have only mask per class
            reference_mask = self.sam_model.set_image(image, mask)  # 1, 3 ,1024, 1024
            image_embedding = self.sam_model.features.squeeze().permute(
                1, 2, 0
            )  # 64, 64, 256
            # transform reference mask to a 64 64 mask
            reference_mask = F.interpolate(
                reference_mask, size=image_embedding.shape[0:2], mode="bilinear"
            )
            reference_mask = reference_mask.squeeze()[0]  # 64, 64
        elif isinstance(self.sam_model, EfficientViTSamPredictor):
            self.sam_model.set_image(image)
            image_embedding = self.sam_model.features.squeeze().permute(
                1, 2, 0
            )  # 64, 64, 256
            # resize relative to longest size
            reference_mask = SamResize(self.sam_model.model.image_size[0])(
                mask
            )  # 576 1024 3
            reference_mask = torch.as_tensor(
                reference_mask, device=self.sam_model.device
            )
            reference_mask = reference_mask.permute(2, 0, 1).contiguous()[
                None, :, :, :
            ]  # 1, 3, 576, 1024
            reference_mask = SamPad(self.sam_model.model.image_size[0])(
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
        masks, scores, logits, *_ = self.sam_model.predict(
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
        masks, scores, logits, *_ = self.sam_model.predict(
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

        if isinstance(self.sam_model, SamPredictor):
            masks, scores, low_res_logits, *_ = self.sam_model.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
                attn_sim=attn_sim,
                target_embedding=target_guided_embedding,
            )
        elif isinstance(self.sam_model, EfficientViTSamPredictor):
            masks, scores, low_res_logits = self.sam_model.predict(
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

        final_mask = np.where(final_mask, 255, 0).astype(np.uint8)
        final_mask = np.squeeze(final_mask)  # Remove extra dimension

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
            if isinstance(self.sam_model, SamPredictor):
                masks, scores, low_res_logits, *_ = self.sam_model.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False,
                    attn_sim=attn_sim,  # Target-guided Attention (not in model api)
                    target_embedding=target_guided_embedding,  # Target-semantic Prompting (not in model api)
                )
            elif isinstance(self.sam_model, EfficientViTSamPredictor):
                masks, scores, low_res_logits = self.sam_model.predict(
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


class PerDinoPredictor(PerSamPredictor):
    """
    Use DinoV2 based patch features for matching instead of SAM features.
    """

    def __init__(
        self,
        sam_model: SamPredictor,
        dino_model: DinoVisionTransformer,
        alpha: float = 1.0,
        beta: float = 0.0,
        exp: float = 0.0,
        use_box: bool = False,
        sample_range: tuple[int, int] = (4, 6),
        max_sample_iterations: int = 30,
        emd_filter: float = 0.0,
        purity_filter: float = 0.0,
        coverage_filter: float = 0.0,
        use_score_filter: bool = True,
        num_merging_masks: int = 10,
        topk_scores_threshold: float = 0.7,
        deep_score_filter: float = 0.33,
        deep_score_norm_filter: float = 0.1,
    ):
        super().__init__(sam_model)
        self.dino = dino_model

        self.use_box = use_box
        self.sample_range = sample_range
        self.max_sample_iterations = max_sample_iterations
        self.emd_filter = emd_filter
        self.purity_filter = purity_filter
        self.coverage_filter = coverage_filter
        self.use_score_filter = use_score_filter
        self.num_merging_masks = num_merging_masks
        self.topk_scores_threshold = topk_scores_threshold
        self.alpha = alpha
        self.beta = beta
        self.exp = exp
        self.deep_score_filter = deep_score_filter
        self.deep_score_norm_filter = deep_score_norm_filter
        self.image_size = self.dino.patch_embed.img_size[0]
        self.feature_size = self.image_size // self.dino.patch_size
        self.patch_size = self.dino.patch_size

        self.encoder_transform = transforms.Compose(
            [
                MaybeToTensor(),
                transforms.Resize((self.image_size, self.image_size)),
                make_normalize_transform(),
            ]
        )
        self.encoder_mask_transform = transforms.Compose(
            [
                MaybeToTensor(),
                transforms.Resize(self.image_size),
            ]
        )
        # Default values of Matcher
        self.generator = MatcherSamAutomaticMaskGenerator(
            self.sam_model.model,
            points_per_side=64,
            points_per_batch=64,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            stability_score_offset=1.0,
            box_nms_thresh=1.0,
            sel_output_layer=3,
            output_layer=0,
            dense_pred=0,
            multimask_output=False,
            sel_multimask_output=False,
        )

    def learn(
        self,
        image: np.array,
        boxes: list[Prompt] | None = None,
        points: list[Prompt] | None = None,
        polygons: list[Prompt] | None = None,
        masks: list[Prompt] | None = None,
        show: bool = False,
        num_clusters: int = 1,
    ) -> tuple[VisualPromptingFeatures, np.array, Optional[Figure]]:
        mask_per_class = self.prepare_input(image, masks, points, boxes, polygons)

        for class_idx, mask in mask_per_class.items():
            # transform mask and image from np to torch tensors
            features, mask_pooled = self.extract_features(image, mask)
            features = features.detach().cpu()
            mask_pooled = mask_pooled.detach().cpu()

            # Accumulate features to allow for few-shot learning
            # TODO: for production we want to be able to just pass all reference images in one go/batch
            if class_idx in self.reference_features:
                self.reference_features[class_idx] = torch.cat(
                    [self.reference_features[class_idx], features], dim=0
                )
                self.reference_masks[class_idx] = torch.cat(
                    [self.reference_masks[class_idx], mask_pooled], dim=0
                )
            else:
                self.reference_features[class_idx] = features
                self.reference_masks[class_idx] = mask_pooled

        return (
            VisualPromptingFeatures(
                feature_vectors=np.stack(
                    [v.cpu().numpy() for v in self.reference_features.values()]
                ),
                used_indices=np.array(self.reference_features.keys()),
            ),
            np.stack(list(self.reference_masks.values())),
            None,  # visual output not implemented for DinoPredictor
        )

    def infer(
        self,
        image: np.array,
        reference_features=None,
        apply_masks_refinement: bool = True,
        target_guided_attention: bool = False,
        mask_generation_method: str = "point-by-point",
        selection_on_similarity_maps: str = "per-map",
        n_clusters: int = 1,
    ):
        # target features extraction
        target_features, _ = self.extract_features(image)
        all_masks: dict[int, list] = defaultdict(list)
        # self.sam_model.set_image(image)

        # image preprocessing
        original_image_size = (image.shape[0], image.shape[1])
        image = self.encoder_transform(image)
        # prepare for SAM
        image_np = image.mul(255).byte().permute(1, 2, 0).cpu().numpy()

        for class_idx, reference_features in self.reference_features.items():
            # Patch level matching
            reference_masks = self.reference_masks[class_idx]
            all_points, box, sim, cost_matrix, reduced_num_of_points = (
                self.patch_level_matching(
                    reference_features.cuda(),
                    target_features,
                    reference_masks.cuda(),
                )
            )
            # point (subset) selection
            points = (
                cluster_points(all_points, n_clusters) if n_clusters > 1 else all_points
            )

            # Mask generation
            masks = self.generate_masks(
                image_np,
                original_image_size=original_image_size,
                points=points,
                reference_masks=reference_masks,
                all_points=all_points,
                cost_matrix=cost_matrix,
                box=box,
            )
            all_masks[class_idx] = masks

        return ZSLVisualPromptingResult(
            data={
                class_idx: PredictedMask(
                    mask=masks,
                    points=np.array(
                        []
                    ),  # TODO add sampled points and scores found in matching algo
                    scores=np.array([]),
                )
                for class_idx, masks in all_masks.items()
            },
        ), None

    def generate_masks(
        self,
        image: np.array,
        original_image_size: tuple[int, int],
        points: np.array,
        reference_masks: torch.Tensor,
        all_points: np.array,
        cost_matrix: torch.Tensor,
        box: np.ndarray | None = None,
    ) -> np.array:
        # TODO Matcher does not seem to make use of background points

        # (subset) point sampling
        sampled_points, label_list = sample_points(
            points, self.sample_range, self.max_sample_iterations
        )

        # TODO: Matcher uses the original SAM mask generator with integrated filtering and merging.
        # TODO: Do we want to apply mask refinement for each mask?

        tar_masks_ori = self.generator.generate(
            image,
            select_point_coords=sampled_points,
            select_point_labels=label_list,
            select_box=[box] if self.use_box else None,
        )
        # can happen that we have no masks
        if len(tar_masks_ori) == 0:
            return np.zeros(
                (1, original_image_size[0], original_image_size[1]), dtype=np.uint8
            )

        masks = (
            torch.cat(
                [
                    torch.from_numpy(qmask["segmentation"])
                    .float()[None, None, ...]
                    .cuda()
                    for qmask in tar_masks_ori
                ],
                dim=0,
            )
            .cpu()
            .numpy()
            > 0
        )

        # metrics based filtering
        purity = torch.zeros(masks.shape[0])
        coverage = torch.zeros(masks.shape[0])
        emd = torch.zeros(masks.shape[0])

        samples = sampled_points[-1]
        labels = torch.ones(masks.shape[0], samples.shape[1])
        samples = torch.ones(masks.shape[0], samples.shape[1], 2)

        # compute mask scores
        for i in range(len(masks)):
            purity_, coverage_, emd_ = self.compute_mask_scores(
                masks=masks[i],
                reference_masks=reference_masks,
                all_points=all_points,
                emd_cost=cost_matrix,
            )
            purity[i] = purity_
            coverage[i] = coverage_
            emd[i] = emd_

        predicted_masks = masks.squeeze(1)
        mask_metrics = {"purity": purity, "coverage": coverage, "emd": emd}
        scores = self.alpha * emd + self.beta * purity * coverage**self.exp

        # Apply filtering based on metrics, to reduce false positive masks)
        scores, samples, labels, predicted_masks, mask_metrics = (
            self._apply_metric_filters(
                mask_metrics, scores, samples, labels, predicted_masks
            )
        )

        # Merge masks
        if self.use_score_filter:
            final_mask = self._merge_masks_with_score_filter(scores, predicted_masks)
        else:
            final_mask = self._merge_masks_with_topk(scores, samples, predicted_masks)

        # Resize mask back to original size
        final_mask = np.where(final_mask, 255, 0).astype(np.uint8).squeeze()
        final_mask = cv2.resize(
            final_mask,
            (original_image_size[1], original_image_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return final_mask[None, ...]

    def compute_mask_scores(
        self,
        masks: torch.Tensor,
        reference_masks: torch.Tensor,
        all_points: np.array,
        emd_cost: torch.Tensor,
    ):
        original_masks = masks

        # resize mask to patch size, to use per-patch EMD cost
        masks = cv2.resize(
            masks[0].astype(np.float32),
            (self.feature_size, self.feature_size),
            interpolation=cv2.INTER_AREA,
        )
        if masks.max() <= 0:
            thres = masks.max() - 1e-6
        else:
            thres = 0
        masks = masks > thres

        # Earth mover distance between reference mask and predicted mask
        emd_batch = emd_cost[reference_masks.flatten().bool(), :][:, masks.flatten()]
        emd = 1 - ot.emd2(
            a=[1.0 / emd_batch.shape[0] for i in range(emd_batch.shape[0])],
            b=[1.0 / emd_batch.shape[1] for i in range(emd_batch.shape[1])],
            M=emd_batch.cpu().numpy(),
        )

        # Purity and Coverage
        assert all_points is not None
        points_in_mask = is_in_mask(all_points, original_masks[0])
        points_in_mask = all_points[points_in_mask]
        mask_area = max(float(masks.sum()), 1.0)
        purity = torch.tensor([float(points_in_mask.shape[0]) / mask_area]) + 1e-6
        coverage = (
            torch.tensor([float(points_in_mask.shape[0]) / all_points.shape[0]]) + 1e-6
        )
        return purity, coverage, emd

    def patch_level_matching(
        self,
        reference_features: torch.Tensor,
        target_features: torch.Tensor,
        reference_masks: torch.Tensor,
    ) -> tuple[np.array, np.array, torch.Tensor, torch.Tensor, int]:
        """
        Performs matching of the reference features with the target features. Uses DinoV2 based patch level features and is based on the algorithm
        of Matcher (Liu et al. 2024). The algorithm uses a bidirectional approach in which matched target features are backward compared to the actual references.
        This decreases false positives and matching outliers.

        Args:
            reference_features: reference features
            target_features: target features
            reference_masks: reference masks

        Returns:
            points: matched points
            box: matched box
            sim: similarity scores
            cost_matrix: cost matrix
            reduced_num_of_points: reduced number of points
        """
        # Cosine similarity
        sim = reference_features @ target_features.t()
        cost_matrix = (1 - sim) / 2

        # forward matching
        forward_sim = sim[
            reference_masks.flatten().bool()
        ]  # select only the features within the mask
        indices_forward = linear_sum_assignment(forward_sim.cpu(), maximize=True)
        indices_forward = [
            torch.as_tensor(index, dtype=torch.int64, device="cuda")
            for index in indices_forward
        ]
        sim_scores_forward = forward_sim[indices_forward[0], indices_forward[1]]
        non_zero_mask_indices = reference_masks.flatten().nonzero()[:, 0]

        # backward matching
        backward_sim = sim.t()[indices_forward[1]]  # THIS USES (and needs) FULL SIM MAP
        indices_backward = linear_sum_assignment(backward_sim.cpu(), maximize=True)
        indices_backward = [
            torch.as_tensor(index, dtype=torch.int64, device="cuda")
            for index in indices_backward
        ]
        # compare forward and backward indices to filter out indices that do not match
        indices_to_keep = torch.isin(indices_backward[1], non_zero_mask_indices)
        if not (indices_to_keep == False).all().item():
            indices_forward = [
                indices_forward[0][indices_to_keep],
                indices_forward[1][indices_to_keep],
            ]
            sim_scores_forward = sim_scores_forward[indices_to_keep]
        similarity_matched = sim_scores_forward

        # limit number of points
        reduced_num_of_points = (
            len(similarity_matched) // 2
            if len(similarity_matched) > 40
            else len(similarity_matched)
        )

        # rank matches by similarity score, and transform back to image level coordinates
        similarity_sorted, indices_sorted = torch.sort(
            similarity_matched, descending=True
        )
        filtered_indices = indices_sorted[:reduced_num_of_points]
        points_matched_indices = indices_forward[1][filtered_indices]
        points_matched_indices = torch.tensor(
            list(set(points_matched_indices.cpu().tolist()))
        )

        # transform points to image level coordinates
        points_matched_width = points_matched_indices % self.feature_size
        points_matched_height = points_matched_indices // self.feature_size

        points_matched_x = (
            points_matched_width * self.patch_size + self.patch_size // 2
        ).tolist()
        points_matched_y = (
            points_matched_height * self.patch_size + self.patch_size // 2
        ).tolist()
        points = []
        for x, y in zip(points_matched_x, points_matched_y):
            if int(x) < self.image_size and int(y) < self.image_size:
                points.append([x, y])
        points = np.array(points)

        box = None
        if self.use_box:
            box = np.array(
                [
                    max(points[:, 0].min(), 0),
                    max(points[:, 1].min(), 0),
                    min(points[:, 0].max(), self.input_size - 1),
                    min(points[:, 1].max(), self.input_size - 1),
                ]
            )
        return (points, box, sim, cost_matrix, reduced_num_of_points)

    def extract_features(
        self, image: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, None]:
        """Extract patch level features using DinoV2 model.

        Args:
            mask: input mask
            image: input image

        Returns:
            image_embedding: 64, 64, 256
            reference_features: X, 256
        """

        # change to bs, 3, h, w instead of h, w, 3
        # image = image.transpose(2, 0, 1)
        image = self.encoder_transform(image)[None, ...]
        features = self.dino.forward_features(image.cuda())["x_prenorm"][:, 1:].detach()
        features = features.reshape(-1, self.dino.embed_dim)
        # normalize features for cosine similarity
        features = F.normalize(features, dim=1, p=2)

        if mask is not None:
            mask = mask.max(axis=2)
            mask[mask > 0] = 1
            mask = self.encoder_mask_transform(mask).cuda().unsqueeze(0)
            # apply pooling to mask
            mask_pooled = F.avg_pool2d(
                mask, kernel_size=(self.dino.patch_size, self.dino.patch_size)
            )
            # apply mask threshold after pooling, since pooling changes the mask values
            # mask_pooled = (mask_pooled > self.threshold).float()
            return features, mask_pooled

        return features, None

    def _apply_metric_filters(
        self,
        mask_metrics: dict,
        scores: torch.Tensor,
        samples: torch.Tensor,
        labels: torch.Tensor,
        predicted_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Apply filtering based on metrics to reduce false positive masks."""

        def _apply_filter(metric: str, filter_value: float):
            if filter_value > 0:
                threshold = min(mask_metrics[metric].max(), filter_value)
                index = torch.where(mask_metrics[metric] > threshold)
                nonlocal scores, samples, labels, predicted_masks
                scores = scores[index]
                samples = samples[index]
                labels = labels[index]
                predicted_masks = self._check_pred_mask(predicted_masks[index])
                for key in mask_metrics:
                    mask_metrics[key] = mask_metrics[key][index]

        _apply_filter("purity", self.purity_filter)
        _apply_filter("coverage", self.coverage_filter)
        _apply_filter("emd", self.emd_filter)

        return scores, samples, labels, predicted_masks, mask_metrics

    def _merge_masks_with_score_filter(
        self, scores: np.ndarray, predicted_masks: np.ndarray
    ) -> np.ndarray:
        """Merge masks using score-based filtering."""
        distances = 1 - scores
        distances, rank = torch.sort(distances, descending=True)
        distances_norm = (distances - distances.min()) / (distances.max() + 1e6)
        filter_distances = distances < self.deep_score_filter
        filter_distances[..., 0] = True
        filter_distances_norm = distances_norm < self.deep_score_norm_filter
        filter_distances = filter_distances * filter_distances_norm
        predicted_masks = self._check_pred_mask(predicted_masks)
        masks = predicted_masks[rank[filter_distances]]
        masks = self._check_pred_mask(masks)
        masks = masks.sum(0) > 0
        return masks[None, ...]

    def _merge_masks_with_topk(
        self, scores: np.ndarray, samples: np.ndarray, predicted_masks: np.ndarray
    ) -> np.ndarray:
        """Merge masks using top-k approach."""
        topk = min(self.num_merging_masks, scores.size(0))
        topk_idx = scores.topk(topk)[1]
        topk_samples = samples[topk_idx].cpu().numpy()
        topk_scores = scores[topk_idx].cpu().numpy()
        topk_pred_masks = self._check_pred_mask(predicted_masks[topk_idx])

        if self.score_filter_cfg["topk_scores_threshold"] > 0:
            # map scores to 0-1
            topk_scores = topk_scores / (topk_scores.max())

        idx = topk_scores > self.score_filter_cfg["topk_scores_threshold"]
        topk_pred_masks = self._check_pred_mask(topk_pred_masks[idx])

        mask_list = [mask[None, ...] for mask in topk_pred_masks]
        masks = np.sum(mask_list, axis=0) > 0
        return self._check_pred_mask(masks)

    @staticmethod
    def _check_pred_mask(pred_masks: torch.Tensor) -> torch.Tensor:
        """Ensure mask has at least 3 dimensions."""
        if len(pred_masks.shape) < 3:  # avoid only one mask
            pred_masks = pred_masks[None, ...]
        return pred_masks

    # Original method now becomes:
    def filter_and_merge_masks(
        self, mask_metrics, scores, samples, labels, predicted_masks
    ):
        scores, samples, labels, predicted_masks, mask_metrics = (
            self._apply_metric_filters(
                mask_metrics, scores, samples, labels, predicted_masks
            )
        )

        if self.use_score_filter:
            masks = self._merge_masks_with_score_filter(scores, predicted_masks)
        else:
            masks = self._merge_masks_with_topk(scores, samples, predicted_masks)

        return torch.tensor(masks, dtype=torch.float, device="cuda")
