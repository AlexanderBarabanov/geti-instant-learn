import os
from collections import defaultdict

from pandas import Series

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
    _point_selection,
)
from utils import transform_point_prompts_to_dict, transform_mask_prompts_to_dict


def load_model(
    sam_name="SAM", algo_name="Personalized SAM"
) -> SamPredictor | EfficientViTSamPredictor | SAMLearnableVisualPrompter:
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
        self.reference_embedding = None

    def learn(
        self,
        image: np.array,
        boxes: list[Prompt] | None = None,
        points: list[Prompt] | None = None,
        polygons: list[Prompt] | None = None,
        masks: list[Prompt] | None = None,
        show: bool = False,
    ) -> tuple[VisualPromptingFeatures, np.array]:
        # get masks by points
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

        self.model.set_image(image)
        for class_idx, mask in mask_per_class.items():
            if show:
                fig = plt.figure(figsize=(10, 10))
                plt.imshow(mask)
                plt.show()
                cv2.imwrite(f"reference_mask_{class_idx}.jpg", mask)

            if isinstance(self.model, SamPredictor):
                # set_image resizes and pads to square input
                reference_mask = self.model.set_image(image, mask)  # 1, 3 ,1024, 1024
                image_embedding = self.model.features.squeeze().permute(
                    1, 2, 0
                )  # image embedding is 64, 64, 256
                # transform reference mask to a 64 64 mask
                reference_mask = F.interpolate(
                    reference_mask, size=image_embedding.shape[0:2], mode="bilinear"
                )
                reference_mask = reference_mask.squeeze()[0]  # 64, 64
            elif isinstance(self.model, EfficientViTSamPredictor):
                image_embedding = self.model.features.squeeze().permute(
                    1, 2, 0
                )  # image embedding is 64, 64, 256
                # resize relative to longest size
                reference_mask = SamResize(self.model.model.image_size[0])(
                    mask
                )  # 576 1024 3
                reference_mask = torch.as_tensor(
                    reference_mask, device=self.model.device
                )
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
                continue

            ### NOTE: this averages the local features into one vector (same as in ModelAPI)
            ### For P2-SAM we want to keep each feature separately
            ### For a Vision-RAG implementation we probably want to store the features (or its average) in a vector DB, while keeping a
            ### maximum num of items and removing outliers to keep a good representation of the target class
            reference_embedding = reference_features.mean(0).unsqueeze(0)
            reference_features = reference_embedding / reference_embedding.norm(
                dim=-1, keepdim=True
            )
            self.reference_embedding = reference_embedding.unsqueeze(0)
            print(
                f"Using {reference_features.shape} reference features for class {class_idx}"
            )
            self.reference_features[class_idx] = (
                reference_features  # note: this is still in CUDA
            )
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

    def infer(
        self,
        target_image,
        reference_features=None,
        apply_masks_refinement: bool = True,
        dev: bool = False,
    ) -> ZSLVisualPromptingResult:
        prediction: dict[int, PredictedMask] = {}
        final_point_prompts: dict[int, list] = defaultdict(list)
        all_point_prompt_candidates: dict[int, np.ndarray] = {}
        all_masks: dict[int, list] = defaultdict(list)
        all_scores: dict[int, list] = defaultdict(list)
        all_bg_prompts: dict[int, list] = defaultdict(list)
        sim_masks_per_class = {}
        # Image feature encoding
        self.model.set_image(target_image)
        test_feat = self.model.features.squeeze()

        # Cosine similarity
        c, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(c, h * w)
        for class_idx, reference_features in self.reference_features.items():
            sim = reference_features @ test_feat
            sim = sim.reshape(1, 1, h, w)
            sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
            sim = self.model.model.postprocess_masks(
                sim,
                input_size=self.model.input_size,
                original_size=self.model.original_size,
            ).squeeze()
            sim_masks_per_class[class_idx] = sim

            # model_api based point selection (multi object and using grid approach)
            point_prompt_candidates, bg_points = _point_selection(
                mask_sim=sim.cpu().numpy(),  # numpy  H W  720 1280
                original_shape=np.array(self.model.original_size),  # [ 720  1280]
                threshold=self.threshold,
                num_bg_points=self.num_bg_points,
                image_size=self.model.input_size[1],  # 1024
                downsizing=self.grid_size,
            )
            all_point_prompt_candidates[class_idx] = point_prompt_candidates
            all_bg_prompts[class_idx] = bg_points

        # filter points
        for class_idx in self.reference_features.keys():
            point_prompt_candidates = all_point_prompt_candidates[class_idx]
            bg_points = all_bg_prompts[class_idx]

            # Obtain the target guidance for cross-attention layers
            sim = sim_masks_per_class[class_idx]
            sim = (sim - sim.mean()) / torch.std(sim)
            sim = F.interpolate(
                sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear"
            )
            attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)

            for i, (x, y, score) in enumerate(point_prompt_candidates):
                # remove points with very low confidence
                if score in [-1.0, 0.0]:
                    continue
                # filter out points that lie inside a previously found mask
                is_done = False
                for predicted_mask in all_masks.get(class_idx, []):
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
                        target_embedding=self.reference_features[
                            class_idx
                        ],  # Target-semantic Prompting (not in model api)
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

                all_masks[class_idx].append(final_mask)
                all_scores[class_idx].append(scores)
                final_point_prompts[class_idx].append([x, y])

        # TODO check overlapping areas of label masks ( see model_api ._inspect_overlapping_areas)

        for label in final_point_prompts:
            prediction[label] = PredictedMask(
                mask=all_masks[label],
                points=final_point_prompts[label],
                scores=all_scores[label],
            )

        if dev:
            return ZSLVisualPromptingResult(prediction), None
        return ZSLVisualPromptingResult(prediction)

    def post_refinement(self, logits, topk_xy, topk_label):
        best_idx = 0
        # Cascaded Post-refinement-1
        masks, scores, logits, *_ = self.model.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
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
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logits[best_idx : best_idx + 1, :, :],
            multimask_output=True,
        )
        best_idx = np.argmax(scores)
        final_mask = masks[best_idx]
        final_score = scores[best_idx]
        return final_mask, masks, final_score


def run_per_segment_anything(
    model: SamPredictor,
    sample: Series,
    output_root: str,
    post_refinement=True,
    show=False,
    save=False,
) -> tuple[plt.Figure, float, float, float]:
    ref_image = cv2.cvtColor(cv2.imread(sample.reference_image), cv2.COLOR_BGR2RGB)
    ref_mask = cv2.cvtColor(cv2.imread(sample.reference_mask), cv2.COLOR_BGR2RGB)
    test_image = cv2.cvtColor(cv2.imread(sample.target_image), cv2.COLOR_BGR2RGB)
    gt_mask = cv2.cvtColor(cv2.imread(sample.target_mask), cv2.COLOR_BGR2RGB)

    # Image features encoding
    ref_mask = model.set_image(ref_image, ref_mask)
    ref_feat = model.features.squeeze().permute(1, 2, 0)

    ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0:2], mode="bilinear")
    ref_mask = ref_mask.squeeze()[0]

    # Target feature extraction
    #  TODO original code mixes the terms target and test. Need to clarify
    target_feat = ref_feat[ref_mask > 0]
    target_embedding = target_feat.mean(0).unsqueeze(0)
    target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
    target_embedding = target_embedding.unsqueeze(0)

    # Image feature encoding
    model.set_image(test_image)
    test_feat = model.features.squeeze()

    # Cosine similarity
    C, h, w = test_feat.shape
    test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
    test_feat = test_feat.reshape(C, h * w)
    sim = target_feat @ test_feat

    sim = sim.reshape(1, 1, h, w)
    sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
    sim = model.model.postprocess_masks(
        sim, input_size=model.input_size, original_size=model.original_size
    ).squeeze()

    # Positive-negative location prior
    topk_xy_i, topk_label_i, last_xy_i, last_label_i = (
        PersonalizeSAM.persam.point_selection(sim, topk=1)
    )
    topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
    topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

    # Obtain the target guidance for cross-attention layers
    sim = (sim - sim.mean()) / torch.std(sim)
    sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
    attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)

    # First-step prediction
    masks, scores, logits, *_ = model.predict(
        point_coords=topk_xy,
        point_labels=topk_label,
        multimask_output=False,
        attn_sim=attn_sim,  # Target-guided Attention
        target_embedding=target_embedding,  # Target-semantic Prompting
    )

    if not post_refinement:
        best_idx = np.argmax(scores)
        final_mask = masks[best_idx]
    else:
        best_idx = 0

        # Cascaded Post-refinement-1
        masks, scores, logits, *_ = model.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
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
        masks, scores, logits, *_ = model.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logits[best_idx : best_idx + 1, :, :],
            multimask_output=True,
        )
        best_idx = np.argmax(scores)
        final_mask = masks[best_idx]

    # compute IOU
    final_mask = np.uint8(final_mask > 0)
    gt_mask = np.uint8(gt_mask[:, :, 0] > 0)
    intersection, union, area_target = intersectionAndUnion(final_mask, gt_mask)
    sample_iou = intersection / (union + 1e-10)
    sample_accuracy = intersection / (area_target + 1e-10)

    if save:
        # Save and show masks
        output_path = os.path.join(output_root, sample.obj_name)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(test_image)
        PersonalizeSAM.show.show_mask(masks[best_idx], plt.gca())
        PersonalizeSAM.show.show_points(topk_xy, topk_label, plt.gca())
        plt.title(
            f"Mask {best_idx}, iou {sample_iou:.2f}, acc {sample_accuracy:.2f}",
            fontsize=18,
        )
        plt.axis("off")
        if save:
            filename_without_ext = sample.file_name.split(".")[0]
            vis_mask_output_path = os.path.join(
                output_path, f"vis_mask_{filename_without_ext}.jpg"
            )
            with open(vis_mask_output_path, "wb") as outfile:
                plt.savefig(outfile, format="jpg")
        if show:
            plt.show()

        return fig, intersection, union, area_target
    else:
        return None, intersection, union, area_target


def run_p2sam(
    model: SamPredictor,
    sample: Series,
    output_root: str,
    post_refinement=True,
    show=False,
    save=False,
) -> tuple[plt.Figure, float, float, float]:
    pass


def finetune_persam_F(
    predictor: SamPredictor, images_path, annotations_path, output_path
) -> SamPredictor:
    pass
