#
# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations  # TODO: remove when Python3.9 support is dropped

import time
from collections import defaultdict
from itertools import product
from typing import Any, NamedTuple, List

import cv2
import numpy as np
from sklearn.cluster import KMeans
import ot

from model_api.models import (
    PredictedMask,
    SAMDecoder,
    SAMImageEncoder,
    VisualPromptingResult,
    ZSLVisualPromptingResult,
)


class VisualPromptingFeatures(NamedTuple):
    feature_vectors: np.ndarray
    used_indices: np.ndarray


class Prompt(NamedTuple):
    data: np.ndarray
    label: int | np.ndarray


class SAMVisualPrompter:
    """A wrapper that implements SAM Visual Prompter.

    Segmentation results can be obtained by calling infer() method
    with corresponding parameters.
    """

    def __init__(
        self,
        encoder_model: SAMImageEncoder,
        decoder_model: SAMDecoder,
    ):
        self.encoder = encoder_model
        self.decoder = decoder_model

    def infer(
        self,
        image: np.ndarray,
        boxes: list[Prompt] | None = None,
        points: list[Prompt] | None = None,
    ) -> VisualPromptingResult:
        """Obtains segmentation masks using given prompts.

        Args:
            image (np.ndarray): HWC-shaped image
            boxes (list[Prompt] | None, optional): Prompts containing bounding boxes (in XYXY torchvision format)
              and their labels (ints, one per box). Defaults to None.
            points (list[Prompt] | None, optional): Prompts containing points (in XY format)
              and their labels (ints, one per point). Defaults to None.

        Returns:
            VisualPromptingResult: result object containing predicted masks and aux information.
        """
        if boxes is None and points is None:
            msg = "boxes or points prompts are required for inference"
            raise RuntimeError(msg)

        outputs: list[dict[str, Any]] = []

        processed_image, meta = self.encoder.preprocess(image)
        image_embeddings = self.encoder.infer_sync(processed_image)
        processed_prompts = self.decoder.preprocess(
            {
                "bboxes": [box.data for box in boxes] if boxes else None,
                "points": [point.data for point in points] if points else None,
                "labels": {
                    "bboxes": [box.label for box in boxes] if boxes else None,
                    "points": [point.label for point in points] if points else None,
                },
                "orig_size": meta["original_shape"][:2],
            },
        )

        for prompt in processed_prompts:
            label = prompt.pop("label")
            prompt.update(**image_embeddings)

            prediction = self.decoder.infer_sync(prompt)
            prediction["scores"] = prediction["iou_predictions"]
            prediction["labels"] = label
            processed_prediction = self.decoder.postprocess(prediction, meta)

            hard_masks, scores, logits = (
                np.expand_dims(processed_prediction["hard_prediction"], 0),
                processed_prediction["iou_predictions"],
                processed_prediction["low_res_masks"],
            )
            _, mask, best_iou, _ = _decide_masks(hard_masks, logits, scores)
            processed_prediction["processed_mask"] = mask
            processed_prediction["best_iou"] = best_iou

            outputs.append(processed_prediction)

        return VisualPromptingResult(
            upscaled_masks=[item["upscaled_masks"] for item in outputs],
            processed_mask=[item["processed_mask"] for item in outputs],
            low_res_masks=[item["low_res_masks"] for item in outputs],
            iou_predictions=[item["iou_predictions"] for item in outputs],
            scores=[item["scores"] for item in outputs],
            labels=[item["labels"] for item in outputs],
            hard_predictions=[item["hard_prediction"] for item in outputs],
            soft_predictions=[item["soft_prediction"] for item in outputs],
            best_iou=[item["best_iou"] for item in outputs],
        )

    def __call__(
        self,
        image: np.ndarray,
        boxes: list[Prompt] | None = None,
        points: list[Prompt] | None = None,
    ) -> VisualPromptingResult:
        """A wrapper of the SAMVisualPrompter.infer() method"""
        return self.infer(image, boxes, points)


class SAMLearnableVisualPrompter:
    """A wrapper that provides ZSL Visual Prompting workflow.
    To obtain segmentation results, one should run learn() first to obtain the reference features,
    or use previously generated ones.
    """

    def __init__(
        self,
        encoder_model: SAMImageEncoder,
        decoder_model: SAMDecoder,
        reference_features: VisualPromptingFeatures | None = None,
        threshold: float = 0.65,
    ):
        """Initializes ZSL pipeline.

        Args:
            encoder_model (SAMImageEncoder): initialized decoder wrapper
            decoder_model (SAMDecoder): initialized encoder wrapper
            reference_features (VisualPromptingFeatures | None, optional): Previously generated reference features.
                Once the features are passed, one can skip learn() method, and start predicting masks right away.
                Defaults to None.
            threshold (float, optional): Threshold to match vs reference features on infer(). Greater value means a
            stricter matching. Defaults to 0.65.
        """
        self.encoder = encoder_model
        self.decoder = decoder_model
        self._used_indices: np.ndarray | None = None
        self._reference_features: np.ndarray | None = None
        self._reference_masks: np.ndarray | None = None

        if reference_features is not None:
            self._reference_features = reference_features.feature_vectors
            self._used_indices = reference_features.used_indices

        self._point_labels_box = np.array([[2, 3]], dtype=np.float32)
        self._has_mask_inputs = [np.array([[0.0]]), np.array([[1.0]])]

        self._is_cascade: bool = False
        if 0 <= threshold <= 1:
            self._threshold: float = threshold
        else:
            msg = "Confidence threshold should belong to [0;1] range."
            raise ValueError(msg)
        self._num_bg_points: int = 1
        self._default_threshold_target: float = 0.0
        self._image_size: int = self.encoder.image_size
        self._grid_size: int = 64
        self._default_threshold_reference: float = 0.3

    def has_reference_features(self) -> bool:
        """Checks if reference features are stored in the object state."""
        return self._reference_features is not None and self._used_indices is not None

    @property
    def reference_features(self) -> VisualPromptingFeatures:
        """Property represents reference features. An exception is thrown if called when
        the features are not presented in the internal object state.
        """
        if self.has_reference_features():
            return VisualPromptingFeatures(
                np.copy(self._reference_features),
                np.copy(self._used_indices),
            )

        msg = "Reference features are not generated"
        raise RuntimeError(msg)

    def learn(
        self,
        image: np.ndarray,
        boxes: list[Prompt] | None = None,
        points: list[Prompt] | None = None,
        polygons: list[Prompt] | None = None,
        masks: list[Prompt] | None = None,
        reset_features: bool = False,
        perform_averaging: bool = True,
        show: bool = False,
    ) -> tuple[VisualPromptingFeatures, np.ndarray]:
        """Executes `learn` stage of SAM ZSL pipeline.

        Reference features are updated according to newly arrived prompts.
        Features corresponding to the same labels are overridden during
        consequent learn() calls.

        Args:
            image (np.ndarray): HWC-shaped image
            boxes (list[Prompt] | None, optional): Prompts containing bounding boxes (in XYXY torchvision format)
              and their labels (ints, one per box). Defaults to None.
            points (list[Prompt] | None, optional): Prompts containing points (in XY format)
              and their labels (ints, one per point). Defaults to None.
            polygons: (list[Prompt] | None): Prompts containing polygons (a sequence of points in XY format)
              and their labels (ints, one per polygon).
              Polygon prompts are used to mask out the source features without implying decoder usage. Defaults to None.
            masks: (list[Prompt] | None): Prompts containing masks (grid like masks)
              and their labels (ints, one per mask).
              Polygon prompts are used to mask out the source features without implying decoder usage. Defaults to None.
            reset_features (bool, optional): Forces learning from scratch. Defaults to False.
            perform_averaging (bool, optional): If True, the reference features are averaged, reducing dimensions.

        Returns:
            tuple[VisualPromptingFeatures, np.ndarray]: return values are the updated VPT reference features and
                reference masks.
            The shape of the reference mask is N_labels x H x W, where H and W are the same as in the input image.
        """
        if boxes is None and points is None and polygons is None and masks is None:
            msg = "boxes, polygons or points prompts are required for learning"
            raise RuntimeError(msg)

        if reset_features or not self.has_reference_features():
            self.reset_reference_info()

        processed_prompts = self.decoder.preprocess(
            {
                "bboxes": [box.data for box in boxes] if boxes else None,
                "points": [point.data for point in points] if points else None,
                "labels": {
                    "bboxes": [box.label for box in boxes] if boxes else None,
                    "points": [point.label for point in points] if points else None,
                },
                "orig_size": image.shape[:2],
            },
        )

        if polygons is not None:
            for poly in polygons:
                processed_prompts.append({"polygon": poly.data, "label": poly.label})

        if masks is not None:
            for mask in masks:
                processed_prompts.append({"masks": mask.data[:, :, 0], "label": mask.label})

        processed_prompts_w_labels = self._gather_prompts_with_labels(processed_prompts)
        largest_label: int = max([int(p) for p in processed_prompts_w_labels] + [0])

        self._expand_reference_info(largest_label)

        original_shape = np.array(image.shape[:2])

        # forward image encoder
        image_embeddings = self.encoder(image)
        processed_embedding = image_embeddings.squeeze().transpose(1, 2, 0)

        # get reference masks
        ref_masks: np.ndarray = np.zeros(
            (largest_label + 1, *original_shape),
            dtype=np.uint8,
        )
        for label, input_prompts in processed_prompts_w_labels.items():
            ref_mask: np.ndarray = np.zeros(original_shape, dtype=np.uint8)
            for inputs_decoder in input_prompts:
                inputs_decoder.pop("label")
                if "masks" in inputs_decoder:
                    masks = inputs_decoder["masks"]
                elif "point_coords" in inputs_decoder:
                    # bboxes and points
                    inputs_decoder["image_embeddings"] = image_embeddings
                    prediction = self._predict_masks(
                        inputs_decoder,
                        original_shape,
                        is_cascade=self._is_cascade,
                    )
                    masks = prediction["upscaled_masks"]
                elif "polygon" in inputs_decoder:
                    masks = _polygon_to_mask(inputs_decoder["polygon"], *original_shape)
                else:
                    msg = "Unsupported type of prompt"
                    raise RuntimeError(msg)

                # direct input masks have shape w, h, 3
                # computed from polygons have shape: w, h
                ref_mask = np.where(masks, 1, ref_mask)

            ref_feat: np.ndarray | None = None
            cur_default_threshold_reference = self._default_threshold_reference

            while ref_feat is None:
                ref_feat = _generate_masked_features(
                    feats=processed_embedding,
                    masks=ref_mask,
                    threshold_mask=cur_default_threshold_reference,
                    image_size=self.encoder.image_size,
                    perform_averaging=perform_averaging,
                )
                cur_default_threshold_reference -= 0.05

            if self._reference_features is not None:
                self._reference_features[label] = ref_feat
            self._used_indices = np.concatenate((self._used_indices, [label]))
            ref_masks[label] = ref_mask

        self._used_indices = np.unique(self._used_indices)

        # transform the ref masks from (720,1280,3,1)  to (720,1280)
        return self.reference_features, ref_masks

    def __call__(
        self,
        image: np.ndarray,
        reference_features: VisualPromptingFeatures | None = None,
        apply_masks_refinement: bool = True,
    ) -> ZSLVisualPromptingResult:
        """A wrapper of the SAMLearnableVisualPrompter.infer() method"""
        return self.infer(image, reference_features, apply_masks_refinement)

    def infer(
        self,
        image: np.ndarray,
        reference_features: VisualPromptingFeatures | None = None,
        apply_masks_refinement: bool = True,
        dev: bool = False,
    ) -> ZSLVisualPromptingResult | tuple[ZSLVisualPromptingResult, dict]:
        """Obtains masks by already prepared reference features.

        Reference features can be obtained with SAMLearnableVisualPrompter.learn() and passed as an argument.
        If the features are not passed, instance internal state will be used as a source of the features.

        Args:
            image (np.ndarray): HWC-shaped image
            reference_features (VisualPromptingFeatures | None, optional): Reference features object obtained during
                previous learn() calls. If not passed, object internal state is used, which reflects the last learn()
                call. Defaults to None.
            apply_masks_refinement (bool, optional): Flag controlling additional refinement stage on inference.
            Once enabled, decoder will be launched 2 extra times to refine the masks obtained with the first decoder
            call. Defaults to True.

        Returns:
            ZSLVisualPromptingResult: Mapping label -> predicted mask. Each mask object contains a list of binary masks,
                and a list of related prompts. Each binary mask corresponds to one prompt point. Class mask can be
                obtained by applying OR operation to all mask corresponding to one label.
        """
        if reference_features is None:
            if self._reference_features is None:
                msg = (
                    "Reference features are not defined. This parameter can be passed via "
                    "SAMLearnableVisualPrompter constructor, or as an argument of infer() method"
                )
                raise RuntimeError(msg)
            reference_feats = self._reference_features

            if self._used_indices is None:
                msg = (
                    "Used indices are not defined. This parameter can be passed via "
                    "SAMLearnableVisualPrompter constructor, or as an argument of infer() method"
                )
                raise RuntimeError(msg)
            used_idx = self._used_indices
        else:
            reference_feats, used_idx = reference_features

        original_shape = np.array(image.shape[:2])
        image_embeddings = self.encoder(image)

        total_points_scores, total_bg_coords = _get_prompt_candidates(
            image_embeddings=image_embeddings,
            reference_feats=reference_feats,
            used_indices=used_idx,
            original_shape=original_shape,
            threshold=self._threshold,
            num_bg_points=self._num_bg_points,
            default_threshold_target=self._default_threshold_target,
            image_size=self._image_size,
            downsizing=self._grid_size,
        )

        predicted_masks: dict[int, list] = defaultdict(list)
        used_points: defaultdict[int, list] = defaultdict(list)
        for label in total_points_scores:
            points_scores = total_points_scores[label]
            bg_coords = total_bg_coords[label]
            for idx, points_score in enumerate(points_scores):
                # remove points with very low confidence
                if points_score[-1] in [-1.0, 0.0]:
                    continue

                x, y = points_score[:2]
                # TODO: for P2SAM we actually want multiple output prompts, one per part-level feature/cluster
                is_done = False
                for pm in predicted_masks.get(label, []):
                    # check if that point is within the current predicted mask
                    if pm[int(y), int(x)] > 0:
                        is_done = True
                        break
                if is_done:
                    continue

                point_coords = np.concatenate(
                    (np.array([[x, y]]), bg_coords),
                    axis=0,
                    dtype=np.float32,
                )
                point_coords = self.decoder.apply_coords(point_coords, original_shape)
                point_labels = np.array([1] + [0] * len(bg_coords), dtype=np.float32)
                inputs_decoder = {
                    "point_coords": point_coords[None],
                    "point_labels": point_labels[None],
                    "orig_size": original_shape[None],
                }
                inputs_decoder["image_embeddings"] = image_embeddings

                _prediction: dict[str, np.ndarray] = self._predict_masks(
                    inputs_decoder,
                    original_shape,
                    apply_masks_refinement,
                )
                _prediction.update({"scores": points_score[-1]})

                predicted_masks[label].append(_prediction[self.decoder.output_blob_name])
                used_points[label].append(points_score)

        # check overlapping area between different label masks
        _inspect_overlapping_areas(predicted_masks, used_points)

        prediction: dict[int, PredictedMask] = {}
        for k in used_points:
            processed_points = []
            scores = []
            for pt in used_points[k]:
                processed_points.append(pt[:2])
                scores.append(float(pt[2]))
            prediction[k] = PredictedMask(predicted_masks[k], processed_points, scores)
        if dev:
            return ZSLVisualPromptingResult(prediction), {}
        return ZSLVisualPromptingResult(prediction)

    def reset_reference_info(self) -> None:
        """Initialize reference information."""
        self._reference_features = np.zeros(
            (0, 1, self.decoder.embed_dim),
            dtype=np.float32,
        )
        self._used_indices = np.array([], dtype=np.int64)

    def _gather_prompts_with_labels(
        self,
        image_prompts: list[dict[str, np.ndarray]],
    ) -> dict[int, list[dict[str, np.ndarray]]]:
        """Gather prompts according to labels."""
        processed_prompts: defaultdict[int, list[dict[str, np.ndarray]]] = defaultdict(
            list,
        )
        for prompt in image_prompts:
            processed_prompts[int(prompt["label"])].append(prompt)

        return dict(sorted(processed_prompts.items(), key=lambda x: x))

    def _expand_reference_info(self, new_largest_label: int) -> None:
        """Expand reference info dimensions if newly given processed prompts have more labels."""
        if self._reference_features is None:
            msg = "Can not expand non existing reference info"
            raise RuntimeError(msg)

        if new_largest_label > (cur_largest_label := len(self._reference_features) - 1):
            diff = new_largest_label - cur_largest_label
            self._reference_features = np.pad(
                self._reference_features,
                ((0, diff), (0, 0), (0, 0)),
                constant_values=0.0,
            )

    def _predict_masks(
        self,
        inputs: dict[str, np.ndarray],
        original_size: np.ndarray,
        is_cascade: bool = False,
    ) -> dict[str, np.ndarray]:
        """Process function of OpenVINO Visual Prompting Inferencer."""
        masks: np.ndarray
        logits: np.ndarray
        scores: np.ndarray
        num_iter = 3 if is_cascade else 1

        #   since post refinement
        for i in range(num_iter):
            if i == 0:
                # First-step prediction
                mask_input = np.zeros(
                    (1, 1, *(x * 4 for x in inputs["image_embeddings"].shape[2:])),
                    dtype=np.float32,
                )
                has_mask_input = self._has_mask_inputs[0]

            elif i == 1:
                # Cascaded Post-refinement-1
                mask_input, masks, _, _ = _decide_masks(
                    masks,
                    logits,
                    scores,
                    is_single=True,
                )
                if masks.sum() == 0:
                    return {"upscaled_masks": masks}

                has_mask_input = self._has_mask_inputs[1]

            elif i == 2:
                # Cascaded Post-refinement-2
                mask_input, masks, _, _ = _decide_masks(
                    masks,
                    logits,
                    scores,
                )
                if masks.sum() == 0:
                    return {"upscaled_masks": masks}

                has_mask_input = self._has_mask_inputs[1]
                y, x = np.nonzero(masks)
                box_coords = self.decoder.apply_coords(
                    np.array(
                        [[x.min(), y.min()], [x.max(), y.max()]],
                        dtype=np.float32,
                    ),
                    original_size,
                )
                box_coords = np.expand_dims(box_coords, axis=0)
                inputs.update(
                    {
                        "point_coords": np.concatenate(
                            (inputs["point_coords"], box_coords),
                            axis=1,
                        ),
                        "point_labels": np.concatenate(
                            (inputs["point_labels"], self._point_labels_box),
                            axis=1,
                        ),
                    },
                )

            inputs.update({"mask_input": mask_input, "has_mask_input": has_mask_input})
            prediction = self.decoder.infer_sync(inputs)
            upscaled_masks, scores, logits = (
                prediction["upscaled_masks"],
                prediction["iou_predictions"],
                prediction["low_res_masks"],
            )
            masks = upscaled_masks > self.decoder.mask_threshold

        _, masks, _, best_idx = _decide_masks(masks, logits, scores)
        return {"upscaled_masks": masks, "masks": masks, "scores": scores, "logits": logits, "best_idx": best_idx}


class SAMPartAwareLearnableVisualPrompter(SAMLearnableVisualPrompter):
    """
    A wrapper that provides ZSL Visual Prompting workflow with part-awareness.
    This is based on the paper "Part-aware Personalized Segment Anything Model for Patient-Specific Segmentation" by Zhao et al.
    """

    def __init__(
        self,
        encoder_model: SAMImageEncoder,
        decoder_model: SAMDecoder,
        reference_features: VisualPromptingFeatures | None = None,
        threshold: float = 0.65,
        min_number_of_part_level_features: int = 1,
        max_number_of_part_level_features: int = 5,
    ):
        super().__init__(encoder_model, decoder_model, reference_features, threshold)
        self.min_number_of_part_level_features = min_number_of_part_level_features
        self.max_number_of_part_level_features = max_number_of_part_level_features
        self._reference_features: dict[int, np.ndarray] | None = None
        if reference_features is not None:
            self._reference_features = {
                label: feature
                for label, feature in zip(reference_features.used_indices, reference_features.feature_vectors)
            }

    def reset_reference_info(self) -> None:
        """Initialize reference information."""
        self._reference_features = {}
        self._used_indices = np.array([], dtype=np.int64)

    def _expand_reference_info(self, new_largest_label: int) -> None:
        """Expand reference info dimensions if newly given processed prompts have more labels."""
        # no need to expand reference info, as we use a dictionary with numpy array per label
        return

    def get_image_embedding(self, image):
        image_embeddings = self.encoder(image)
        processed_embedding = image_embeddings.squeeze().transpose(1, 2, 0)
        return image_embeddings, processed_embedding

    def learn(
        self,
        image: np.ndarray,
        boxes: list[Prompt] | None = None,
        points: list[Prompt] | None = None,
        polygons: list[Prompt] | None = None,
        reset_features: bool = False,
        perform_averaging: bool = False,
    ) -> tuple[VisualPromptingFeatures, np.ndarray]:
        """Executes `learn` stage of SAM ZSL pipeline.

        Reference features are updated according to newly arrived prompts.
        Features corresponding to the same labels are overridden during
        consequent learn() calls.

        Args:
            image (np.ndarray): HWC-shaped image
            boxes (list[Prompt] | None, optional): Prompts containing bounding boxes (in XYXY torchvision format)
              and their labels (ints, one per box). Defaults to None.
            points (list[Prompt] | None, optional): Prompts containing points (in XY format)
              and their labels (ints, one per point). Defaults to None.
            polygons: (list[Prompt] | None): Prompts containing polygons (a sequence of points in XY format)
              and their labels (ints, one per polygon).
              Polygon prompts are used to mask out the source features without implying decoder usage. Defaults to None.
            reset_features (bool, optional): Forces learning from scratch. Defaults to False.
            perform_averaging (bool, optional): If True, the reference features are averaged, reducing dimensions.

        Returns:
            tuple[VisualPromptingFeatures, np.ndarray]: return values are the updated VPT reference features and
                reference masks.
            The shape of the reference mask is N_labels x H x W, where H and W are the same as in the input image.
        """
        if boxes is None and points is None and polygons is None:
            msg = "boxes, polygons or points prompts are required for learning"
            raise RuntimeError(msg)

        if reset_features or not self.has_reference_features():
            self.reset_reference_info()

        processed_prompts = self.decoder.preprocess(
            {
                "bboxes": [box.data for box in boxes] if boxes else None,
                "points": [point.data for point in points] if points else None,
                "labels": {
                    "bboxes": [box.label for box in boxes] if boxes else None,
                    "points": [point.label for point in points] if points else None,
                },
                "orig_size": image.shape[:2],
            },
        )

        if polygons is not None:
            for poly in polygons:
                processed_prompts.append({"polygon": poly.data, "label": poly.label})

        processed_prompts_w_labels = self._gather_prompts_with_labels(processed_prompts)
        largest_label: int = max([int(p) for p in processed_prompts_w_labels] + [0])

        # self._expand_reference_info(largest_label)

        original_shape = np.array(image.shape[:2])

        # forward image encoder
        image_embeddings = self.encoder(image)
        processed_embedding = image_embeddings.squeeze().transpose(1, 2, 0)

        # get reference masks
        ref_masks: np.ndarray = np.zeros(
            (largest_label + 1, *original_shape),
            dtype=np.uint8,
        )
        for label, input_prompts in processed_prompts_w_labels.items():
            ref_mask: np.ndarray = np.zeros(original_shape, dtype=np.uint8)
            for inputs_decoder in input_prompts:
                inputs_decoder.pop("label")
                if "point_coords" in inputs_decoder:
                    # bboxes and points
                    inputs_decoder["image_embeddings"] = image_embeddings
                    prediction = self._predict_masks(
                        inputs_decoder,
                        original_shape,
                        is_cascade=self._is_cascade,
                    )
                    masks = prediction["upscaled_masks"]
                elif "polygon" in inputs_decoder:
                    masks = _polygon_to_mask(inputs_decoder["polygon"], *original_shape)
                else:
                    msg = "Unsupported type of prompt"
                    raise RuntimeError(msg)
                ref_mask = np.where(masks, 1, ref_mask)

            ref_feat: np.ndarray | None = None
            cur_default_threshold_reference = self._default_threshold_reference

            while ref_feat is None:
                ref_feat = _generate_masked_features(
                    feats=processed_embedding,
                    masks=ref_mask,
                    threshold_mask=cur_default_threshold_reference,
                    image_size=self.encoder.image_size,
                    perform_averaging=perform_averaging,
                )
                cur_default_threshold_reference -= 0.05

            if self._reference_features is not None:
                # TODO reference_features uses predefined size based on one reference feature (1,256) per label
                #   here we use a variable number of part-level features per label
                #       need to handle both reset_reference_info and expand_reference_info
                #      to handle this we could use a dictionary with numpy array per label
                if label in self._reference_features:
                    self._reference_features[label] = np.vstack((self._reference_features[label], ref_feat))
                else:
                    self._reference_features[label] = ref_feat
            self._used_indices = np.concatenate((self._used_indices, [label]))
            ref_masks[label] = ref_mask

        self._used_indices = np.unique(self._used_indices)

        return self.reference_features, ref_masks

    def infer(
        self,
        image: np.ndarray,
        reference_features: VisualPromptingFeatures | None = None,
        apply_masks_refinement: bool = True,
        dev: bool = False,
    ) -> ZSLVisualPromptingResult | tuple[ZSLVisualPromptingResult, dict]:
        """Obtains masks by already prepared reference features.

        Reference features can be obtained with SAMLearnableVisualPrompter.learn() and passed as an argument.
        If the features are not passed, instance internal state will be used as a source of the features.

        Args:
            image (np.ndarray): HWC-shaped image
            reference_features (VisualPromptingFeatures | None, optional): Reference features object obtained during
                previous learn() calls. If not passed, object internal state is used, which reflects the last learn()
                call. Defaults to None.
            apply_masks_refinement (bool, optional): Flag controlling additional refinement stage on inference.
            Once enabled, decoder will be launched 2 extra times to refine the masks obtained with the first decoder
            call. Defaults to True.
            dev (bool, optional): Flag controlling whether to return additional debug information. Defaults to False.

        Returns:
            ZSLVisualPromptingResult: Mapping label -> predicted mask. Each mask object contains a list of binary masks,
                and a list of related prompts. Class mask can be obtained by applying OR operation to all mask
                 corresponding to one label.
        """

        if reference_features is None:
            if self._reference_features is None:
                msg = (
                    "Reference features are not defined. This parameter can be passed via "
                    "SAMLearnableVisualPrompter constructor, or as an argument of infer() method"
                )
                raise RuntimeError(msg)
            original_reference_features = self._reference_features

            if self._used_indices is None:
                msg = (
                    "Used indices are not defined. This parameter can be passed via "
                    "SAMLearnableVisualPrompter constructor, or as an argument of infer() method"
                )
                raise RuntimeError(msg)
            used_idx = self._used_indices
        else:
            original_reference_features, used_idx = reference_features

        target_img_original_shape = np.array(image.shape[:2])
        target_img_features = self.encoder(image)
        target_features = target_img_features.squeeze()
        c_feat, h_feat, w_feat = target_features.shape
        target_features = target_features / np.linalg.norm(target_features, axis=0, keepdims=True)
        target_features = target_features.reshape(c_feat, h_feat * w_feat)

        # TODO instead of recomputing with different number of clusters we could predefine the number of clusters
        #   and compute wasserstein distance for different number of clusters in one go without recomputing.

        best_prediction_per_label = {}
        all_predictions_per_label = {}

        # Check multiple number of clusters to see which set of part-level features matches target distribution best
        for label in used_idx:
            best_distribution_score = 1e6
            for n_clusters in range(self.min_number_of_part_level_features, self.max_number_of_part_level_features + 1):
                start_time = time.time()
                class_reference_feature = original_reference_features[label]
                part_level_features = _create_part_level_features(class_reference_feature, n_clusters)

                # similarity matching of part-level features to target features
                # TODO use vectorDB to store part level features of multiple prior objects and then query for similarity
                sim_map_of_part_level_features = part_level_features @ target_features  # [n_clusters, h, w]
                sim_map_of_part_level_features = np.array(
                    [
                        _resize_to_original_shape(sim_map, self._image_size, target_img_original_shape)
                        for sim_map in sim_map_of_part_level_features
                    ]
                )  # [n_clusters, h, w]

                # Find point prompt candidates for each part level feature
                all_point_prompt_candidates = []
                for part_level_feature_idx in range(n_clusters):
                    # default implementation from PerSAM only selects top point here as it is not multi-object aware
                    # here we allow multiple points to be selected but filter out points in same grid cell
                    part_level_sim_map = sim_map_of_part_level_features[part_level_feature_idx]  # [h, w]
                    point_prompts, _ = _point_selection(
                        mask_sim=part_level_sim_map,
                        original_shape=target_img_original_shape,
                        threshold=self._threshold,
                        num_bg_points=0,
                        image_size=self._image_size,
                        downsizing=self._grid_size,
                    )
                    if point_prompts is not None:  # select top1 point
                        point_prompts = point_prompts[:1]
                        all_point_prompt_candidates.extend(point_prompts)
                print(
                    f"Created {len(all_point_prompt_candidates)} point prompt candidates for label {label} with {n_clusters} part-level features"
                )

                # Background point, is selected based on the average similarity of all part level features
                if n_clusters == 1:
                    average_part_level_sim_mask = sim_map_of_part_level_features.squeeze()
                else:
                    average_part_level_sim_mask = sim_map_of_part_level_features.mean(axis=1).squeeze()  # [h, w]
                background_points = _get_background_points(
                    mask_sim=average_part_level_sim_mask,
                    num_bg_points=self._num_bg_points,
                )

                # Compute Target Guidance for cross-attention layers to bias/focus the decoder towards the target
                # object.
                # TODO pass the target guidance to the decoder (need to change the OpenVino Visual Prompter for this)
                # attention_vector = _compute_target_guidance(average_part_level_sim_mask)

                # First step prediction
                prediction = self.preprocess_and_predict_masks(
                    all_point_prompt_candidates,
                    background_points,
                    target_img_features,
                    target_img_original_shape,
                    apply_masks_refinement,
                )
                # prediction["upscaled_masks"] is passed as bools, should be int
                prediction["upscaled_masks"] = prediction["upscaled_masks"].astype(np.uint8)
                # compute the similarity between the distribution of the part-level features and the target

                masked_target_feature = _generate_masked_features(
                    feats=target_img_features.squeeze().transpose(1, 2, 0),
                    masks=prediction["upscaled_masks"],
                    threshold_mask=self._threshold,
                    image_size=self._image_size,
                )
                # TODO official implementation uses
                #  class reference features in size (1285, 256) and masked target with (52, 256)
                #   here (1, 256) and (64, 64) are used
                #   1, 256 since we normalized the reference feature to a single 256 vector
                #   64 64 is wrong and should be a multiple of 256

                distribution_distance = _compute_wasserstein_distance(class_reference_feature, masked_target_feature)

                # TODO for production remove these dicts to reduce memory usage
                prediction.update({"distribution_distance": distribution_distance})
                prediction.update({"num_clusters": n_clusters})
                prediction.update({"point_prompts": all_point_prompt_candidates})
                # if all predictions per label [label] does not yet have a list for this label, create it
                if label not in all_predictions_per_label:
                    all_predictions_per_label[label] = []
                all_predictions_per_label[label].append(prediction)

                print(f"Distribution distance for label {label} with {n_clusters} clusters: {distribution_distance}")
                print(f"Time taken for {n_clusters} clusters: {time.time() - start_time}")

                if distribution_distance < best_distribution_score:
                    best_distribution_score = distribution_distance
                    best_prediction_per_label[label] = prediction

            print(
                f"Best score for label {label} is {best_distribution_score} with {best_prediction_per_label[label]['num_clusters']} clusters"
            )

        # check overlapping area between different label masks
        # TODO adapt for multi object output (add multiple masks per label)
        predicted_masks = {
            label: [best_prediction_per_label[label]["upscaled_masks"]] for label in best_prediction_per_label
        }
        used_points = {label: best_prediction_per_label[label]["point_prompts"] for label in best_prediction_per_label}
        _inspect_overlapping_areas(predicted_masks, used_points)

        # Format output to match the expected output of the OpenVino Visual Prompter
        best_prediction: dict[int, PredictedMask] = {}
        for k in used_points:
            processed_points = []
            scores = []
            for pt in used_points[k]:
                processed_points.append(pt[:2])
                scores.append(float(pt[2]))
            best_prediction[k] = PredictedMask(predicted_masks[k], processed_points, scores)

        if dev:
            return ZSLVisualPromptingResult(best_prediction), all_predictions_per_label
        return ZSLVisualPromptingResult(best_prediction)

    def preprocess_and_predict_masks(
        self,
        point_prompts: List[np.ndarray],
        background_points: np.ndarray,
        target_img_features: np.ndarray,
        target_img_original_shape: np.ndarray,
        apply_masks_refinement: bool,
    ) -> dict:
        """
        Preprocesses the point prompts and predicts the masks
        :param point_prompts:  List of point prompts for target object List[x, y, score]
        :param background_points: Background points
        :param target_img_features: Features of the target image
        :param target_img_original_shape:  Original shape of the target image (for resizing)
        :param apply_masks_refinement: Whether to apply the 3 step cascaded refinement of the decoder output.
        :return: Prediction of the masks in dictionary format
        """
        # TODO check if decoder supports parralel processing so we can drop the loop of the point prompts in Infer.
        point_coords = np.empty((len(point_prompts) + len(background_points), 2), dtype=np.float32)
        for idx, point in enumerate(point_prompts):
            x, y = point[:2]
            point_coords[idx] = [x, y]
        point_coords[len(point_prompts) :] = background_points
        point_coords = self.decoder.apply_coords(point_coords, target_img_original_shape)

        # 0 is a negative input point, 1 is a positive input point, 2 is a top-left box corner, 3 is a bottom-right box
        # corner, and -1 is a padding point.
        point_labels = np.array([1] * len(point_prompts) + [0] * len(background_points), dtype=np.float32)
        inputs_decoder = {
            "point_coords": point_coords[None],  # vec[None] adds an extra dimension for expected decoder input
            "point_labels": point_labels[None],
            "orig_size": target_img_original_shape[None],
            "image_embeddings": target_img_features,
        }
        prediction: dict[str, np.ndarray] = self._predict_masks(
            inputs_decoder,
            target_img_original_shape,
            apply_masks_refinement,
            # TODO Verify implementation and performance of Cascaded Post Refinement for increased mask quality
        )
        # have to return a score for visualization, lets show highest score here
        prediction.update({"scores": point_prompts[0][2]})
        return prediction


def _compute_target_guidance(average_part_level_sim_mask: np.ndarray) -> np.ndarray:
    """
    Compute target guidance for cross-attention layers to bias/focus the decoder towards the target object.
    :param average_part_level_sim_mask: Average similarity mask of all part level features
    :return: Target Guidance Vector
    """
    normalized_sim = (
        average_part_level_sim_mask - average_part_level_sim_mask.mean()
    ) / average_part_level_sim_mask.std()
    normalized_sim = cv2.resize(normalized_sim, (64, 64), interpolation=cv2.INTER_LINEAR)
    # create attention map using sigmoid by mapping values to [0,1]
    attention_map = 1 / (1 + np.exp(-normalized_sim))  # [64, 64]
    attention_vector = attention_map.reshape(1, 1, -1)  # [1, 1, 4096]
    return attention_vector


def _create_part_level_features(reference_features: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    To create part level features from reference features, we cluster the reference features into n_clusters
    and take the centroid as prototype.

    :param reference_features: Features of the reference object, needs to be multiple features e.g. [X, 256]
    :param n_clusters: Number of clusters (e.g. number of part-level-features) to cluster the reference features into
    :return: Part level features
    """
    if n_clusters == 1:
        part_level_features = reference_features.mean(axis=0)
        part_level_features = part_level_features / np.linalg.norm(part_level_features, axis=-1, keepdims=True)
        return part_level_features[None]  # [1, c]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)  # TODO replace with Fais?
    cluster = kmeans.fit_predict(reference_features)
    part_level_features = []
    for c in range(n_clusters):
        # part level feature is the centroid of the cluster
        part_level_feature = reference_features[cluster == c].mean(axis=0)
        # normalize the part level feature
        part_level_feature = part_level_feature / np.linalg.norm(part_level_feature, axis=-1, keepdims=True)
        part_level_features.append(part_level_feature)
    part_level_features = np.stack(part_level_features, axis=0)  # [n_clusters, c]

    return part_level_features


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


def _polygon_to_mask(
    polygon: np.ndarray | list[np.ndarray],
    height: int,
    width: int,
) -> np.ndarray:
    """Converts a polygon represented as an array of 2D points into a mask"""
    if isinstance(polygon, np.ndarray) and np.issubdtype(polygon.dtype, np.integer):
        contour = polygon.reshape(-1, 2)
    else:
        contour = [[int(point[0]), int(point[1])] for point in polygon]
    gt_mask = np.zeros((height, width), dtype=np.uint8)
    return cv2.drawContours(gt_mask, np.asarray([contour]), 0, 1, cv2.FILLED)


def _generate_masked_features(
    feats: np.ndarray,
    masks: np.ndarray,
    threshold_mask: float,
    image_size: int = 1024,
    perform_averaging: bool = True,
) -> np.ndarray | None:
    """Generate masked features.

    Args:
        feats (np.ndarray): Raw reference features. It will be filtered with masks.
        masks (np.ndarray): Reference masks used to filter features.
        threshold_mask (float): Threshold to control masked region.
        image_size (int): Input image size.

    Returns:
        (np.ndarray): Masked features.
    """
    target_shape = image_size / max(masks.shape) * np.array(masks.shape)
    target_shape = target_shape[::-1].astype(np.int32)

    # Post-process masks
    masks = cv2.resize(masks, target_shape, interpolation=cv2.INTER_LINEAR)
    masks = _pad_to_square(masks, image_size)
    masks = cv2.resize(masks, feats.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)

    # Target feature extraction
    if (masks > threshold_mask).sum() == 0:
        # (for stability) there is no area to be extracted
        return None

    # TODO we compute the mean of the masked features, we need all features for wasserstein distance
    masked_feat = feats[masks > threshold_mask]
    if perform_averaging:
        masked_feat = masked_feat.mean(0)[None]
        return masked_feat / np.linalg.norm(masked_feat, axis=-1, keepdims=True)
    return masked_feat


def _pad_to_square(x: np.ndarray, image_size: int = 1024) -> np.ndarray:
    """Pad to a square input.

    Args:
        x (np.ndarray): Mask to be padded.

    Returns:
        (np.ndarray): Padded mask.
    """
    h, w = x.shape[-2:]
    padh = image_size - h
    padw = image_size - w
    return np.pad(x, ((0, padh), (0, padw)), constant_values=0.0)


def _decide_masks(
    masks: np.ndarray,
    logits: np.ndarray,
    scores: np.ndarray,
    is_single: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, int] | tuple[None, np.ndarray, float, int]:
    """Post-process logits for resized masks according to best index based on scores."""
    if is_single:
        best_idx = 0
    else:
        # skip the first index components
        scores, masks, logits = (x[:, 1:] for x in (scores, masks, logits))

        # filter zero masks
        while len(scores[0]) > 0 and masks[0, (best_idx := np.argmax(scores[0]))].sum() == 0:
            scores, masks, logits = (
                np.concatenate((x[:, :best_idx], x[:, best_idx + 1 :]), axis=1) for x in (scores, masks, logits)
            )

        if len(scores[0]) == 0:
            # all predicted masks were zero masks, ignore them.
            return (
                None,
                np.zeros(masks.shape[-2:]),
                0.0,
            )

        best_idx = np.argmax(scores[0])
    return (logits[:, [best_idx]], masks[0, best_idx], float(np.clip(scores[0][best_idx], 0, 1)), best_idx)


def _get_prompt_candidates(
    image_embeddings: np.ndarray,
    reference_feats: np.ndarray,
    used_indices: np.ndarray,
    original_shape: np.ndarray,
    threshold: float = 0.0,
    num_bg_points: int = 1,
    default_threshold_target: float = 0.65,
    image_size: int = 1024,
    downsizing: int = 64,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Get prompt candidates."""
    target_feat = image_embeddings.squeeze()
    c_feat, h_feat, w_feat = target_feat.shape
    target_feat = target_feat / np.linalg.norm(target_feat, axis=0, keepdims=True)
    target_feat = target_feat.reshape(c_feat, h_feat * w_feat)

    total_points_scores: dict[int, np.ndarray] = {}
    total_bg_coords: dict[int, np.ndarray] = {}
    for label in used_indices:
        sim = reference_feats[label] @ target_feat
        sim = sim.reshape(h_feat, w_feat)
        sim = _resize_to_original_shape(sim, image_size, original_shape)

        threshold = (threshold == 0) * default_threshold_target + threshold
        points_scores, bg_coords = _point_selection(
            mask_sim=sim,
            original_shape=original_shape,
            threshold=threshold,
            num_bg_points=num_bg_points,
            image_size=image_size,
            downsizing=downsizing,
        )

        if points_scores is not None:
            total_points_scores[label] = points_scores
            total_bg_coords[label] = bg_coords
    return total_points_scores, total_bg_coords


def _point_selection(
    mask_sim: np.ndarray,
    original_shape: np.ndarray,
    threshold: float = 0.0,
    num_bg_points: int = 1,
    image_size: int = 1024,
    downsizing: int = 64,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None] | tuple[np.ndarray, None]:
    """Select point used as point prompts.

    :param mask_sim: Similarity mask.
    :param original_shape: Original shape of the input image.
    :param threshold: Threshold to filter out points.
    :param num_bg_points: Number of background points.
    :param image_size: Image size.
    :param downsizing: Downsizing factor.
    """
    _, w_sim = mask_sim.shape

    # Find point candidates by comparing the similarity mask to the threshhold
    point_coords = np.where(mask_sim > threshold)
    fg_coords_scores = np.stack(
        point_coords[::-1] + (mask_sim[point_coords],),
        axis=0,
    ).T   # (107313, 3)

    # skip if there are no prompt candidates
    if len(fg_coords_scores) == 0:
        return None, None

    # Create a grid of the original image size. This is used to filter out points that are in the same grid cell.
    ratio = image_size / original_shape.max()   # ratio = 0.8
    width = (original_shape[1] * ratio).astype(np.int64)  # width= 1024
    number_of_grid_cells = width // downsizing

    # get grid numbers
    idx_grid = (
        fg_coords_scores[:, 1] * ratio // downsizing * number_of_grid_cells
        + fg_coords_scores[:, 0] * ratio // downsizing
    )
    idx_grid_unique = np.unique(idx_grid.astype(np.int64))

    # get matched indices
    matched_matrix = np.expand_dims(idx_grid, axis=-1) == idx_grid_unique  # (totalN, uniqueN)

    # sample fg_coords_scores matched by matched_matrix
    matched_grid = np.expand_dims(fg_coords_scores, axis=1) * np.expand_dims(
        matched_matrix,
        axis=-1,
    )

    # sample the highest score one of the samples that are in the same grid
    matched_indices = _topk_numpy(matched_grid[..., -1], k=1, axis=0, largest=True)[1][0].astype(np.int64)
    points_scores = matched_grid[matched_indices].diagonal().T

    # sort by the highest score
    sorted_points_scores_indices = np.flip(
        np.argsort(points_scores[:, -1]),
        axis=-1,
    ).astype(np.int64)
    points_scores = points_scores[sorted_points_scores_indices]

    # Background point selection
    background_points = _get_background_points(mask_sim, num_bg_points)

    return points_scores, background_points


def _get_background_points(mask_sim, num_bg_points=1) -> np.ndarray | None:
    """
    Select background point based on the similarity mask.
    :param mask_sim: Similarity mask.
    :param num_bg_points: Number of background points.
    :return:
    """
    if num_bg_points == 0:
        return None
    w_sim = mask_sim.shape[1]
    bg_indices = _topk_numpy(mask_sim.flatten(), num_bg_points, largest=False)[1]
    bg_x = np.expand_dims(bg_indices // w_sim, axis=0)
    bg_y = bg_indices - bg_x * w_sim
    bg_coords = np.concatenate((bg_y, bg_x), axis=0).transpose(1, 0)
    bg_coords = bg_coords.astype(np.float32)
    return bg_coords


def _resize_to_original_shape(
    masks: np.ndarray,
    image_size: int,
    original_shape: np.ndarray,
) -> np.ndarray:
    """Resize feature size to original shape of input image by interpolation"""
    # resize feature size to input size
    masks = cv2.resize(masks, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    # remove padding
    prepadded_size = _get_prepadded_size(original_shape, image_size)
    masks = masks[..., : prepadded_size[0], : prepadded_size[1]]

    # resize unpadded one to original shape
    original_shape = original_shape.astype(np.int64)
    h, w = original_shape[0], original_shape[1]
    return cv2.resize(masks, (w, h), interpolation=cv2.INTER_LINEAR)


def _get_prepadded_size(original_shape: np.ndarray, image_size: int) -> np.ndarray:
    """Get pre-padded size."""
    scale = image_size / np.max(original_shape)
    transformed_size = scale * original_shape
    return np.floor(transformed_size + 0.5).astype(np.int64)


def _topk_numpy(
    x: np.ndarray,
    k: int,
    axis: int = -1,
    largest: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k function for numpy same with torch.topk."""
    if largest:
        k = -k
        indices = range(k, 0)
    else:
        indices = range(k)
    partitioned_ind = np.argpartition(x, k, axis=axis).take(indices=indices, axis=axis)
    partitioned_scores = np.take_along_axis(x, partitioned_ind, axis=axis)
    sorted_trunc_ind = np.argsort(partitioned_scores, axis=axis)
    if largest:
        sorted_trunc_ind = np.flip(sorted_trunc_ind, axis=axis)
    ind = np.take_along_axis(partitioned_ind, sorted_trunc_ind, axis=axis)
    scores = np.take_along_axis(partitioned_scores, sorted_trunc_ind, axis=axis)
    return scores, ind


def _inspect_overlapping_areas(
    predicted_masks: dict[int, list[np.ndarray]],
    used_points: dict[int, list[np.ndarray]],
    threshold_iou: float = 0.8,
) -> None:
    """
    Inspect overlapping areas between different label masks.
    :param predicted_masks:    Predicted masks per label
    :param used_points:        Used points per label
    :param threshold_iou:    Threshold for IOU
    :return:
    """

    def _calculate_mask_iou(
        mask1: np.ndarray,
        mask2: np.ndarray,
    ) -> tuple[float, np.ndarray | None]:
        assert mask1.ndim == 2
        assert mask2.ndim == 2
        # Avoid division by zero
        if (union := np.logical_or(mask1, mask2).sum().item()) == 0:
            return 0.0, None
        intersection = np.logical_and(mask1, mask2)
        return intersection.sum().item() / union, intersection

    for (label, masks), (other_label, other_masks) in product(
        predicted_masks.items(),
        predicted_masks.items(),
    ):
        if other_label <= label:
            continue

        overlapped_label = []
        overlapped_other_label = []
        for (im, mask), (jm, other_mask) in product(
            enumerate(masks),
            enumerate(other_masks),
        ):
            _mask_iou, _intersection = _calculate_mask_iou(mask, other_mask)
            if _mask_iou > threshold_iou:
                if used_points[label][im][2] > used_points[other_label][jm][2]:
                    overlapped_other_label.append(jm)
                else:
                    overlapped_label.append(im)
            elif _mask_iou > 0:
                # refine the slightly overlapping region
                overlapped_coords = np.where(_intersection)
                if used_points[label][im][2] > used_points[other_label][jm][2]:
                    other_mask[overlapped_coords] = 0.0
                else:
                    mask[overlapped_coords] = 0.0

        for im in sorted(set(overlapped_label), reverse=True):
            masks.pop(im)
            used_points[label].pop(im)

        for jm in sorted(set(overlapped_other_label), reverse=True):
            other_masks.pop(jm)
            used_points[other_label].pop(jm)
