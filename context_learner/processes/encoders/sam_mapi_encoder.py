# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from context_learner.processes.encoders.encoder_base import Encoder
from context_learner.types.features import Features
from context_learner.types.image import Image
from context_learner.types.masks import Masks
from context_learner.types.priors import Priors
from context_learner.types.state import State
from model_api.models.visual_prompting import SAMLearnableVisualPrompter
from model_api.models import Prompt
import cv2


class SamMAPIEncoder(Encoder):
    """
    This is a wrapper around the ModelAPI SAM encoder.
    This encoder extracts features from images using a SAM model. It can be used to extract reference/local features.
    """

    def __init__(self, state: State, model: SAMLearnableVisualPrompter):
        super().__init__(state)
        self._model = model

    @staticmethod
    def _mask_to_polygons(mask: np.ndarray) -> List[np.ndarray]:
        """Converts a binary mask to a list of polygons in format XY"""

        # Find contours
        contours, hierarchy = cv2.findContours(
            mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Loop through contours
        polygons = []
        for cnt in contours:
            # Approximate contour to polygon (optional, for simplification)
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            polygon = approx.reshape(-1, 2)
            polygons.append(polygon)
        return polygons

    def __call__(
        self, images: List[Image], priors_per_image: Optional[List[Priors]] = None
    ) -> tuple[List[Features], List[Masks]]:
        """
        This method creates an embedding from the images. If masks are provided,
        extracts local features from masked regions. If none are provided,  it extracts global features.

        Args:
            images: A list of images, expected to be in HWC uint8 format, with pixel values in [0, 255].
            priors_per_image: Optional list of priors per image. If None, returns global features.

        Returns:
            A list of extracted features per image (local reference if masks provided, global if not).
            A list of resized masks per image.
        """

        # Convert input to MAPI format
        if len(images) != len(priors_per_image):
            raise ValueError("Both images and priors need to be specified")

        features: List[Features] = []
        masks: List[Masks] = []

        for image, prior in zip(images, priors_per_image):
            image_np = image.to_numpy()
            mask_np = np.moveaxis(prior.masks.to_numpy(), 0, 2) * 255
            polygons = self._mask_to_polygons(mask_np)
            prompt = [Prompt(data=polygon, label=0) for polygon in polygons]
            # Learn features
            sam_features, sam_masks = self._model.learn(image=image_np, polygons=prompt)
            features.append(
                Features(global_features=torch.from_numpy(sam_features.feature_vectors))
            )
            m = Masks()
            m.add(np.moveaxis(sam_masks, 0, 2))
            masks.append(m)

        return features, masks

    def _extract_global_features(self, image: Image) -> torch.Tensor:
        """
        Extract image embedding from the image.

        Args:
            image: The image to extract the embedding from.

        Returns:
            The image embedding.
        """
        self.predictor.set_image(image.data)
        # save the size after preprocessing for later use
        image.transformed_size = self.predictor.input_size
        embedding = self.predictor.get_image_embedding().squeeze().permute(1, 2, 0)
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding

    def _extract_local_features(
        self, features: Features, masks_per_class: Masks
    ) -> tuple[Features, Masks]:
        """
        This method extracts the local features from the image by only keeping the features
        that are inside the masks.

        Args:
            image: The image to extract the embedding from.
            masks: The masks to extract the features from.

        Returns:
            Features object containing the local features per class and mask.
            The processed masks. These are resized to match the encoder embedding shape.
        """

        resized_masks = Masks()
        for class_id, masks in masks_per_class.data.items():
            # perform per mask as the current predictor does not support batches
            masks: torch.Tensor  # 3D tensor with n_masks x H x W
            for mask in masks:
                input_mask = self.predictor.transform.apply_image(
                    mask.numpy().astype(np.uint8) * 255
                )
                input_mask_torch = torch.as_tensor(
                    input_mask, device=self.predictor.device
                )
                input_mask_torch = input_mask_torch.unsqueeze(0).unsqueeze(
                    0
                )  # add color and batch dimension
                input_mask = self.predictor.model.preprocess(
                    input_mask_torch
                )  # (normalize) and pad
                # transform mask to embedding shape
                input_mask = F.interpolate(
                    input_mask,
                    size=features.global_features_shape[:2],
                    mode="bilinear",
                )
                input_mask = input_mask.squeeze(0)[0]  # (emb_shape, emb_shape)
                local_features = features.global_features[input_mask > 0]
                if local_features.shape[0] == 0:
                    print("The reference mask is too small to detect any features")
                else:
                    features.add_local_features(
                        local_features=local_features, class_id=class_id
                    )
                resized_masks.add(mask=input_mask, class_id=class_id)

        return features, resized_masks
