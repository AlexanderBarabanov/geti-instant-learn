# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import torch
from torch.nn import functional as F

from visionprompt.context_learner.processes.encoders.encoder_base import Encoder
from visionprompt.context_learner.types import Features, Image, Masks, Priors
from visionprompt.third_party.Matcher.segment_anything.predictor import SamPredictor


class SamEncoder(Encoder):
    """This encoder extracts features from images using a SAM model.

    It can be used to extract reference/local features.
    """

    def __init__(self, sam_predictor: SamPredictor) -> None:
        super().__init__()
        self.predictor: SamPredictor = sam_predictor
        if hasattr(self.predictor.model.image_encoder, "img_size"):
            self.encoder_input_size = self.predictor.model.image_encoder.img_size
        else:
            self.encoder_input_size = self.predictor.model.image_size

    def __call__(
        self,
        images: list[Image] | None = None,
        priors_per_image: list[Priors] | None = None,
    ) -> tuple[list[Features], list[Masks]]:
        """This method creates an embedding from the images.

        If masks are provided, it extracts local features from masked regions.
        If no masks are provided, it extracts global features.

        Args:
            images: A list of images, expected to be in HWC uint8 format, with pixel values in [0, 255].
            priors_per_image: Optional list of priors per image. If None, returns global features.

        Returns:
            A list of extracted features per image (local reference if masks provided, global if not).
            A list of resized masks per image.
        """
        features: list[Features] = []
        resized_masks_per_image: list[Masks] = []
        for idx, image in enumerate(images):
            global_features = self._extract_global_features(image)
            image_features = Features(global_features)

            if priors_per_image is not None:
                priors = priors_per_image[idx]
                image_features, resized_masks = self._extract_local_features(
                    image_features,
                    priors.masks,
                )
            else:
                resized_masks = Masks()

            resized_masks_per_image.append(resized_masks)
            features.append(image_features)

        return features, resized_masks_per_image

    def _extract_global_features(self, image: Image) -> torch.Tensor:
        """Extract image embedding from the image.

        Args:
            image: The image to extract the embedding from.

        Returns:
            The image embedding.
        """
        self.predictor.set_image(image.data)
        # save the size after preprocessing for later use
        image.sam_preprocessed_size = self.predictor.input_size
        embedding = self.predictor.features.squeeze().permute(1, 2, 0)
        return F.normalize(embedding, p=2, dim=-1)

    def _extract_local_features(
        self,
        features: Features,
        masks_per_class: Masks,
    ) -> tuple[Features, Masks]:
        """This method extracts the local features from the image.

        This only keeping the features that are inside the masks.

        Args:
            features: The features to extract the local features from.
            masks_per_class: The masks to extract the features from.

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
                    mask.numpy().astype(np.uint8) * 255,
                )
                input_mask_torch = torch.as_tensor(
                    input_mask,
                    device=self.predictor.device,
                )
                input_mask_torch = input_mask_torch.unsqueeze(0).unsqueeze(
                    0,
                )  # add color and batch dimension
                input_mask = self.predictor.model.preprocess(
                    input_mask_torch,
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
                    e = f"The reference mask is too small to detect any features for class {class_id}"
                    raise ValueError(e)
                features.add_local_features(
                    local_features=local_features,
                    class_id=class_id,
                )
                resized_masks.add(mask=input_mask, class_id=class_id)

        return features, resized_masks
