import os
from typing import List, Optional

import torch
from context_learner.types.masks import Masks
from context_learner.types.priors import Priors
from dinov2.data.transforms import MaybeToTensor, make_normalize_transform
from dinov2.models import vision_transformer
from dinov2.models.vision_transformer import DinoVisionTransformer
import dinov2.utils.utils as dinov2_utils

from context_learner.processes.encoders.encoder_base import Encoder
from context_learner.types.features import Features
from context_learner.types.image import Image
from context_learner.types.state import State
from torchvision import transforms
from torch.nn import functional as F


class DinoEncoder(Encoder):
    """
    This encoder uses the DINOv2 model to encode the images.
    """

    def __init__(self, state: State):
        super().__init__(state)
        self.input_image_size = 518
        self.patch_size = 14
        self.feature_size = self.input_image_size // self.patch_size

        # Store encoder configuration in the state
        self._state.encoder_input_size = self.input_image_size
        self._state.encoder_patch_size = self.patch_size
        self._state.encoder_feature_size = self.feature_size

        self.model: DinoVisionTransformer = self._setup_model()
        self.encoder_transform = transforms.Compose(
            [
                MaybeToTensor(),
                transforms.Resize((self.input_image_size, self.input_image_size)),
                make_normalize_transform(),
            ]
        )
        self.encoder_mask_transform = transforms.Compose(
            [
                MaybeToTensor(),
                transforms.Lambda(lambda x: x.unsqueeze(0) if x.ndim == 2 else x),
                transforms.Lambda(lambda x: x.float()),
                transforms.Resize([self.input_image_size, self.input_image_size]),
                # torch min pool instead of avg pool:
                transforms.Lambda(lambda x: (x * -1) + 1),
                torch.nn.AvgPool2d(
                    kernel_size=(self.patch_size, self.patch_size),
                ),
                transforms.Lambda(lambda x: (x * -1) + 1),
            ]
        )

    def __call__(
        self, images: List[Image], priors_per_image: Optional[List[Priors]] = None
    ) -> tuple[List[Features], List[Masks]]:
        """
        This method creates an embedding from the images for locations inside the mask.

        Args:
            images: A list of images.
            priors_per_image: A list of priors per image.

        Returns:
            A list of extracted features.
        """
        resized_masks_per_image: List[Masks] = []
        image_features: List[Features] = self._extract_global_features_batch(images)

        if priors_per_image is not None:
            for features, priors in zip(image_features, priors_per_image):
                features, resized_masks = self._extract_local_features(
                    features=features, masks_per_class=priors.masks
                )
                resized_masks_per_image.append(resized_masks)
        else:
            resized_masks_per_image = [Masks() for _ in image_features]

        return image_features, resized_masks_per_image

    def _extract_local_features(
        self, features: Features, masks_per_class: Masks
    ) -> tuple[Features, Masks]:
        """
        This method extracts the local features from the image by only keeping the features
        that are inside the masks.

        Args:
            features: The features to extract the local features from.
            masks_per_class: The masks to extract the local features from.

        Returns:
            The features with the local features extracted.
        """
        resized_masks = Masks()
        for class_id, masks in masks_per_class.data.items():
            for mask in masks:
                # preprocess mask, add batch dim, convert to float and resize
                pooled_mask = self.encoder_mask_transform(mask.data).cuda()
                resized_masks.add(mask=pooled_mask, class_id=class_id)
                # extract local features
                indices = pooled_mask.flatten().bool()
                local_features = features.global_features[indices]
                if local_features.shape[0] == 0:
                    print("The reference mask is too small to detect any features")
                else:
                    features.add_local_features(
                        local_features=local_features, class_id=class_id
                    )
        return features, resized_masks

    @torch.no_grad()
    def _extract_global_features_batch(self, images: List[Image]) -> List[Features]:
        """
        Extract all global features from the images.
        """
        image_batch = torch.stack(
            [self.encoder_transform(image.data) for image in images]
        ).cuda()
        features = self.model.forward_features(image_batch)["x_prenorm"][:, 1:]
        features = features.reshape(len(images), -1, self.model.embed_dim)
        features = F.normalize(features, p=2, dim=-1)
        image_features: List[Features] = []
        for idx, image in enumerate(images):
            image_features.append(Features(global_features=features[idx]))
        return image_features

    @torch.no_grad()
    def _resize_masks(self, priors_per_image: List[Priors]) -> List[Masks]:
        """
        Resize the masks to the input image size.
        """
        resized_masks = []
        for priors in priors_per_image:
            for mask in priors.masks:
                mask = self.encoder_mask_transform(priors.masks.data).cuda()
                pooled_mask = F.avg_pool2d(
                    mask, kernel_size=(self.model.patch_size, self.model.patch_size)
                )
                resized_masks.append(Masks(pooled_mask))

        return resized_masks

    def _setup_model(self):
        """
        This method initializes the DINO model.
        """
        dinov2: DinoVisionTransformer = vision_transformer.__dict__["vit_large"](
            img_size=self.input_image_size,
            patch_size=self.patch_size,
            init_values=1e-5,
            ffn_layer="mlp",
            block_chunks=0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
        )
        path = os.path.expanduser("~/data/dinov2_vitl14_pretrain.pth")
        dinov2_utils.load_pretrained_weights(dinov2, path, "teacher")
        dinov2.eval()
        dinov2.to(device="cuda")
        return dinov2
