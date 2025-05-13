# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import requests
import torch
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from torch.nn import functional as F
from torchvision import transforms

import visionprompt.third_party.dinov2.utils.utils as dinov2_utils
from visionprompt.context_learner.processes.encoders.encoder_base import Encoder
from visionprompt.context_learner.types import Features, Image, Masks, Priors, State
from visionprompt.third_party.dinov2.data.transforms import MaybeToTensor, make_normalize_transform
from visionprompt.third_party.dinov2.models import vision_transformer
from visionprompt.third_party.dinov2.models.vision_transformer import DinoVisionTransformer
from visionprompt.utils.constants import DINO_WEIGHTS


class DinoEncoder(Encoder):
    """This encoder uses the DINOv2 model to encode the images."""

    def __init__(self, state: State) -> None:
        super().__init__(state)
        self.input_image_size = 518
        self.patch_size = 14
        self.feature_size = self.input_image_size // self.patch_size

        # Store encoder configuration in the state
        self._state.encoder_input_size = self.input_image_size
        self._state.encoder_patch_size = self.patch_size
        self._state.encoder_feature_size = self.feature_size

        self.model: DinoVisionTransformer = self._setup_model()
        self.encoder_transform = transforms.Compose([
            MaybeToTensor(),
            transforms.Resize((self.input_image_size, self.input_image_size)),
            make_normalize_transform(),
        ])
        self.encoder_mask_transform = transforms.Compose([
            MaybeToTensor(),
            transforms.Lambda(lambda x: x.unsqueeze(0) if x.ndim == 2 else x),
            transforms.Lambda(lambda x: x.float()),
            transforms.Resize([self.input_image_size, self.input_image_size]),
            # MinPool to make sure we do not use background features
            transforms.Lambda(lambda x: (x * -1) + 1),
            torch.nn.MaxPool2d(
                kernel_size=(self.patch_size, self.patch_size),
            ),
            transforms.Lambda(lambda x: (x * -1) + 1),
        ])

    def __call__(
        self,
        images: list[Image],
        priors_per_image: list[Priors] | None = None,
    ) -> tuple[list[Features], list[Masks]]:
        """This method creates an embedding from the images for locations inside the mask.

        Args:
            images: A list of images.
            priors_per_image: A list of priors per image.

        Returns:
            A list of extracted features.
        """
        resized_masks_per_image: list[Masks] = []
        image_features: list[Features] = self._extract_global_features_batch(images)

        if priors_per_image is not None:
            for features, priors in zip(image_features, priors_per_image, strict=False):
                _, resized_masks = self._extract_local_features(features=features, masks_per_class=priors.masks)
                resized_masks_per_image.append(resized_masks)
        else:
            resized_masks_per_image = [Masks() for _ in image_features]

        return image_features, resized_masks_per_image

    def _extract_local_features(self, features: Features, masks_per_class: Masks) -> tuple[Features, Masks]:
        """This method extracts the local features from the image by only keeping the features that are inside the masks.

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
                    e = f"The reference mask is too small to detect any features for class {class_id}"
                    raise ValueError(e)
                features.add_local_features(local_features=local_features, class_id=class_id)
        return features, resized_masks

    @torch.no_grad()
    def _extract_global_features_batch(self, images: list[Image]) -> list[Features]:
        """Extract all global features from the images."""
        image_batch = torch.stack([self.encoder_transform(image.data) for image in images]).cuda()
        features = self.model.forward_features(image_batch)["x_prenorm"][:, 1:]
        features = features.reshape(len(images), -1, self.model.embed_dim)
        features = F.normalize(features, p=2, dim=-1)
        image_features: list[Features] = []
        for idx, _image in enumerate(images):
            image_features.append(Features(global_features=features[idx]))
        return image_features

    def _setup_model(self) -> DinoVisionTransformer:
        """This method initializes the DINO model."""
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
        path = Path("~/data").expanduser() / DINO_WEIGHTS["local_filename"]
        url = DINO_WEIGHTS["download_url"]
        _download_weights(path, url)
        dinov2_utils.load_pretrained_weights(dinov2, str(path), "teacher")
        dinov2.eval()
        dinov2.to(device="cuda")
        return dinov2


def _download_weights(path: Path, url: str) -> None:
    """This method downloads the DINO weights."""
    if not path.exists():  # noqa: PLR1702
        progress = Progress(
            TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            " • ",
            DownloadColumn(),
            " • ",
            TransferSpeedColumn(),
            " • ",
            TimeRemainingColumn(),
            transient=True,
            disable=not sys.stderr.isatty(),
        )
        disable_progress = not sys.stderr.isatty()
        try:
            with requests.get(url, stream=True, timeout=10) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                print(f"Downloading {path.name} ({total_size / (1024 * 1024):.2f} MB) from {url}...")
                with progress:
                    task_id = progress.add_task("download", total=total_size, filename=path.name)
                    with path.open("wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                progress.update(task_id, advance=len(chunk))

                if not disable_progress and total_size > 0:
                    progress.update(task_id, completed=total_size)
        except Exception as e:
            print(f"\nAn unexpected error occurred during download: {e}")
            if path.exists():
                try:
                    path.unlink()
                except OSError as unlink_e:
                    print(f"Error removing file {path} after error: {unlink_e}")
            raise
