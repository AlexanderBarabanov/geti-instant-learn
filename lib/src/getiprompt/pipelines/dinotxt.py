# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 zero-shot classification pipeline."""

import torch
from getiprompt.pipelines.pipeline_base import Pipeline
from getiprompt.types import Image, Priors, Results, Masks
from getiprompt.models.dinotxt import IMAGENET_TEMPLATES, DinoTextEncoder

class DinoTxtZeroShotClassification(Pipeline):
    def __init__(
        self,
        pretrained: bool = True,
        prompt_templates: list[str] = IMAGENET_TEMPLATES,
        precision: str = torch.bfloat16,
        device: str = "cuda",
        image_size: int | tuple[int, int] | None = 512,
    ) -> None:
        """DinoTxt pipeline.


        Args:
            pretrained: Whether to use pretrained weights.
            prompt_templates: The prompt templates to use for the model.
            precision: The precision to use for the model.
            device: The device to use for the model.
            image_size: The size of the image to use.

        Examples:
            >>> from getiprompt.pipelines import DINOTxt
            >>> from getiprompt.types import Image, Priors
            >>> from pathlib import Path
            >>>
            >>> dinotxt = DINOTxt(
            >>>     pretrained=True,
            >>>     prompt_templates=["a photo of a {}."],  # default is IMAGENET_TEMPLATES
            >>>     precision=torch.bfloat16,
            >>>     device="cuda",
            >>>     image_size=(512, 512),
            >>> )
            >>> ref_priors = Priors(text={0: "cat", 1: "dog"})
        """
        super().__init__(image_size=image_size)
        self.dino_encoder = DinoTextEncoder(
            repo_id="facebookresearch/dinov3",
            model_id="dinov3_vitl16_dinotxt_tet1280d20h24l",
            pretrained=pretrained,
            device=device,
            image_size=image_size,
            precision=precision,
        )
        self.prompt_templates = prompt_templates
        self.precision = precision

    def learn(
        self, 
        reference_images: list[Image], 
        reference_priors: list[Priors],
    ) -> None:
        """Perform learning step on the priors.

        DINOTxt does not need reference images, but we keep it for consistency.

        Args:
            reference_images: A list of reference images.
            reference_priors: A list of reference priors.

        Returns:
            None

        Examples:
            >>> import torch
            >>> import numpy as np
            >>> from getiprompt.pipelines import DINOTxt
            >>> from getiprompt.types import Image, Priors
            >>> dinotxt = DINOTxt()
            >>> ref_priors = Priors(text={0: "cat", 1: "dog"})
            >>> dinotxt.learn(reference_images=[], reference_priors=[ref_priors])
            >>> dinotxt.infer(target_images=[Image()])
        """
        
        if not reference_priors:
            msg = "reference_priors must be provided"
            raise ValueError(msg)

        reference_prior = reference_priors[0]
        self.class_maps = reference_prior.text.items()
        # reference features is zero shot weights from DinoTxtEncoder
        self.reference_features = self.dino_encoder.encode_text(
            reference_prior,
            self.prompt_templates
        )

    @torch.no_grad()
    def infer(self, target_images: list[Image]):
        target_features = self.dino_encoder.encode_image(target_images)
        target_features /= target_features.norm(dim=-1, keepdim=True)
        logits = 100. * target_features @ self.reference_features
        scores = logits.softmax(dim=1)
        max_scores, max_class_ids = scores.max(dim=1)

        masks = []
        for target_image, max_score, max_class_id in zip(target_images, max_scores, max_class_ids):
            m = torch.zeros(target_image.shape)
            # NOTE: Due to the current type contract, for zero-shot classification, 
            # we need to create a mask for each target image
            # This part should be refactored when we have a Label type class
            mask_type = Masks()
            mask_type.add(mask=m, class_id=max_class_id)
            masks.append(mask_type)
        result = Results()
        result.masks = masks
        return result