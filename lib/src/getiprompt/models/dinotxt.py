# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOTxt model."""

import torch
from torch import nn

from getiprompt.types import Priors, Features
import torchvision

from torchvision.transforms.v2.functional import to_dtype, to_image


# ImageNet templates for zero shot classification
IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
    "a photo of {}.",
    "a satellite photo of {}.",
]


class DinoTextEncoder(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        image_size: int = 512,
        repo_id = "facebookresearch/dinov3",
        model_id = "dinov3_vitl16_dinotxt_tet1280d20h24l",
        precision = torch.bfloat16,
        device: str = "cuda",
        mean: list[float] = [123.675, 116.28, 103.53],
        std: list[float] = [58.395, 57.12, 57.375],
    ) -> None:
        super().__init__()
        model, tokenizer = torch.hub.load(
            repo_id, 
            model_id, 
            pretrained=pretrained,
        )
        self.tokenizer = tokenizer
        
        self.device = device
        self.precision = precision
        model = model.to(dtype=self.precision)
        self.model = model.cuda()
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.v2.Resize(image_size),
            torchvision.transforms.v2.Normalize(mean=mean, std=std),
            torchvision.transforms.v2.ToDtype(dtype=self.precision),
        ])

    @torch.no_grad()
    def encode_text(
        self, 
        reference_prior: Priors,
        prompt_template: list[str] = IMAGENET_TEMPLATES,
    ) -> Features:
        """Encode the class text prompt to text embedding.
        
        Args:
            reference_prior: The prior to encode. 
            prompt_template: The prompt template to use for the model.

        Returns:
            The text embedding.

        Examples:
            >>> from getiprompt.models.dinotxt import DinoTextEncoder
            >>> from getiprompt.types import Priors
            >>> encoder = DinoTextEncoder()
            >>> prior = Priors(text={0: "cat", 1: "dog"})
            >>> encoder.encode_text(prior)
            Features(shape=(2, 1, 4), embedding_dim=4)
        """
        zero_shot_weights = Features()
        for class_id, label_name in reference_prior.text.items():
            texts = [template.format(label_name) for template in prompt_template] 
            texts = self.tokenizer.tokenize(texts).cuda()
            class_embeddings = self.model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zero_shot_weights.add_local_features(class_embedding, class_id=class_id)
        return zero_shot_weights
    
    @torch.no_grad()
    def encode_image(
        self,
        target_images: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the reference images to image embedding."""

        images = [self.transforms(to_dtype(to_image(image), dtype=self.precision)) for image in target_images]
        images = torch.stack(images, dim=0)
        with torch.autocast(device_type=self.device, dtype=self.precision):
            image_features = self.model.encode_image(images.cuda())
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.to(self.precision)