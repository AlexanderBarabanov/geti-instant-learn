import os
from typing import List, Optional
from dinov2.models import vision_transformer
from dinov2.models.vision_transformer import DinoVisionTransformer
import dinov2.utils.utils as dinov2_utils

from context_learner.processes.encoders.encoder_base import Encoder
from context_learner.types.features import Features
from context_learner.types.image import Image
from context_learner.types.annotations import Annotations
from context_learner.types.state import State


class DinoEncoder(Encoder):
    def __init__(self, state: State):
        super().__init__(state)
        self.model: DinoVisionTransformer = None
        self._setup_model()

    def __call__(
        self, images: List[Image], annotations: Optional[List[Annotations]] = None
    ) -> List[Features]:
        """
        This method creates an embedding from the images for locations inside the polygon.

        Args:
            images: A list of images.
            annotations: A list of a collection of annotations per image.

        Returns:
            A list of extracted features.
        """

        return [Features()]

    def _setup_model(self):
        """
        This method initializes the DINO model.
        """
        dinov2: DinoVisionTransformer = vision_transformer.__dict__["vit_large"](
            img_size=518,
            patch_size=14,
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
