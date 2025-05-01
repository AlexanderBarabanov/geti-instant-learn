# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import TYPE_CHECKING

from visionprompt.context_learner.types.data import Data

if TYPE_CHECKING:
    from visionprompt.context_learner.types.annotations import Annotations
    from visionprompt.context_learner.types.features import Features
    from visionprompt.context_learner.types.image import Image
    from visionprompt.context_learner.types.masks import Masks
    from visionprompt.context_learner.types.points import Points
    from visionprompt.context_learner.types.priors import Priors
    from visionprompt.context_learner.types.similarities import Similarities


class State(Data):
    """State object used in the Visual Prompting Pipeline."""

    def __init__(self) -> None:
        self.reference_images: list[Image] = []
        self.reference_priors: list[Priors] = []
        self.reference_features: list[Features] = []
        self.processed_reference_masks: list[Masks] = []
        self.target_images: list[Image] = []
        self.target_features: list[Features] = []
        self.similarities: list[Similarities] = []
        self.priors: list[Priors] = []
        self.masks: list[Masks] = []
        self.annotations: list[Annotations] = []
        self.used_points: list[Points] = []

        # Encoder configuration
        self.encoder_input_size: int = None
        self.encoder_patch_size: int = None
        self.encoder_feature_size: int = None

    def __repr__(self) -> str:
        """Represent the state object as a string."""
        # Show the state a little prettier
        s = ""
        for k, v in self.__dict__.items():
            if isinstance(v, list):
                if len(v) > 0:
                    s = f"{s}\n  {k}[{len(v)}]: List[{type(v[0]).__name__}]"
                else:
                    s = f"{s}\n  {k}[{len(v)}]: List[Any]"
            else:
                s = f"{s}\n  {k}: {type(v).__name__}"

        return f"In-context learning State object containing: {s}"
