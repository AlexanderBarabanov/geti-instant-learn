from typing import List

import numpy as np
from model_api.models import SAMLearnableVisualPrompter
from model_api.models.visual_prompting import VisualPromptingFeatures

from context_learner.processes.segmenters.segmenter_base import Segmenter
from context_learner.types.image import Image
from context_learner.types.masks import Masks
from context_learner.types.priors import Priors
from context_learner.types.points import Points
from context_learner.types.state import State


class SamMAPIDecoder(Segmenter):
    """
    This is a wrapper around the ModelAPI SAM decoder.
    """

    def __init__(
        self,
        state: State,
        model: SAMLearnableVisualPrompter,
    ):
        super().__init__(state)
        self.model = model

    def __call__(
        self, images: List[Image], priors: List[Priors]
    ) -> tuple[List[Masks], List[Points]]:
        """Create masks from priors using SAM.

        Args:
            images: List of target images.
            priors: Priors are ignored, instead the features are taken directly from
                _state.reference_features.

        Returns:
            A tuple of a list of masks, one for each class in each target image,
            and a list of points, one for each class in each target image.
        """
        # Recreate feature object from the _state
        if len(self._state.reference_features) != 1:
            raise ValueError(
                "MAPISamDecoder only supports one set of global reference features"
            )
        reference = VisualPromptingFeatures(
            self._state.reference_features[0].global_features.numpy(),
            used_indices=np.array([0]),
        )

        masks_per_image: List[Masks] = []
        points_per_image: List[Points] = []

        for image in images:
            # Get results an stack into a single np array
            result = self.model.infer(
                image.to_numpy(),
                reference_features=reference,
                apply_masks_refinement=False,
            )
            mask = result.get_mask(0).mask
            mask = np.stack(mask)
            masks = Masks()
            for m in mask:
                masks.add(m, class_id=0)

            # Convert output into Points and Priors
            points = Points()
            ps = np.stack(result.data[0].points)
            scores = result.data[0].scores
            # Generate [x, y, score, label]
            # Note that Model API does not return the used background points
            points_scores = np.ones([len(ps), 4])
            points_scores[:, 0] = ps[:, 0]
            points_scores[:, 1] = ps[:, 1]
            points_scores[:, 2] = scores
            points.add(points_scores, class_id=0)

            # Add to return list
            points_per_image.append(points)
            masks_per_image.append(masks)

        return masks_per_image, points_per_image
