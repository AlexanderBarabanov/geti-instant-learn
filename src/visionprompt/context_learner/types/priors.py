# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from visionprompt.context_learner.types.data import Data
from visionprompt.context_learner.types.masks import Masks
from visionprompt.context_learner.types.points import Points
from visionprompt.context_learner.types.prompts import Prompt


class Priors(Data):
    """This class represents priors for a single image.

    These can contain points, boxes, masks or polygons. They mainly serve as input for Segmentation models.
    """

    def __init__(
        self,
        points: Points | None = None,
        boxes: Prompt | None = None,
        masks: Masks | None = None,
        polygons: Prompt | None = None,
    ) -> None:
        self._points: Points = points if points is not None else Points()
        self._boxes: Prompt = boxes if boxes is not None else Prompt()
        self._masks: Masks = masks if masks is not None else Masks()
        self._polygons: Prompt = polygons if polygons is not None else Prompt()

    @property
    def points(self) -> Points:
        """Get the points."""
        return self._points

    @property
    def boxes(self) -> Prompt:
        """Get the boxes."""
        return self._boxes

    @property
    def masks(self) -> Masks:
        """Get the masks."""
        return self._masks

    @property
    def polygons(self) -> Prompt:
        """Get the polygons."""
        return self._polygons
