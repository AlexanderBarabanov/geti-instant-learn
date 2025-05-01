# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from visionprompt.context_learner.processes import Process
from visionprompt.context_learner.types import Annotations, Masks


class MaskProcessor(Process):
    """This class processes masks to create annotations (polygons)."""

    def __call__(self, masks: list[Masks]) -> list[Annotations]:
        """This method extracts polygons from masks.

        Args:
            masks: A list of masks.

        Returns:
            A list of polygons that have been created from the masks.

        Examples:
            >>> from visionprompt.context_learner.types.state import State
            >>> state = State()
            >>> process_masks = MaskProcessor(state=state)
            >>> r = process_masks([Masks()])
        """
        return [Annotations()]
