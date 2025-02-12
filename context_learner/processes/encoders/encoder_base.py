from typing import List, Optional

from context_learner.processes.process_base import Process
from context_learner.types.features import Features
from context_learner.types.image import Image
from context_learner.types.annotations import Annotations


class Encoder(Process):
    def __call__(self, images: List[Image], annotations: Optional[List[Annotations]] = None) -> List[Features]:
        """
        This method creates an embedding from the images for locations inside the polygon.

        Args:
            images: A list of images.
            annotations: A list of a collection of annotations per image.

        Returns:
            A list of extracted features.

        Examples:
            >>> from context_learner.types.state import State
            >>> state = State()
            >>> enc = Encoder(state=state)
            >>> r = enc([Image()], [Annotations()])
        """
        return [Features()]
