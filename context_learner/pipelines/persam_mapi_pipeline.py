from typing import List

from Matcher.segment_anything import SamPredictor
from context_learner.pipelines.pipeline_base import Pipeline
from context_learner.types.annotations import Annotations
from context_learner.types.image import Image
from context_learner.types.priors import Priors


class PerSamMAPI(Pipeline):
    """
    This is the PerSam algorithm pipeline using the ModelAPI implementation

    >>> p = PerSamMAPI()
    >>> p.learn([Image()] * 3, [Annotations()] * 3)
    >>> a = p.infer([Image()])
    >>> isinstance(a[0], Annotations)
    True
    """

    def __init__(self, sam_predictor: SamPredictor):
        super().__init__()


    def learn(self, reference_images: List[Image], reference_priors: List[Priors]):
        pass

    def infer(self, target_images: List[Image]):

        return Annotations()
