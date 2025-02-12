from typing import List

from context_learner.types.annotations import Annotations
from context_learner.types.data import Data
from context_learner.types.image import Image
from context_learner.types.masks import Masks
from context_learner.types.priors import Priors
from context_learner.types.similarities import Similarities


class State(Data):
    def __init__(self):
        self.reference_images: List[Image] = []
        self.reference_annotations: List[Annotations] = []
        self.target_features: List[Annotations] = []
        self.similarities: List[Similarities] = []
        self.priors: List[Priors] = []
        self.masks: List[Masks] = []
        self.annotations: List[Annotations] = []

    def __repr__(self):
        # Show the state a little prettier
        s = ""
        for k, v in self.__dict__.items():
            if isinstance(v, List):
                if len(v) > 0:
                    s = f"{s}\n  {k}[{len(v)}]: List[{type(v[0]).__name__}]"
                else:
                    s = f"{s}\n  {k}[{len(v)}]: List[Any]"
            else:
                s = f"{s}\n  {k}: {type(v).__name__}"

        return f"In-context learning State object containing: {s}"


