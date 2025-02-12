from typing import List, Optional

from context_learner.processes.encoders.encoder_base import Encoder
from context_learner.types.features import Features
from context_learner.types.image import Image
from context_learner.types.annotations import Annotations


class SamEncoder(Encoder):
    def __call__(self, images: List[Image], annotations: Optional[List[Annotations]] = None) -> List[Features]:
        return [Features()]
