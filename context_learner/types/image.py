import torch
import numpy as np
from context_learner.types.data import Data


class Image(Data):
    def __init__(self):
        pass

    def from_tensor(self, image: torch.Tensor):
        pass

    def from_ndarray(self, image: np.ndarray):
        pass
