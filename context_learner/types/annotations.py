import torch
from context_learner.types.data import Data

class Annotations(Data):
    def __init__(self):
        pass

    def from_tensor(self, points: torch.Tensor):
        pass