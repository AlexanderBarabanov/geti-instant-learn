from typing import List

import torch
import numpy as np
from context_learner.processes.process_base import Process
from context_learner.types.features import Features
from context_learner.types.image import Image
from context_learner.types.masks import Masks
from context_learner.types.priors import Priors
from context_learner.types.state import State


class Calculator(Process):
    def __init__(self, state: State):
        super().__init__(state)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
