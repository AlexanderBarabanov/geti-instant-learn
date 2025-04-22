from context_learner.processes import Process
from context_learner.types import State


class Calculator(Process):
    def __init__(self, state: State):
        super().__init__(state)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
