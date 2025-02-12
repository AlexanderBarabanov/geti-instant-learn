from typing import List

from context_learner.processes.prompt_generators.prompt_generator_base import PromptGenerator
from context_learner.types.priors import Priors
from context_learner.types.similarities import Similarities


class GridPrompt(PromptGenerator):
    def __call__(self, similarities: List[Similarities]) -> List[Priors]:
        return [Priors()]
