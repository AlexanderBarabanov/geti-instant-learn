# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List

from context_learner.filters import Filter
from context_learner.types import Similarities


class SimilarityFilter(Filter):
    def __call__(self, priors: List[Similarities]) -> List[Similarities]:
        return priors
