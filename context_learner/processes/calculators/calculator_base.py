# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from context_learner.processes import Process
from context_learner.types import State


class Calculator(Process):
    def __init__(self, state: State):
        super().__init__(state)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
