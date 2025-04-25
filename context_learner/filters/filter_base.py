# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from context_learner.types.state import State


class Filter:
    def __init__(self, state: State):
        self._state = state
