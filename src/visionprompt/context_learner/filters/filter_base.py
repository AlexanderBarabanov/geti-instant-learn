# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from visionprompt.context_learner.types.state import State


class Filter:
    """This is the base class for all filters."""

    def __init__(self, state: State) -> None:
        """Initialize the filter."""
        self._state = state
