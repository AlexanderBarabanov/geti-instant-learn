# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from visionprompt.context_learner.types.state import State


class Process:
    """This is the base class of processes within pipelines.

    Typically classes that inherit from this method implement a __call__ method that
    accepts lists of objects. Each index in the list represents the image that it came from,
    so a List[Features] represents multiple features per image because it was generated using the
    Encoder that used a List[Image] as an input.
    """

    def __init__(self, state: State) -> None:
        """This initializes the process.

        Typically the process should not manipulate the state directly, it is passed for
        convenience to read earlier results. Consider creating extra inputs or outputs in
        the inherited __call__ method if the state is important for the pipeline's flow.

        Args:
            state: The state of the parent pipeline

        Returns:
            None

        Examples:
            >>> p = Process(State())
        """
        self._state = state
