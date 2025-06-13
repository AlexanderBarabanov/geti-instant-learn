"""Base class for all pipelines."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from logging import getLogger

from visionprompt.processes.preprocessors import ResizeImages, ResizeMasks
from visionprompt.processes.process_base import Process
from visionprompt.types import Image, Priors, Results

logger = getLogger("Vision Prompt")


class Pipeline(ABC):
    """This class is the base class for all pipelines.

    Examples:
        >>> from visionprompt.pipelines import Pipeline
        >>> from visionprompt.types import Image, Priors, Results
        >>>
        >>> class MyPipeline(Pipeline):
        ...     def learn(self, reference_images: list[Image], reference_priors: list[Priors]) -> Results:
        ...         return Results()
        ...     def infer(self, target_images: list[Image]) -> Results:
        ...         return Results()
        >>>
        >>> my_pipeline = MyPipeline(image_size=512)
        >>> my_pipeline.learn([Image()], [Priors()])
        >>> results = my_pipeline.infer([Image()])
    """

    def __init__(self, image_size: int | tuple[int, int] | None = None) -> None:
        """Initialization method that caches all parameters.

        Args:
            image_size: The size of the image to use, if None, the image will not be resized.
        """
        self.resize_images = ResizeImages(size=image_size)
        self.resize_masks = ResizeMasks(size=image_size)

    @abstractmethod
    def learn(self, reference_images: list[Image], reference_priors: list[Priors]) -> Results:
        """This method learns the context.

        Args:
            reference_images: A list of images ot learn from.
            reference_priors: A list of priors associated with the image.

        Returns:
            None
        """

    @abstractmethod
    def infer(self, target_images: list[Image]) -> Results:
        """This method uses the learned context to infer object locations.

        Args:
            target_images: A List of images to infer.

        Returns:
            None
        """

    def _get_process_durations(self) -> list[tuple[str, float]]:
        """Get the durations of the processes.

        Returns:
            A list of tuples containing the name of the component and the duration.
        """
        return [
            (attr_value.__class__.__name__, attr_value.last_duration)
            for attr_value in self.__dict__.values()
            if isinstance(attr_value, Process) and hasattr(attr_value, "last_duration")
        ]

    def _reset_process_durations(self) -> None:
        """Reset the durations of the processes."""
        for attr_value in self.__dict__.values():
            if isinstance(attr_value, Process):
                attr_value.last_duration = 0.0

    def log_timing(self, title: str = "Inference") -> float:
        """Print the timing of the processes in a table.

        Args:
            title: The title of the table.

        Returns:
            The total time of the processes.
        """
        process_durations = self._get_process_durations()
        output_str = f"\n--- {title} Timings ---"
        max_name_len = max((len(name) for name, _ in process_durations), default=0)
        max_name_len = max(max_name_len, len("Total"))

        total_time = sum(t for _, t in process_durations)
        output_str += f"\n{'Name':<{max_name_len}} | {'Total':<10}"
        output_str += "\n" + "-" * (max_name_len + 10 + 20 + 3)
        for name, duration in process_durations:
            output_str += f"\n{name:<{max_name_len}} | {duration:<10.4f}"
        output_str += "\n" + "-" * (max_name_len + 10 + 20 + 3)
        output_str += f"\n{'Total':<{max_name_len}} | {total_time:<10.4f}"
        output_str += "\n" + "-" * (max_name_len + 10 + 20 + 3) + "\n"
        logger.debug(output_str)
        return total_time
