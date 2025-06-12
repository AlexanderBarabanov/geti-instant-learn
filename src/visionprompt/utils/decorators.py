import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from visionprompt.context_learner.types.results import Results


def time_call(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to time the execution of a method and store its duration on the instance."""

    @wraps(func)
    def duration_wrapper(self, *args, **kwargs) -> Any:  # noqa: ANN001
        start_time = time.time()
        result = func(self, *args, **kwargs)
        duration = time.time() - start_time
        self.last_duration = duration
        return result

    return duration_wrapper


def track_duration(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to reset durations, run the pipeline method, and log timing."""

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:  # noqa: ANN001
        method_name = func.__name__.capitalize()
        self._reset_process_durations()
        result = func(self, *args, **kwargs)
        total_time = self.log_timing(title=method_name)
        if isinstance(result, Results):
            result.duration = total_time
        return result

    return wrapper
