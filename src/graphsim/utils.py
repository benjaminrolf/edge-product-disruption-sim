"""Utility functions."""

import json
import time
from functools import wraps
from typing import Any


def timeit(func: callable) -> callable:
    """Measure the execution time of a function."""

    @wraps(func)
    def timeit_wrapper(*args: tuple, **kwargs: dict[str, Any]) -> callable:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        return result

    return timeit_wrapper


def load_json(path: str) -> dict:
    """Load a JSON file."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)
