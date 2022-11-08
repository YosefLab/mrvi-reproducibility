import json
import pathlib
from inspect import signature
from typing import Callable, List, Union

import click


def make_parents(path: Union[str, List[str]]) -> None:
    """Make parent directories of a file path if they do not exist."""
    if isinstance(path, str):
        path = [path]
    for p in path:
        pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)


def wrap_kwargs(fn: Callable) -> Callable:
    """Wrap a function to accept keyword arguments from the command line."""
    for param in signature(fn).parameters:
        fn = click.option("--" + param, type=str)(fn)
    return click.command()(fn)


def load_config(config_path: str) -> dict:
    """Load a JSON configuration file as a Python dictionary."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
