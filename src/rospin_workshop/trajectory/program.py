from __future__ import annotations

import hashlib
import importlib.util
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from rospin_workshop.tasks import TASK_ID_PATTERN

if TYPE_CHECKING:
    from rospin_workshop.trajectory.runner import EpisodeContext


@dataclass(frozen=True)
class TrajectoryProgram:
    """A participant-authored episode program and its task binding."""

    task_id: str
    function: Callable[[EpisodeContext], None]
    name: str


def trajectory(*, task: str) -> Callable[[Callable[[EpisodeContext], None]], TrajectoryProgram]:
    """Mark a single-argument function as a workshop trajectory program."""

    if not TASK_ID_PATTERN.fullmatch(task):
        raise ValueError("trajectory task must be a valid task id")

    def decorate(function: Callable[[EpisodeContext], None]) -> TrajectoryProgram:
        parameters = list(inspect.signature(function).parameters.values())
        if len(parameters) != 1:
            raise TypeError("A trajectory function must accept exactly one context")
        return TrajectoryProgram(
            task_id=task,
            function=function,
            name=function.__name__,
        )

    return decorate


def load_trajectory_program(path: str | Path, root: Path) -> TrajectoryProgram:
    """Load exactly one decorated program from the mounted trajectory root."""

    root = Path(root).resolve()
    requested = Path(path)
    candidate = requested.resolve() if requested.is_absolute() else (root / requested).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Trajectory programs must be inside the trajectory directory") from exc
    if candidate.suffix != ".py" or not candidate.is_file():
        raise FileNotFoundError(f"Trajectory program does not exist: {candidate}")

    fingerprint = hashlib.sha256(
        f"{candidate}:{candidate.stat().st_mtime_ns}".encode()
    ).hexdigest()[:16]
    module_name = f"rospin_participant_trajectory_{fingerprint}"
    spec = importlib.util.spec_from_file_location(module_name, candidate)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load trajectory program: {candidate}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    programs = {
        id(value): value
        for value in vars(module).values()
        if isinstance(value, TrajectoryProgram)
    }
    if len(programs) != 1:
        raise ValueError(
            f"{candidate.name} must define exactly one @trajectory function"
        )
    return next(iter(programs.values()))
