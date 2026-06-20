"""Tasks: the training-objective plug point."""

from __future__ import annotations

from synbiotorch.tasks.base import Task, build_task
from synbiotorch.tasks.supervised import SupervisedTask

__all__ = ["Task", "build_task", "SupervisedTask"]
