"""Bound-informed, simulator-agnostic multi-objective design discovery."""

from .space import DesignSpace
from .workflow import Workflow, WorkflowConfig

__all__ = ["DesignSpace", "Workflow", "WorkflowConfig"]
__version__ = "0.1.2"
