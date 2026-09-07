"""Shared state for the simulator interpreters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AtomicControllerState:
    gripper_command: float
    gripper_name: str = "OPEN"
    last_atomic: Optional[str] = None
    closed_empty_width: Optional[float] = None
    open_width: Optional[float] = None
