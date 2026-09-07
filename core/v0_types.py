from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    end_reason: str
    video_path: str
    run_dir: str


@dataclass
class Subgoal:
    id: str
    target: str
    affordance: str
    motion: str
    description: str
    completion: str

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int) -> "Subgoal":
        motion = _normalized_motion(data.get("motion"), index)
        affordance = str(data.get("affordance", "")).strip()
        if not affordance:
            raise ValueError(f"Subgoal {index} is missing required affordance")
        description = str(data.get("description") or "").strip()
        if not description:
            raise ValueError(f"Subgoal {index} is missing required description")
        return cls(
            id=str(data.get("id") or f"subgoal_{index}"),
            target=str(data.get("target") or ""),
            affordance=affordance,
            motion=motion,
            description=description,
            completion=_normalized_completion(
                motion=motion,
                target=str(data.get("target") or ""),
                affordance=affordance,
                completion=str(data.get("completion") or ""),
            ),
        )

    def to_prompt_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "target": self.target,
            "affordance": self.affordance,
            "motion": self.motion,
            "description": self.description,
            "completion": self.completion,
        }


def _normalized_completion(
    motion: str, target: str, affordance: str, completion: str
) -> str:
    text = completion.strip()
    return text


def _normalized_motion(value: Any, index: int) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Subgoal {index} is missing required motion")
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_").upper()
    if not normalized:
        raise ValueError(f"Subgoal {index} has invalid motion {text!r}")
    return normalized


@dataclass
class V0Config:
    max_subgoal_steps: int
    max_replans: int
    video_fps: float
    gripper_settle_steps: int = 1
    recover_descend_steps: int = 1
    controller_prompt_log_every: int = 20

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "V0Config":
        return cls(
            max_subgoal_steps=int(data.get("max_subgoal_steps", 45)),
            max_replans=int(data.get("max_replans", 1)),
            video_fps=float(data.get("video_fps", 2.0)),
            gripper_settle_steps=int(data.get("gripper_settle_steps", 1)),
            recover_descend_steps=int(data.get("recover_descend_steps", 1)),
            controller_prompt_log_every=int(
                data.get("controller_prompt_log_every", 20)
            ),
        )


@dataclass
class SkillContext:
    task: str
    subgoal: Subgoal
    subgoal_index: int
    step_idx: int
    subgoal_step_idx: int
    obs: dict[str, Any]
    agentview: np.ndarray
    wrist: Optional[np.ndarray]
    proprio: dict[str, Any]
    debug: bool


@dataclass
class SkillCommand:
    action: np.ndarray
    atomic_action: str
    primary_view: str
    action_source: str
    view_direction: Optional[str] = None
    supervisor_status: Optional[str] = None
    raw_supervisor_output: Optional[str] = None
    raw_monitor_output: Optional[str] = None
    stop_status: Optional[str] = None
    raw_stop_output: Optional[str] = None
    gripper_command_name: str = "HOLD"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillOutcome:
    stop_status: Optional[str] = None
    raw_stop_output: Optional[str] = None
    supervisor_status: Optional[str] = None
    raw_supervisor_output: Optional[str] = None
    subgoal_done: bool = False
    realign: bool = False
    episode_done: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
