"""Reference-video replication (baseline): a demo video is distilled into an ordered
operation brief (arm, grasp part, destination) that the subgoal planner replicates.

Thin re-export of the public API. See :mod:`plugins.video_ref.plugin`.
"""
from plugins.video_ref.plugin import BRIEF_SCHEMA, VideoRefPlugin, load_video_ref_prompt

__all__ = ["VideoRefPlugin", "BRIEF_SCHEMA", "load_video_ref_prompt"]
