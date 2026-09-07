"""Action-chunk capability: commit several open-loop moves per VLM call when far.

Public API:
  * ``ActionChunkPlugin`` -- ``render_prompt`` adds the "plan your next moves" instruction and
    ``parse_plan(text, target_in_wrist)`` recovers the model's ordered ``PLAN:`` of up to
    ``step_num`` distinct ``MV_`` moves while the TARGET is far (else ``[]`` -> one move per
    call). Reuses the shared ``target_in_wrist`` signal (:mod:`core.prompting.wrist_marker`).

See :mod:`plugins.action_chunk.plugin` for the implementation.
"""
from .plugin import ActionChunkPlugin

__all__ = ["ActionChunkPlugin"]
