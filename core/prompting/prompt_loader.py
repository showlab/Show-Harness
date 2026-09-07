from __future__ import annotations

from pathlib import Path
from typing import Any


# NOTE: the subgoal-planner prompt is NOT here -- it is owned by the subgoal capability
# (plugins/subgoal/subgoal_planner.txt) and loaded by that tool. Only prompts consumed by
# the shared runtime / controller live in the top-level prompts/ directory.
REQUIRED_PROMPT_FILES = {
    "common_context": "common_context.txt",
    "controller_prompt": "controller.txt",
}


def load_prompt_dir(path: str | Path) -> dict[str, Any]:
    prompt_dir = Path(path)
    if not prompt_dir.is_dir():
        raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")

    prompts: dict[str, Any] = {}
    for key, filename in REQUIRED_PROMPT_FILES.items():
        prompt_path = prompt_dir / filename
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Required prompt file missing: {prompt_path}")
        prompts[key] = prompt_path.read_text(encoding="utf-8").strip()
    # Optional per-backend controller variants: controller_<backend>.txt is loaded
    # as `controller_<backend>_prompt` (e.g. controller_gemma.txt). A backend with
    # no such file falls back to the default controller.txt.
    for prompt_path in sorted(prompt_dir.glob("controller_*.txt")):
        key = f"{prompt_path.stem}_prompt"
        prompts[key] = prompt_path.read_text(encoding="utf-8").strip()
    prompts["skill_prompts"] = _load_skill_prompts(prompt_dir / "skills")
    return prompts


def _load_skill_prompts(skills_dir: Path) -> dict[str, str]:
    if not skills_dir.is_dir():
        return {}
    prompts: dict[str, str] = {}
    for prompt_path in sorted(skills_dir.rglob("*.txt")):
        key = "_".join(prompt_path.relative_to(skills_dir).with_suffix("").parts)
        prompts[key] = prompt_path.read_text(encoding="utf-8").strip()
    return prompts
