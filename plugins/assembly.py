"""Shared construction for the plugins that do not depend on the inference mode.

``dagger``, ``video_ref`` and ``auto_release`` behave identically whether the
runner drives the zero-shot planner or a fine-tuned mvtoken policy: they read
the config, the session and the live view, nothing else. Every mode entry point
used to build them inline and the copies drifted -- diverging warning text, and a
viewer guard that checked ``viewer is None`` in the dual runners but not in the
single-arm ones. These builders are the one place they are constructed, so a
change to their wiring reaches every mode at once.

Runners still construct their plugins explicitly: this is a shared constructor,
not a registry -- nothing here decides WHICH plugins a mode runs.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from plugins.auto_release import AutoReleasePlugin
from plugins.config import PluginsConfig
from plugins.dagger import DaggerPlugin

# DAGGER keys arrive on the live-view window's STREAM thread, so it needs both the
# window and a session that can feed it.
DAGGER_NO_STREAM = (
    "dagger ignored: this session type has no live camera stream "
    "(the keys are pumped on the streaming window's render thread)"
)
DAGGER_NEEDS_WINDOW = (
    "plugins.dagger reads keys from the live view window; run without --no-show."
)


def install_dagger(
    plugins: PluginsConfig,
    viewer: Any,
    *,
    single: bool = False,
    session: Any = None,
    hardware: Optional[str] = None,
    notify: Optional[Callable[[str], None]] = None,
) -> DaggerPlugin:
    """Build the DAGGER plugin, check it against the viewer, and install it.

    Single-arm runners pass ``single=True`` plus ``session``/``hardware``: the
    plugin is ignored on a session that cannot stream frames (the single-Piper
    one), and the rotate keys follow ``plugins.rotation`` off Piper. Dual-arm
    runners keep the dual key layout and the default rotate keys.

    ``notify`` renders the ignored-plugin warning in the caller's own style.
    """
    if single:
        asked = plugins.enabled("dagger", default=False)
        supported = hasattr(session, "get_camera_frames")
        if asked and not supported:
            (notify or print)(DAGGER_NO_STREAM)
        plugin = DaggerPlugin(
            enabled=asked and supported,
            include_rotate_keys=(
                plugins.enabled("rotation", default=False) and hardware != "piper"
            ),
            single=True,
        )
    else:
        plugin = DaggerPlugin(enabled=plugins.enabled("dagger", default=False))
    if plugin.enabled and (viewer is None or not viewer.enabled):
        raise ValueError(DAGGER_NEEDS_WINDOW)
    plugin.install(viewer)
    return plugin


def build_video_ref(
    plugins: PluginsConfig,
    cfg: dict,
    video_ref_arg: Any = None,
    *,
    single: bool = False,
) -> Any:
    """Build the reference-video plugin; ``--video-ref`` is itself the opt-in.

    Config errors raise HERE, before any VLM cost.
    """
    from plugins.video_ref import VideoRefPlugin  # lazy: pulls imageio/numpy/PIL

    plugin = VideoRefPlugin(
        enabled=plugins.enabled("video_ref", default=False) or bool(video_ref_arg),
        video_path=video_ref_arg or cfg.get("video_ref_path"),
        num_frames=int(cfg.get("video_ref_frames", 8)),
        single=single,
    )
    if plugin.enabled and plugin.video_path is None:
        raise ValueError(
            "plugins.video_ref is enabled but no reference video was given: pass "
            "--video-ref <path> or set video_ref_path in the robot config."
        )
    if plugin.enabled and not plugin.video_path.is_file():
        raise ValueError(f"video_ref: reference video not found: {plugin.video_path}")
    return plugin


def build_auto_release(
    plugins: PluginsConfig,
    cfg: dict,
    *,
    default_empty_width_m: float = 0.001,
) -> AutoReleasePlugin:
    """Reopen a closed gripper whose measured width says it is holding nothing.

    ``cfg`` is the scope that owns ``empty_width_m`` -- the run config for one arm,
    or that arm's sub-config in a dual setup.
    """
    return AutoReleasePlugin(
        enabled=plugins.enabled("auto_release", default=True),
        empty_width_m=float(cfg.get("empty_width_m", default_empty_width_m)),
    )
