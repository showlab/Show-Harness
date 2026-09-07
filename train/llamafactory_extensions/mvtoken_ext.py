"""A layer on top of upstream LlamaFactory - its source is never modified.

mvtoken training needs two things upstream does not provide. Both can be attached from
the outside, which is why this repo depends on UPSTREAM rather than anyone's fork:

  1. The ``gemma4_unified`` composite-model registration. Upstream already ships
     Gemma4Plugin and the gemma4/gemma4n templates; it just has no entry for this
     model_type's module layout. ``COMPOSITE_MODELS`` is a module-level dict and
     ``_register_composite_model()`` is upstream's own writer for it.
  2. Camera dropout - blanking a random subset of camera views during training. It hooks
     in before the collator sees the features, so it only edits inputs and leaves
     upstream's collation untouched.

Both are opt-in: training Qwen3.5 or InternVL loads this module and it does nothing.

See sitecustomize.py for how this gets loaded - multi-GPU runs re-launch workers through
torchrun, so the extension has to attach at interpreter startup, not in the parent.

Debug: MVTOKEN_EXT_DEBUG=1 prints what was installed.
"""
from __future__ import annotations

import os
import random


def _debug(msg: str) -> None:
    if os.getenv("MVTOKEN_EXT_DEBUG"):
        print(f"[mvtoken_ext] {msg}", flush=True)


# ── 1. gemma4_unified ───────────────────────────────────────────────────────

def register_gemma4_unified() -> bool:
    """Register the module layout of released gemma-4 checkpoints.

    Released gemma-4 weights report ``model_type: gemma4_unified``, whose layout differs
    from the gemma4 entry upstream ships: the vision backbone is ``model.vision_embedder``
    (no ``vision_tower``) and the audio side is projection-only (no ``audio_tower``).
    Without this entry ``freeze_vision_tower`` finds nothing to freeze.
    """
    try:
        from llamafactory.model.model_utils.visual import COMPOSITE_MODELS, _register_composite_model
    except ImportError as e:
        _debug(f"gemma4_unified: skipped (import failed: {e})")
        return False

    if "gemma4_unified" in COMPOSITE_MODELS:
        _debug("gemma4_unified: already registered upstream")   # yields once upstream merges it
        return False

    _register_composite_model(
        model_type="gemma4_unified",
        projector_keys=["model.embed_vision", "model.embed_audio"],
        vision_model_keys=["vision_embedder"],
        lora_conflict_keys=["per_layer_projection_norm"],
    )
    _debug("gemma4_unified: registered")
    return True


# ── 2. camera dropout ───────────────────────────────────────────────────────

def _blank_like(image):
    """A black image of the same size.

    PIL opens lazily (header only), so this does not decode an image about to be dropped.
    """
    from io import BytesIO

    from PIL import Image

    if isinstance(image, dict):
        image = image["bytes"] if image.get("bytes") is not None else image["path"]

    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    elif isinstance(image, str):
        image = Image.open(image)

    return Image.new("RGB", image.size, (0, 0, 0))


def _drop_cameras(images: list, p: float) -> list:
    """Blank each view independently with probability p, never all of them.

    Multi-camera policies tend to collapse onto whichever view is easiest to read, then
    fall apart when that view goes uninformative (a wrist camera against a featureless
    table) or is missing at deployment. The replacement is the SAME SIZE, so the visual
    token count is unchanged and the sequence length computed during preprocessing still
    holds. Applied per batch, so a sample gets a fresh mask every epoch.
    """
    keep = [random.random() >= p for _ in images]
    if not any(keep):
        keep[random.randrange(len(images))] = True
    if all(keep):
        return images
    return [img if k else _blank_like(img) for img, k in zip(images, keep)]


def install_camera_dropout(p: float) -> bool:
    """Hook camera dropout in ahead of upstream's collator.

    Only ``feature["images"]`` is edited before upstream's ``__call__`` runs; its collation
    logic is untouched. Patching the base class is enough:
    ``SFTDataCollatorWith4DAttentionMask`` goes through ``super().__call__()``.
    """
    if p <= 0.0:
        return False

    try:
        from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq as C
    except ImportError as e:
        _debug(f"camera_dropout: skipped (import failed: {e})")
        return False

    if getattr(C, "_mvtoken_camera_dropout_installed", False):
        return False

    original_call = C.__call__

    def patched_call(self, features):
        for feature in features:
            images = feature.get("images") or []
            if len(images) > 1:   # nothing to drop from a single view
                feature["images"] = _drop_cameras(images, p)
        return original_call(self, features)

    C.__call__ = patched_call
    C._mvtoken_camera_dropout_installed = True
    _debug(f"camera_dropout: installed (p={p})")
    return True


# ── Entry point ─────────────────────────────────────────────────────────────

def apply() -> None:
    """Install whatever this run needs. A no-op when nothing is enabled."""
    register_gemma4_unified()

    try:
        p = float(os.getenv("MVTOKEN_CAMERA_DROPOUT", "0") or 0)
    except ValueError:
        p = 0.0
        _debug("camera_dropout: MVTOKEN_CAMERA_DROPOUT is not a number, treating as 0")
    install_camera_dropout(p)
