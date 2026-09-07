from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

# Default location of the gitignored secrets file (API keys). Shell helpers that
# source secrets read the same path so the bash and Python sides agree.
DEFAULT_SECRETS_ENV = Path(__file__).resolve().parents[1] / "configs" / "secrets.env"


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load one YAML mapping, resolving its optional layer lists.

    A config may compose other files around its own body::

        defaults:                      # merged UNDER the body (body wins)
          - site/franka.yaml           #   rig identity a new install must provide
          - {path: backends/internal.yaml, optional: true}
        overlays:                      # merged OVER the body (overlay wins)
          - {path: experiments/current_franka.yaml, optional: true}

    ``defaults`` hold shared or site-provided base layers the body may refine;
    calibration scripts that write into the body therefore always take effect.
    ``overlays`` carry state that outranks the file (e.g. an active experiment).
    Paths are relative to the declaring file; entries apply in order (later
    wins within each list) via :func:`deep_merge`; layered files may declare
    layers of their own. A missing required layer raises with guidance; mark
    entries ``optional: true`` when their absence is a valid configuration.
    """
    return _load_yaml(Path(path), seen=())


def _load_yaml(path: Path, seen: tuple[str, ...]) -> Dict[str, Any]:
    resolved = str(path.resolve())
    if resolved in seen:
        chain = " -> ".join(seen + (resolved,))
        raise ValueError(f"config layer cycle: {chain}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")

    def _layers(key: str) -> list[Dict[str, Any]]:
        entries = data.pop(key, None) or []
        if not isinstance(entries, list):
            raise ValueError(f"{path}: {key!r} must be a list")
        loaded: list[Dict[str, Any]] = []
        for entry in entries:
            optional = False
            if isinstance(entry, dict):
                optional = bool(entry.get("optional", False))
                ref = str(entry.get("path", ""))
            else:
                ref = str(entry)
            if not ref:
                raise ValueError(f"{path}: {key} entry without a path: {entry!r}")
            layer_path = (path.parent / ref).resolve()
            if not layer_path.is_file():
                if optional:
                    continue
                raise FileNotFoundError(
                    f"{path} requires config layer {ref!r} which does not exist. "
                    f"If this is a site file, copy {ref}.example to {ref} and "
                    "fill in your rig's values."
                )
            loaded.append(_load_yaml(layer_path, seen + (resolved,)))
        return loaded

    under = _layers("defaults")
    over = _layers("overlays")  # popped before the body merges
    merged: Dict[str, Any] = {}
    for layer in under:
        merged = deep_merge(merged, layer)
    merged = deep_merge(merged, data)
    for layer in over:
        merged = deep_merge(merged, layer)
    return merged


def load_secrets_env(
    path: str | Path | None = None, *, override: bool = False
) -> Dict[str, str]:
    """Load ``KEY=VALUE`` pairs from ``configs/secrets.env`` into ``os.environ``.

    Dependency-free dotenv parser (no python-dotenv): blank lines and ``#`` comments
    are skipped, an optional ``export`` prefix is allowed, surrounding single/double
    quotes are stripped, and whitespace around ``=`` is tolerated (``KEY = "v"``).
    A missing file is not an error (returns ``{}``). Existing environment variables
    are preserved unless ``override=True``, so a value exported in the shell wins.
    With the default path, an optional gitignored per-machine overlay
    ``configs/secrets.local.env`` is read on top (its keys win over the shared file).

    Returns the mapping that was parsed from the file (regardless of whether it was
    written to the environment), so callers can report which keys are available.
    """
    if path is None:
        # Default flow: the shared file, then an optional per-machine overlay
        # (gitignored, same format). Overlay keys win over the shared file;
        # shell-exported variables still win over both.
        parsed = _parse_env_file(DEFAULT_SECRETS_ENV)
        parsed.update(_parse_env_file(DEFAULT_SECRETS_ENV.with_name("secrets.local.env")))
        for key, value in parsed.items():
            if override or key not in os.environ:
                os.environ[key] = value
        return parsed
    env_path = Path(path)
    parsed = _parse_env_file(env_path)
    for key, value in parsed.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


def _parse_env_file(env_path: Path) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    if not env_path.is_file():
        return parsed
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not key:
            continue
        parsed[key] = value
    return parsed


def resolve_api_key(vlm_cfg: Dict[str, Any]) -> str:
    """Resolve the API key for a (resolved) vlm config without baking secrets into YAML.

    Precedence: the environment variable named by ``api_key_env`` (e.g.
    ``CHATGPT_API_KEY`` / ``GEMINI_API_KEY``, populated from configs/secrets.env), then
    a literal ``api_key`` in the config (legacy), then the ``VLLM_API_KEY`` environment
    variable, then ``"EMPTY"`` (which is what a local vLLM server expects).
    """
    env_name = vlm_cfg.get("api_key_env")
    if env_name:
        value = os.environ.get(str(env_name))
        if value:
            return value
    literal = vlm_cfg.get("api_key")
    if literal:
        return str(literal)
    return os.environ.get("VLLM_API_KEY") or "EMPTY"


def make_api_key_refresher(vlm_cfg: Dict[str, Any]):
    """Build a zero-arg callable that returns the CURRENT API key for this backend.

    A long rollout resolves its key once at startup, but a short-lived bearer
    token can expire mid-run even while an external refresher keeps writing fresh
    ones to configs/secrets.env.
    The refresher re-parses secrets.env and installs the fresh value for THIS
    backend's ``api_key_env`` into os.environ (other keys keep the documented
    shell-wins precedence), then re-runs the normal resolution. The VLM client
    calls it on an HTTP 401/403 and retries with the new key instead of crashing.
    """
    env_name = vlm_cfg.get("api_key_env")

    def _refresh() -> str:
        if env_name:
            parsed = load_secrets_env()
            value = parsed.get(str(env_name))
            if value:
                os.environ[str(env_name)] = value
        return resolve_api_key(vlm_cfg)

    return _refresh


def resolve_env_field(vlm_cfg: Dict[str, Any], field: str) -> None:
    """Override a resolved VLM field from an env var named by ``<field>_env``.

    This is useful for non-secret deployment details that may differ across accounts,
    e.g. a hosted deployment id. The YAML can declare ``model_env: MY_MODEL``
    and keep a documented fallback ``model`` value.
    """
    env_name = vlm_cfg.get(f"{field}_env")
    if not env_name:
        return
    value = os.environ.get(str(env_name))
    if value:
        vlm_cfg[field] = value


#: The camera transform contract, as it is spelled in every ``configs/robot_*.yaml``.
#: Maps config key -> the ``core.record.images.prepare_view`` argument it feeds, per view.
CAMERA_CONTRACT_KEYS = (
    "agentview_camera",
    "wrist_camera",
    "agentview_rotation_degrees",
    "wrist_rotation_degrees",
    "agentview_flip",
    "wrist_flip",
    "agentview_crop_aspect",
    "wrist_crop_aspect",
    "agentview_square_size",
    "wrist_square_size",
)


def camera_contract(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the camera transform contract from a robot config, as backend kwargs.

    Training frames and inference frames must be byte-identical, which means the
    DEPLOYMENT runner and the real2sim GENERATOR have to apply the same rotation, flip,
    crop and camera selection. They used to state it twice -- the runner read
    ``configs/robot_<sim>.yaml`` while the generators carried their own argparse
    defaults -- and the two drifted apart the moment either side changed:

      * ``agentview_camera`` moved to ``front_cam`` in the config, but the generator's
        default still named ``over_shoulder_left_camera``, so generation died with a
        ``KeyError`` on a camera the scene no longer had (the loud, lucky failure);
      * the wrist rotation was re-measured for the panda hand in the config, and the
        generator kept storing frames at the old Robotiq angle (the silent, expensive
        one -- a whole dataset 90 deg off, invisible in every aggregate statistic).

    So the config is THE source of truth and both sides read it through here. Keys absent
    from ``cfg`` are omitted rather than defaulted, letting the backend's own signature
    supply the fallback -- a config that says nothing changes nothing.

    ``*_square_size`` is included: the letterbox is part of the contract like everything
    else, and it is applied by the backend to BOTH views (see
    ``atomic_tokenizer.prepared_pair``). It used to live in ``RolloutWriter`` and to touch
    the agentview only, which is how both simulators ended up storing a wrist that filled
    the frame while every real training wrist has 32 rows of letterbox top and bottom.
    """
    return {k: cfg[k] for k in CAMERA_CONTRACT_KEYS if k in cfg}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_vlm_config(cfg: Dict[str, Any], backend: str | None = None) -> Dict[str, Any]:
    """Merge the selected ``vlm_backends`` profile (model / max_tokens /
    chat_template_kwargs) onto the shared ``vlm`` block and return it.

    The shared ``vlm`` block holds only connection fields; per-model fields live
    in ``vlm_backends`` so models can be swapped with one selector. The
    returned dict always carries a ``backend`` key naming the active profile.
    """
    vlm = dict(cfg.get("vlm", {}))
    backends = cfg.get("vlm_backends") or {}
    name = backend or cfg.get("vlm_backend") or "gemma"
    if backends:
        if name not in backends:
            raise ValueError(
                f"Unknown vlm_backend {name!r}; choices: {sorted(backends)}"
            )
        profile = backends[name]
        if not isinstance(profile, dict):
            raise ValueError(f"vlm_backends.{name} must be a mapping")
        vlm = deep_merge(vlm, profile)
    resolve_env_field(vlm, "base_url")
    resolve_env_field(vlm, "model")
    if "model" not in vlm:
        raise ValueError(
            f"Resolved vlm config for backend {name!r} has no 'model'; "
            "define it under vlm_backends or vlm."
        )
    vlm["backend"] = name
    # Resolve the key from the environment (api_key_env, populated by
    # load_secrets_env) so no secret has to live in the git-tracked YAML.
    vlm["api_key"] = resolve_api_key(vlm)
    return vlm
