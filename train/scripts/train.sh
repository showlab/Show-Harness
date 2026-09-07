#!/usr/bin/env bash
# Launch one mvtoken LoRA training run. Shared entry point for all three model families.
#
# Runs upstream LlamaFactory's llamafactory-cli unpatched. Qwen3.5 and InternVL go through
# stock upstream with nothing injected; train/llamafactory_extensions/ is mounted only for
# gemma4 or when camera dropout is on (see that directory's README).
#
#   CONFIG=<yaml>       training config (required); relative paths resolve against this
#                       repo first, then LF_ROOT
#   GPU=0,1             CUDA_VISIBLE_DEVICES (default 0)
#   FAMILY=             qwen3_5 | internvl3_5 | gemma4; inferred from the config's `template:`
#   MODEL_PATH=         override model_name_or_path from the yaml
#   CAMERA_DROPOUT=0.15 blank a random subset of camera views per sample (default 0 = off)
#   WANDB_PROJECT=      wandb project (default llamafactory)
#   WANDB_ENTITY=       wandb entity; unset uses your logged-in default
#   MEDIA_DIR=          base dir for relative image paths; inferred when empty
#   OUTPUT_DIR=         where the run writes; default is the yaml's output_dir resolved
#                       under train/ (so train/saves/<model>/robot/<name>/)
#   SKIP_CHECKS=1       skip the dataset/media checks
#   DRY_RUN=1           run the checks, print the command, do not train
#
#   CONFIG=train/configs/my_run.yaml GPU=1,3 bash train/scripts/train.sh
#   CONFIG=train/configs/my_run.yaml CAMERA_DROPOUT=0.15 bash train/scripts/train.sh

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/llamafactory_env.sh"

CONFIG="${CONFIG:?CONFIG (training yaml) is required}"
GPU="${GPU:-0}"

# Prefer this repo (train/configs/), then LF_ROOT (upstream's own examples/).
if [ ! -f "$CONFIG" ]; then
  for _base in "$REPO_ROOT" "$LF_ROOT"; do
    [ -f "$_base/$CONFIG" ] && { CONFIG="$_base/$CONFIG"; break; }
  done
fi
[ -f "$CONFIG" ] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

# ── Model family: decides which venv to use ──────────────────────────────────
# Read the yaml's `template:` first. File names are unreliable once someone copies a
# template and renames it; fall back to the name only if the template is unknown.
if [ -z "${FAMILY:-}" ]; then
  _tpl="$(sed -n 's/^template:[[:space:]]*//p' "$CONFIG" | head -1 | tr -cd 'A-Za-z0-9._-')"
  case "$_tpl" in
    intern_vl*)        FAMILY=internvl3_5 ;;
    gemma4*)           FAMILY=gemma4 ;;
    qwen3_5*|qwen3.5*) FAMILY=qwen3_5 ;;
    *)
      case "$(basename "$CONFIG")" in
        *gemma4*|*gemma_4*) FAMILY=gemma4 ;;
        *internvl*)         FAMILY=internvl3_5 ;;
        *)                  FAMILY=qwen3_5 ;;
      esac ;;
  esac
fi

case "$FAMILY" in
  # gemma4 needs transformers>=5.10, which cannot share Qwen3.5's env.
  gemma4) VENV="${LF_VENV_GEMMA4:-${LF_ROOT}/.venv-gemma4}" ;;
  qwen3_5|internvl3_5) VENV="${LF_VENV}" ;;
  *) echo "ERROR: FAMILY must be qwen3_5 | internvl3_5 | gemma4, got: $FAMILY" >&2; exit 1 ;;
esac
[ -f "${VENV}/bin/activate" ] || {
  echo "ERROR: venv not found: ${VENV} (FAMILY=$FAMILY)" >&2
  echo "  fix: bash ${REPO_ROOT}/train/scripts/setup_llamafactory.sh" >&2
  exit 1; }

# Load-bearing on the gemma4 path: its venv runs transformers 5.12.1, past LlamaFactory's
# own declared bound (<=5.8.0), so the gate would abort the run. Harmless on the others.
export DISABLE_VERSION_CHECK=1

# tilelang JIT (Qwen3.5's GDN backward) needs a gcc whose cc1plus exists. CC_SHIM is
# located by llamafactory_env.sh; only prepend it once it actually compiles.
if [ -n "${CC_SHIM:-}" ] && echo 'int main(){return 0;}' | "${CC_SHIM}/gcc" -x c++ - -o /dev/null >/dev/null 2>&1; then
  export PATH="${CC_SHIM}:${PATH}"
fi

# ── wandb ────────────────────────────────────────────────────────────────────
# Upstream reads WANDB_PROJECT from the environment. But transformers' TrainingArguments
# carries its own `project` field (for Trackio, defaulting to the string "huggingface"),
# and forks that touched ReporterCallback use it as the wandb project, short-circuiting
# the env var - runs then land silently in a "huggingface" project. Set both.
export WANDB_PROJECT="${WANDB_PROJECT:-llamafactory}"

# ── Extensions outside upstream: mounted only when actually needed ───────────
# Mounted via PYTHONPATH + llamafactory_extensions/sitecustomize.py, which every torchrun
# worker inherits.
CAMERA_DROPOUT="${CAMERA_DROPOUT:-0}"
NEED_EXT=0
_ext_why=()
if awk "BEGIN{exit !($CAMERA_DROPOUT > 0)}" 2>/dev/null; then
  export MVTOKEN_CAMERA_DROPOUT="$CAMERA_DROPOUT"
  NEED_EXT=1; _ext_why+=("camera_dropout=$CAMERA_DROPOUT")
fi
if [ "$FAMILY" = "gemma4" ]; then
  NEED_EXT=1; _ext_why+=("gemma4_unified registration")
fi
if [ "$NEED_EXT" = 1 ]; then
  export PYTHONPATH="${REPO_ROOT}/train/llamafactory_extensions${PYTHONPATH:+:${PYTHONPATH}}"
  echo "mounting llamafactory_extensions: ${_ext_why[*]}"
fi

source "${VENV}/bin/activate"

# ── Locate the dataset, check it, and work out media_dir ────────────────────
# Distributable datasets store image paths relative to the sample file. LF joins them
# against its own data/ dir, and on a miss only warns and keeps the original path - the
# FileNotFoundError then lands mid-training. Datasets with absolute paths are unaffected.
MEDIA_DIR_AUTO="$(python - "$CONFIG" "$LF_ROOT" "${SKIP_CHECKS:-}" <<'PY'
import json, os, sys, re, pathlib

config_path, lf_root, skip = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
log = lambda *a: print(*a, file=sys.stderr)

m = re.search(r"^dataset:\s*(\S+)", config_path.read_text(encoding="utf-8"), re.M)
if not m:
    log("WARN: no `dataset:` in the yaml, skipping data checks"); raise SystemExit(0)
names = [n.strip().strip('"\'') for n in m.group(1).split(",") if n.strip()]

info = json.loads((lf_root / "data" / "dataset_info.json").read_text(encoding="utf-8"))
media_dirs, checked = set(), []
for name in names:
    if name not in info:
        sys.exit(f"ERROR: dataset [{name}] is not in dataset_info.json\n"
                 f"  register it: python train/data_preparation/register_dataset.py {name} --samples <rollouts.json>")

    fn = info[name]["file_name"]
    samples_path = pathlib.Path(fn) if os.path.isabs(fn) else lf_root / "data" / fn
    if not samples_path.is_file():
        sys.exit(f"ERROR: dataset file not found: {samples_path}")

    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    media = [p for s in samples[:200] for key in ("images", "videos")
             for item in s.get(key, []) for p in (item if isinstance(item, list) else [item])]

    # Relative paths: media_dir is the sample file's directory.
    media_dir = str(samples_path.parent) if (media and not os.path.isabs(media[0])) else ""
    media_dirs.add(media_dir)
    checked.append((name, len(samples), media_dir, media))

# media_dir is global in LF (DatasetAttr has no such field), so one run has exactly one
# value. Mixing relative-path datasets requires them to share a media root.
if len(media_dirs) > 1:
    sys.exit("ERROR: these datasets have different media roots, but media_dir is global:\n"
             + "\n".join(f"  {n}: {d or '(absolute paths)'}" for n, _, d, _ in checked)
             + "\n  Train them separately, or merge them into one sample file under a shared root.")
media_dir = media_dirs.pop() if media_dirs else ""

if skip:
    print(media_dir); raise SystemExit(0)

for name, n, _, media in checked:
    log(f"dataset [{name}]: {n} samples")
    missing = [p for p in media if not os.path.exists(os.path.join(media_dir, p) if media_dir else p)]
    if missing:
        sys.exit(f"ERROR: [{name}] {len(missing)} of the first 200 samples reference missing media, e.g. {missing[0]}")
    log(f"  media check passed (first 200 samples / {len(media)} files)")
if media_dir:
    log(f"relative image paths -> media_dir={media_dir}")
print(media_dir)
PY
)"
MEDIA_DIR="${MEDIA_DIR:-$MEDIA_DIR_AUTO}"

# Single GPU + deepspeed: upstream only switches to torchrun when it sees more than one
# GPU, and otherwise aborts with "Please use FORCE_TORCHRUN=1". Set it here.
if grep -qE '^\s*deepspeed:' "$CONFIG" && [[ "$GPU" != *,* ]]; then
  export FORCE_TORCHRUN=1
  echo "single GPU + deepspeed -> FORCE_TORCHRUN=1"
fi

CLI_ARGS=("$CONFIG" "project=${WANDB_PROJECT}")
[ -n "${MODEL_PATH:-}" ] && CLI_ARGS+=("model_name_or_path=${MODEL_PATH}")
# Do not override a media_dir already set in the yaml (CLI args win).
if [ -n "${MEDIA_DIR:-}" ] && ! grep -qE '^\s*media_dir:' "$CONFIG"; then
  CLI_ARGS+=("media_dir=${MEDIA_DIR}")
fi

# Checkpoints belong to this repo, not to the upstream checkout. The yamls carry a
# relative output_dir and LlamaFactory resolves it against its own cwd, which would bury
# runs in third_party/LlamaFactory/saves/ - inside the very directory "Upgrading upstream"
# tells you to delete. Resolve relative paths under train/ instead; an absolute output_dir
# in the yaml is deliberate and left alone.
if [ -z "${OUTPUT_DIR:-}" ]; then
  _yaml_out="$(sed -n 's/^output_dir:[[:space:]]*//p' "$CONFIG" | head -1)"
  _yaml_out="${_yaml_out%%#*}"                        # strip a trailing comment
  _yaml_out="$(printf '%s' "$_yaml_out" | tr -d '"'"'"' ' | tr -d "'")"   # and quotes/space
  case "$_yaml_out" in
    "" | /*) ;;                                        # unset, or already absolute
    *) OUTPUT_DIR="${REPO_ROOT}/train/${_yaml_out}" ;;
  esac
fi
[ -n "${OUTPUT_DIR:-}" ] && CLI_ARGS+=("output_dir=${OUTPUT_DIR}")

echo "FAMILY=${FAMILY}  GPU=${GPU}  config=$(basename "$CONFIG")"
[ -n "${OUTPUT_DIR:-}" ] && echo "output_dir=${OUTPUT_DIR}"
cd "$LF_ROOT"

if [ -n "${DRY_RUN:-}" ]; then
  echo "[dry-run] cd $LF_ROOT && CUDA_VISIBLE_DEVICES=${GPU}${FORCE_TORCHRUN:+ FORCE_TORCHRUN=1} llamafactory-cli train ${CLI_ARGS[*]}"
  exit 0
fi

exec env CUDA_VISIBLE_DEVICES="${GPU}" llamafactory-cli train "${CLI_ARGS[@]}"
