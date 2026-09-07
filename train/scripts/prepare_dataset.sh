#!/usr/bin/env bash
# Convert a batch of rollouts into a training set and register it.
#
# SRC points at a PARENT directory: one subdirectory per task, each holding rollout_000/,
# rollout_001/, ... Each task uses the task_text from its own metadata.json, which is the
# language signal multi-task training needs. Samples carry absolute image paths.
#
#   SRC=<dir>      parent of the task directories (required)
#   NAME=<name>    dataset name, also the key in dataset_info.json (required)
#   VERSION=v3     prompt version, i.e. prompts/<version>/ (default v3)
#   EMBODIMENT=    extra flag for the converter: franka | piper | empty
#   TASK=          one instruction for every rollout (skips metadata.json lookup)
#   OUT=           output dir (default <repo>/train/data/$NAME)
#
#   SRC=$REPO_ROOT/rollouts/robolab/GEN3 NAME=robolab_0816_12task \
#       bash train/scripts/prepare_dataset.sh
#   SRC=... NAME=piper_0705 VERSION=v4 EMBODIMENT=piper bash train/scripts/prepare_dataset.sh
#
# Re-running overwrites rollouts.json. Overwriting it mid-training desynchronizes LF's
# dataset fingerprint from the data on disk - stop the run before rebuilding.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/llamafactory_env.sh"

SRC="${SRC:?SRC (parent of the task directories) is required}"
NAME="${NAME:?NAME (dataset name) is required}"
VERSION="${VERSION:-v3}"
EMBODIMENT="${EMBODIMENT:-}"
OUT="${OUT:-${REPO_ROOT}/train/data/${NAME}}"

PREP_DIR="${REPO_ROOT}/train/data_preparation"
CONV="${PREP_DIR}/rollouts_to_alpaca.py"
PY="${DATA_PREP_PYTHON}"

[ -d "$SRC" ] || { echo "ERROR: SRC not found: $SRC" >&2; exit 1; }
[ -d "${REPO_ROOT}/prompts/${VERSION}" ] || {
  echo "ERROR: prompt version dir not found: ${REPO_ROOT}/prompts/${VERSION}" >&2; exit 1; }

EMB_ARGS=()
case "$EMBODIMENT" in
  franka) EMB_ARGS=(--franka) ;;
  piper)  EMB_ARGS=(--piper) ;;
  "")     ;;
  *) echo "ERROR: EMBODIMENT must be franka | piper | empty, got: $EMBODIMENT" >&2; exit 1 ;;
esac

echo "[prep] source: $SRC"
echo "[prep] dataset: $NAME (prompt=$VERSION${EMBODIMENT:+, $EMBODIMENT}) -> $OUT"

mkdir -p "$OUT"
# Collect each task directory with its instruction, then convert in one call: the
# converter takes multiple directories and --task-map assigns per-directory instructions.
dirs=(); maps=()
for d in "$SRC"/*/; do
  T="$(basename "$d")"
  [ -d "$d/rollout_000" ] || { echo "skip $T (no rollout_000)"; continue; }
  if [ -n "${TASK:-}" ]; then
    TASK_TEXT="$TASK"
  else
    TASK_TEXT="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['task_text'])" \
                 "$d/rollout_000/metadata.json")"
  fi
  N="$(find "$d" -maxdepth 1 -type d -name 'rollout_*' | wc -l)"
  echo "[$T] $N episodes  instruction: $TASK_TEXT"
  dirs+=("$d"); maps+=("${T}=${TASK_TEXT}")
done

[ "${#dirs[@]}" -gt 0 ] || { echo "ERROR: no task directory with rollout_000 under $SRC" >&2; exit 1; }

echo
"$PY" "$CONV" "${dirs[@]}" --version "$VERSION" "${EMB_ARGS[@]+"${EMB_ARGS[@]}"}" \
    --task-map "${maps[@]}" --output "$OUT/rollouts.json"

echo
"$PY" "${PREP_DIR}/register_dataset.py" "$NAME" --samples "$OUT/rollouts.json"

cat <<TIP

Next: set `dataset: ${NAME}` in a config from train/configs/, then
  CONFIG=<your yaml> GPU=0,1 bash train/scripts/train.sh
TIP
