#!/usr/bin/env bash
# Download the released Show-Harness training data and register it with LlamaFactory.
#
# The released splits are already converted: each carries a rollouts.json whose image
# paths are RELATIVE to its own directory, so the data stays portable. register_dataset.py
# pins them to absolute paths when it registers, which is what lets several splits be
# trained together (LlamaFactory has one global media_dir for all datasets).
#
#   HF_REPO=<id>    HuggingFace dataset repo (default showlab/Show-Harness-Data)
#   OUT=<dir>       destination (default <repo>/train/data/<HF_REPO basename>)
#   SPLITS="a b"    which splits to fetch (default "real sim")
#   PREFIX=showharness  dataset_info.json names are <PREFIX>_<split>
#   REGISTER=0      download only, leave dataset_info.json alone
#   REVISION=       git revision to pin (branch, tag or commit)
#   DRY_RUN=1       list what would be fetched, download and register nothing
#
#   bash train/scripts/download_dataset.sh
#   SPLITS=sim bash train/scripts/download_dataset.sh
#   HF_REPO=me/my-data PREFIX=mine bash train/scripts/download_dataset.sh
#
# Re-running is safe: hf skips files already present. Re-registering an existing name
# replaces that entry only. Do neither mid-training - LF's dataset fingerprint hashes
# path strings, so data replaced underneath a running job goes unnoticed.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/llamafactory_env.sh"

HF_REPO="${HF_REPO:-showlab/Show-Harness-Data}"
OUT="${OUT:-${REPO_ROOT}/train/data/${HF_REPO##*/}}"
SPLITS="${SPLITS:-real sim}"
PREFIX="${PREFIX:-showharness}"
REGISTER="${REGISTER:-1}"

command -v hf >/dev/null || {
  echo "ERROR: the 'hf' CLI is not on PATH." >&2
  echo "  fix: pip install -U huggingface_hub" >&2
  exit 1
}

# One --include per split, so asking for one split does not pull the rest. The root
# files (README and friends) are small and always worth having next to the data.
INCLUDE=(--include "*.md")
for s in $SPLITS; do INCLUDE+=(--include "${s}/*"); done
REV_ARGS=()
[ -n "${REVISION:-}" ] && REV_ARGS=(--revision "$REVISION")
DRY_ARGS=()
[ -n "${DRY_RUN:-}" ] && DRY_ARGS=(--dry-run)

echo "[download] ${HF_REPO} [${SPLITS}] -> ${OUT}"
[ -n "${DRY_RUN:-}" ] || mkdir -p "$OUT"
hf download "$HF_REPO" --repo-type dataset --local-dir "$OUT" \
    "${REV_ARGS[@]+"${REV_ARGS[@]}"}" "${DRY_ARGS[@]+"${DRY_ARGS[@]}"}" "${INCLUDE[@]}"

if [ -n "${DRY_RUN:-}" ]; then
  echo
  echo "[download] DRY_RUN, nothing downloaded or registered"
  exit 0
fi

if [ "$REGISTER" = "0" ]; then
  echo
  echo "[download] REGISTER=0, not touching dataset_info.json"
  exit 0
fi

echo
NAMES=()
for s in $SPLITS; do
  SAMPLES="${OUT}/${s}/rollouts.json"
  [ -f "$SAMPLES" ] || { echo "ERROR: no rollouts.json for split '${s}' at ${SAMPLES}" >&2; exit 1; }
  "$DATA_PREP_PYTHON" "${REPO_ROOT}/train/data_preparation/register_dataset.py" \
      "${PREFIX}_${s}" --samples "$SAMPLES"
  NAMES+=("${PREFIX}_${s}")
  echo
done

# Comma-separated is how LlamaFactory takes several datasets in one run.
IFS=,; JOINED="${NAMES[*]}"; unset IFS
cat <<TIP
Next: set \`dataset: ${JOINED}\` in a config from train/configs/, then
  CONFIG=<your yaml> GPU=0,1 bash train/scripts/train.sh
TIP
