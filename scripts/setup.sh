#!/usr/bin/env bash
set -euo pipefail

# Build the environments needed to RUN Show-Harness.
#
#   bash scripts/setup.sh                # show what exists and what is missing
#   bash scripts/setup.sh base           # .venv        the harness
#   bash scripts/setup.sh base --real    #              + Franka/Piper hardware layer
#   bash scripts/setup.sh serve          # .venv-vllm   serve a VLM locally
#
# Two venvs because their pins conflict: serving holds transformers where the harness does
# not want it. Start with `base`; driving a robot against an already-served VLM needs
# nothing else, and a hosted endpoint needs no `serve` either.
#
# TRAINING is separate and self-contained under train/ -- it builds its own venvs against
# upstream LLaMA-Factory, with per-family transformers pins:
#
#   bash train/scripts/setup_llamafactory.sh              # qwen3_5 / internvl3_5
#   bash train/scripts/setup_llamafactory.sh --gemma4     # add this to also train gemma4

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${SHOWHARNESS_VENV:-${REPO_ROOT}/.venv}"
VLLM_VENV="${SHOWHARNESS_VLLM_VENV:-${REPO_ROOT}/.venv-vllm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

usage() { sed -n '4,20p' "$0" | sed 's/^# \?//'; }

status() {
  local mark
  echo "Environments (relative to ${REPO_ROOT}):"
  printf '\n  %-9s %-38s %s\n' "" "venv" "state"
  for row in \
    "base:${VENV}:the harness" \
    "serve:${VLLM_VENV}:serve a VLM locally"
  do
    IFS=: read -r what path desc <<< "$row"
    [ -f "${path}/bin/activate" ] && mark="built" || mark="--"
    printf '  %-9s %-38s %-7s %s\n' "$what" "${path#"${REPO_ROOT}/"}" "$mark" "$desc"
  done
  usage | sed -n '2,6p'
  echo
  echo "  Training has its own setup: bash train/scripts/setup_llamafactory.sh"
}

require_uv() {
  command -v uv >/dev/null 2>&1 && return 0
  echo "ERROR: uv is required. https://docs.astral.sh/uv/getting-started/installation/" >&2
  return 1
}

setup_base() {
  local req="${REPO_ROOT}/requirements/requirements.txt"
  [ "${1:-}" = --real ] && req="${REPO_ROOT}/requirements/requirements-real.txt"

  # uv when it is around (much faster), plain venv+pip otherwise. Nothing here needs a
  # custom index or install ordering, so either works.
  if command -v uv >/dev/null 2>&1; then
    uv venv --python "${PYTHON_VERSION}" "${VENV}"
    uv pip install --python "${VENV}/bin/python" -r "${req}"
  else
    echo "[setup] uv not found, using python -m venv (slower)."
    python3 -m venv "${VENV}"
    "${VENV}/bin/python" -m pip install --upgrade pip
    "${VENV}/bin/python" -m pip install -r "${req}"
  fi

  cat <<EOF

Base environment ready.

  venv         : ${VENV}
  requirements : ${req#"${REPO_ROOT}/"}

  source ${VENV}/bin/activate
  python scripts/check_setup.py --robot-config configs/robot_franka.yaml

To serve a model locally:  bash scripts/setup.sh serve
EOF
}

setup_serve() {
  require_uv
  uv venv --python "${PYTHON_VERSION}" "${VLLM_VENV}"
  uv pip install --python "${VLLM_VENV}/bin/python" \
      -r "${REPO_ROOT}/requirements/requirements-vllm.txt"
  mkdir -p "${REPO_ROOT}/models/huggingface"

  cat <<EOF

Serving environment ready. It only serves; the harness runs from ${VENV##*/}.

  venv : ${VLLM_VENV}

Fetch weights, then serve:
  ADAPTER=qwen3_5_2b WITH_BASE=1 bash scripts/model/download_vlm_model.sh
  bash scripts/serve_vlm.sh
EOF
}

case "${1:-}" in
  ""|-h|--help)  [ "${1:-}" = "" ] && status || usage ;;
  base)   shift; setup_base "$@" ;;
  serve)  shift; [ $# -gt 0 ] && { echo "serve takes no options" >&2; exit 1; }; setup_serve ;;
  train)  echo "Training setup lives with the training code:" >&2
          echo "  bash train/scripts/setup_llamafactory.sh              # qwen3_5 / internvl3_5" >&2
          echo "  bash train/scripts/setup_llamafactory.sh --gemma4     # add this to also train gemma4" >&2
          exit 1 ;;
  *) echo "unknown target: $1 (base / serve)" >&2; exit 1 ;;
esac
