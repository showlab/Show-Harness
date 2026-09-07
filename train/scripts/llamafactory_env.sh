#!/usr/bin/env bash
# Resolve the paths the other scripts need. Sourced, never run directly.
#
# Everything is derived from the repo layout: LLaMA-Factory lives in third_party/,
# where setup_llamafactory.sh puts it. To use a checkout elsewhere, either point
# third_party/LlamaFactory at it (setup does this for you when given LF_ROOT) or
# override LF_ROOT for a single command.
#
# Debug: TRAIN_ENV_DEBUG=1 <script>

_TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"      # <repo>/train
export REPO_ROOT="${REPO_ROOT:-$(dirname "$_TRAIN_DIR")}"
export LF_ROOT="${LF_ROOT:-${REPO_ROOT}/third_party/LlamaFactory}"
export LF_VENV="${LF_VENV:-${LF_ROOT}/.venv}"

# The one thing worth checking: LF_ROOT must be a real LlamaFactory checkout.
# Otherwise the failure surfaces much later as a missing llamafactory-cli or an
# import error, where the root cause is no longer obvious.
if [ ! -d "${LF_ROOT}/src/llamafactory" ]; then
  echo "ERROR[env]: no LlamaFactory at ${LF_ROOT}" >&2
  echo "  fix: bash ${_TRAIN_DIR}/scripts/setup_llamafactory.sh" >&2
  echo "       (or LF_ROOT=/path/to/LLaMA-Factory <command>)" >&2
  return 1 2>/dev/null || exit 1
fi

# Python for data conversion: this repo's .venv, else system python3.
if [ -z "${DATA_PREP_PYTHON:-}" ]; then
  if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    DATA_PREP_PYTHON="${REPO_ROOT}/.venv/bin/python"
  else
    DATA_PREP_PYTHON="$(command -v python3)"
  fi
fi
export DATA_PREP_PYTHON

# tilelang JIT needs a gcc whose cc1plus is installed. setup builds a shim in third_party/;
# a LlamaFactory checkout set up by its own scripts carries one at its root.
for _shim in "${REPO_ROOT}/third_party/.cc-shim" "${LF_ROOT}/.cc-shim"; do
  [ -x "${_shim}/gcc" ] && { export CC_SHIM="${_shim}"; break; }
done

if [ -n "${TRAIN_ENV_DEBUG:-}" ]; then
  echo "[env] REPO_ROOT=$REPO_ROOT" >&2
  echo "[env] LF_ROOT=$LF_ROOT  LF_VENV=$LF_VENV" >&2
  echo "[env] DATA_PREP_PYTHON=$DATA_PREP_PYTHON  CC_SHIM=${CC_SHIM:-<none>}" >&2
fi
