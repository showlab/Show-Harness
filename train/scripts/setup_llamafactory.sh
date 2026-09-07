#!/usr/bin/env bash
# Install everything mvtoken training needs: upstream LLaMA-Factory plus a working venv.
#
# Depends on UPSTREAM hiyouga/LLaMA-Factory, pinned to a commit that has been trained on.
# Upstream source is never modified - the two things mvtoken needs on top live outside it,
# in train/llamafactory_extensions/.
#
# Training is self-contained: this builds its own venvs, independent of the .venv /
# .venv-vllm that scripts/setup.sh builds for running the robot.
#
#   bash train/scripts/setup_llamafactory.sh              # clone + the qwen3_5/internvl3_5 venv
#   bash train/scripts/setup_llamafactory.sh --gemma4     # also build the gemma4 venv
#   bash train/scripts/setup_llamafactory.sh --skip-venv  # clone only
#   LF_ROOT=/path/to/existing/LLaMA-Factory \
#       bash train/scripts/setup_llamafactory.sh          # symlink an existing checkout
#
# Everything lands in third_party/, so nothing needs configuring afterwards.
#
# The version pins below are not arbitrary; each one is a bug worked around. Read the
# comments before changing them.

set -euo pipefail

# Upstream pin (2026-07-27): mvtoken training has been verified end-to-end on this commit.
# Before bumping it, actually run a training on the new version - upstream refactors have
# broken Qwen3.5 support more than once.
LF_UPSTREAM_URL="${LF_UPSTREAM_URL:-https://github.com/hiyouga/LLaMA-Factory.git}"
LF_UPSTREAM_PIN="${LF_UPSTREAM_PIN:-9ce6b663e9d87cd3c0cb42a1d3ff5cdfe292426d}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

# The transformers each released adapter was actually trained under (the model cards in
# LlamaFactory's own output dirs record it). Not interchangeable, hence two venvs.
TRANSFORMERS_MAIN="${TRANSFORMERS_MAIN:-5.7.0}"      # qwen3_5 x4, internvl3_5
TRANSFORMERS_GEMMA4="${TRANSFORMERS_GEMMA4:-5.12.1}" # gemma4 e4b

SKIP_VENV=0
WITH_GEMMA4=0
for arg in "$@"; do
  case "$arg" in
    --skip-venv) SKIP_VENV=1 ;;
    --gemma4) WITH_GEMMA4=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg (--gemma4 / --skip-venv / --help)" >&2; exit 1 ;;
  esac
done

TRAINING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(dirname "$TRAINING_DIR")"
THIRD_PARTY="${REPO_ROOT}/third_party"
TARGET="${THIRD_PARTY}/LlamaFactory"      # every script derives LF_ROOT from this

# Pointed at an existing checkout? Symlink it into place instead of cloning, so the
# scripts find it at the default path and nothing needs configuring afterwards.
if [ -n "${LF_ROOT:-}" ] && [ -d "${LF_ROOT}/src/llamafactory" ]; then
  EXTERNAL="$(cd "$LF_ROOT" && pwd)"
  if [ "$EXTERNAL" != "$TARGET" ]; then
    mkdir -p "$THIRD_PARTY"
    [ -L "$TARGET" ] && rm -f "$TARGET"
    if [ -e "$TARGET" ]; then
      echo "[setup] ${TARGET} already exists and is not a symlink; leaving it alone" >&2
    else
      ln -s "$EXTERNAL" "$TARGET"
      echo "[setup] linked ${TARGET} -> ${EXTERNAL}"
    fi
  fi
  LF_ROOT="$TARGET"
  CLONED=0
else
  LF_ROOT="$TARGET"
  CLONED=1
fi

# ── 1. Upstream source ──────────────────────────────────────────────────────
if [ "$CLONED" = 1 ]; then
  if [ -d "${LF_ROOT}/.git" ]; then
    echo "[setup] already present: ${LF_ROOT}"
  else
    echo "[setup] cloning upstream LLaMA-Factory -> ${LF_ROOT}"
    mkdir -p "$(dirname "$LF_ROOT")"
    # Fetch only the pinned commit, not the full history.
    git init -q "$LF_ROOT"
    git -C "$LF_ROOT" remote add origin "$LF_UPSTREAM_URL" 2>/dev/null || true
    git -C "$LF_ROOT" fetch -q --depth 1 origin "$LF_UPSTREAM_PIN"
    git -C "$LF_ROOT" checkout -q FETCH_HEAD
  fi
  echo "[setup] upstream version: $(git -C "$LF_ROOT" rev-parse --short HEAD)"
fi

LF_VENV="${LF_VENV:-${LF_ROOT}/.venv}"
LF_VENV_GEMMA4="${LF_VENV_GEMMA4:-${LF_ROOT}/.venv-gemma4}"

if [ "$SKIP_VENV" = 1 ]; then
  echo "[setup] --skip-venv: skipping the venv build"
  exit 0
fi

# ── 3. venvs ────────────────────────────────────────────────────────────────
# Two, because the released models were trained under two different transformers:
#
#   .venv         transformers 5.7.0   qwen3_5, internvl3_5
#   .venv-gemma4  transformers 5.12.1  gemma4
#
# 5.12.1 is PAST LlamaFactory's own declared bound (<=5.8.0), which is why the training
# scripts export DISABLE_VERSION_CHECK=1.
#
# Collapsing gemma4 into the 5.7.0 venv was measured, not assumed: 30 steps, same data,
# same seed, single GPU. It RUNS -- identical tokenization (same 749 input_ids), identical
# processor config, no image-token mismatch -- but the forward pass is not the same. Two
# 5.12.1 runs give bit-identical loss at steps 1-2 (0.0e0), while 5.7.0 differs by 1.7e-2
# there; at step 1 LoRA B is still zero, so that is the BASE model computing different
# logits from the same tokens, not run-to-run noise. Later steps drift within bf16 noise.
# So: both versions train E4B, but only 5.12.1 reproduces the released adapter.
# train.sh routes to one venv or the other on FAMILY.
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

export UV_CACHE_DIR="${REPO_ROOT}/third_party/.uv-cache"   # system cache dirs hit permission issues
export UV_LINK_MODE=copy                                          # hardlinks fail across filesystems

build_venv() {  # $1 = venv path, $2 = transformers version, $3 = prompt
  local venv="$1" tf="$2" prompt="$3"

  echo "[setup] building venv: ${venv}  (transformers ${tf})"
  uv venv --python "${PYTHON_VERSION}" "${venv}" --prompt "${prompt}"

  # Stop uv from silently reinstalling this env: `uv run` implicitly syncs, replacing the
  # declared packages with uv.lock's versions. That once swapped torch 2.8.0+cu129 for
  # 2.13.0 and left every flash-attn / vllm C++ extension with undefined symbols - broken
  # silently for six days. Scoped to this venv's activate, so other projects are unaffected.
  cat >> "${venv}/bin/activate" <<'ACTIVATE_EOF'

export UV_NO_SYNC=1
ACTIVATE_EOF

  # shellcheck disable=SC1090
  source "${venv}/bin/activate"

  # torch goes first: flash-attn compiles against its headers
  uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu129

  uv pip install setuptools wheel packaging "hatchling>=1.18.0" editables

  # Upstream LlamaFactory, editable; core deps come from its pyproject
  uv pip install --no-build-isolation -e "${LF_ROOT}"

  uv pip install flash-attn --no-build-isolation
  uv pip install -r "${LF_ROOT}/requirements/liger-kernel.txt"
  uv pip install -r "${LF_ROOT}/requirements/deepspeed.txt"
  uv pip install -r "${LF_ROOT}/requirements/metrics.txt"
  uv pip install wandb

  # Pin transformers last: the installs above would otherwise resolve it themselves.
  uv pip install "transformers==${tf}"

  # Qwen3.5's gated-delta-net needs fla: LF uses fla-core, flash-linear-attention adds the
  # layers/models. --no-deps keeps them off the transformers pin above (flash-linear-attention
  # declares transformers>=4.45.0 and would resolve to a newer one).
  uv pip install "fla-core==0.5.1" "flash-linear-attention==0.5.1" --no-deps

  # On Hopper + Triton>=3.4 fla's triton GDN backward is broken, so the tilelang backend is
  # required; tilelang 0.1.11 with apache-tvm-ffi 0.1.12 crashes on import (duplicate tvm-ffi
  # registration), hence the 0.1.11 pin. Nothing else in this venv depends on it.
  uv pip install "tilelang==0.1.11" "apache-tvm-ffi==0.1.11"

  deactivate
}

build_venv "${LF_VENV}" "${TRANSFORMERS_MAIN}" "lf-mvtoken"
[ "$WITH_GEMMA4" = 1 ] && build_venv "${LF_VENV_GEMMA4}" "${TRANSFORMERS_GEMMA4}" "lf-gemma4"

# ── 4. gcc shim for tilelang JIT ────────────────────────────────────────────
# tilelang invokes `gcc`/`g++` directly (it ignores CC/CXX) and needs one whose cc1plus is
# actually installed. Rather than hardcode a version, probe the default gcc and fall back to
# the newest working g++-N.
# The probe must test `gcc -x c++` (the C driver on C++ source), NOT `g++`: a box can have a
# working g++ while its gcc lacks cc1plus, and a g++-based probe would pass while tilelang
# still fails.
SHIM_DIR="${REPO_ROOT}/third_party/.cc-shim"
rm -rf "${SHIM_DIR}"
_cc_probe() { echo 'int main(){return 0;}' | "$1" -x c++ - -o /dev/null >/dev/null 2>&1; }
if _cc_probe gcc; then
  echo "[setup] default gcc compiles C++; no shim needed."
else
  _cc=""
  for cand in gcc-13 gcc-12 gcc-11 gcc-10 gcc-9; do
    if command -v "${cand}" >/dev/null 2>&1 && _cc_probe "${cand}"; then _cc="${cand}"; break; fi
  done
  if [ -n "${_cc}" ]; then
    _cxx="${_cc/gcc/g++}"
    mkdir -p "${SHIM_DIR}"
    ln -sf "$(command -v "${_cc}")"                         "${SHIM_DIR}/gcc"
    ln -sf "$(command -v "${_cxx}" || command -v "${_cc}")"  "${SHIM_DIR}/g++"
    ln -sf "$(command -v "${_cc}")"                         "${SHIM_DIR}/cc"
    ln -sf "$(command -v "${_cxx}" || command -v "${_cc}")"  "${SHIM_DIR}/c++"
    echo "[setup] default gcc cannot compile C++; shimmed gcc/g++ -> ${_cc} (${SHIM_DIR})."
  else
    echo "WARNING: no gcc can compile C++ (default and gcc-9..13 all failed). tilelang JIT" >&2
    echo "         will fail; install a CUDA-compatible g++ (e.g. apt install g++-11) and rerun." >&2
  fi
fi

cat <<TIP

Done.

  LlamaFactory : ${LF_ROOT}  ($(git -C "$LF_ROOT" rev-parse --short HEAD 2>/dev/null || echo external))
  venv         : ${LF_VENV}  (transformers ${TRANSFORMERS_MAIN}) -- qwen3_5, internvl3_5
$([ "$WITH_GEMMA4" = 1 ] \
  && echo "  venv         : ${LF_VENV_GEMMA4}  (transformers ${TRANSFORMERS_GEMMA4}) -- gemma4" \
  || echo "  gemma4       : not built. Add --gemma4 if you need FAMILY=gemma4.")

Check it:
  source ${LF_VENV}/bin/activate && llamafactory-cli version

Next - register a dataset, then train:
  python train/data_preparation/register_dataset.py my_dataset --samples <rollouts.json>
  cp train/configs/qwen3_5_2b_lora.yaml train/configs/my_run.yaml   # fill the <FILL_ME>
  CONFIG=train/configs/my_run.yaml GPU=0,1 bash train/scripts/train.sh
TIP
