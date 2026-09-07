#!/usr/bin/env bash
set -euo pipefail

# Fetch what scripts/serve_vlm.sh serves, into the layout it reads.
#
#   MODEL=<hub-id>            base model  -> models/huggingface/hub/<hub-id>
#   ADAPTER=<name>|all        released LoRA -> models/Show-Harness-VLMs/<name>
#   ADAPTER=<name> WITH_BASE=1  also fetch the base that adapter declares
#
# Both are optional on their own; give at least one. Downloading is optional in general
# (vLLM fetches a hub id itself), it just keeps the wait out of server start-up.
#
#   MODEL=Qwen/Qwen3.5-2B bash scripts/model/download_vlm_model.sh
#   ADAPTER=qwen3_5_2b WITH_BASE=1 bash scripts/model/download_vlm_model.sh
#   ADAPTER=all bash scripts/model/download_vlm_model.sh

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
HF_HOME="${HF_HOME:-${REPO_ROOT}/models/huggingface}"
ADAPTER_REPO="${ADAPTER_REPO:-showlab/Show-Harness-VLMs}"
ADAPTER_DIR="${ADAPTER_DIR:-${REPO_ROOT}/models/Show-Harness-VLMs}"

# The released adapters, i.e. the folders in ADAPTER_REPO. Each is a LoRA for the base
# named in its own adapter_config.json; FAMILY is what serve_vlm.sh needs to pick the
# chat template the adapter was trained under.
RELEASED=(qwen3_5_0_8b qwen3_5_2b qwen3_5_4b qwen3_5_9b gemma4_e4b qwen3_5_2b_sim)
adapter_name_of() {  # released folder -> the name clients ask for, <model>_showharness_<split>
  case "$1" in
    *_sim) echo "${1%_sim}_showharness_sim" ;;
    *)     echo "${1}_showharness_ft" ;;
  esac
}

family_of() {
  case "$1" in
    qwen3_5_*)     echo qwen3_5 ;;
    gemma4_*)      echo gemma4 ;;
    internvl3_5_*) echo internvl3_5 ;;
    *)             echo "" ;;
  esac
}

MODEL_ID="${MODEL:-${VLLM_MODEL_ID:-}}"
ADAPTER="${ADAPTER:-}"
if [ -z "${MODEL_ID}" ] && [ -z "${ADAPTER}" ]; then
  echo "ERROR: give MODEL=<hub-id> and/or ADAPTER=<name>|all" >&2
  echo "  released adapters: ${RELEASED[*]}" >&2
  echo "  e.g. ADAPTER=qwen3_5_2b WITH_BASE=1 bash $0" >&2
  exit 1
fi

# `hf` from the environment if there is one, else the vLLM venv's. Downloading an adapter
# should not require a built vLLM venv.
VLLM_VENV_PATH="${SHOWHARNESS_VLLM_VENV:-${REPO_ROOT}/.venv-vllm}"
if command -v hf >/dev/null 2>&1; then
  HF_BIN="$(command -v hf)"
elif [ -x "${VLLM_VENV_PATH}/bin/hf" ]; then
  HF_BIN="${VLLM_VENV_PATH}/bin/hf"
else
  echo "ERROR: the 'hf' CLI was not found." >&2
  echo "  pip install huggingface_hub[cli]   or   bash ${REPO_ROOT}/scripts/setup.sh serve" >&2
  exit 1
fi

export HF_HOME
mkdir -p "${HF_HOME}"

if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ] \
   && [ -f "${HOME}/.cache/huggingface/token" ]; then
  HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
  export HF_TOKEN
fi

fetch_base() {  # $1 = hub id
  local id="$1" dest="${HF_HOME}/hub/$1"
  # Skip only when the WEIGHTS are there. An interrupted download (a full disk, a
  # dropped connection) leaves config.json and the tokenizer behind with every shard
  # still a .incomplete blob, so config.json alone would report a truncated model as
  # ready and the serve step would fail much later. `hf download` resumes from the
  # blobs, so re-running is cheap when this check is unsure.
  if [ -f "${dest}/config.json" ] \
     && ! find "${dest}/.cache" -name '*.incomplete' -print -quit 2>/dev/null | grep -q . \
     && find "${dest}" -maxdepth 1 \( -name '*.safetensors' -o -name '*.bin' -o -name '*.gguf' \) \
          -print -quit 2>/dev/null | grep -q .; then
    echo "base    ${id} already at ${dest}"
    return
  fi
  if [ -z "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "note: HF_TOKEN is not set; a gated base (e.g. google/*) will fail" >&2
  fi
  "${HF_BIN}" download "${id}" --local-dir "${dest}"
  echo "base    ${id} -> ${dest}"
}

fetch_adapter() {  # $1 = released folder name
  local name="$1" dest="${ADAPTER_DIR}/$1"
  "${HF_BIN}" download "${ADAPTER_REPO}" --include "${name}/*" --local-dir "${ADAPTER_DIR}"
  if [ ! -f "${dest}/adapter_config.json" ]; then
    echo "ERROR: ${ADAPTER_REPO} has no folder '${name}' (nothing downloaded)." >&2
    echo "  released adapters: ${RELEASED[*]}" >&2
    exit 1
  fi
  echo "adapter ${name} -> ${dest}"
}

base_of() {  # $1 = adapter dir -> the base it was trained on
  python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['base_model_name_or_path'])" \
    "$1/adapter_config.json" 2>/dev/null || true
}

[ -n "${MODEL_ID}" ] && fetch_base "${MODEL_ID}"

if [ -n "${ADAPTER}" ]; then
  if [ "${ADAPTER}" = all ]; then
    TARGETS=("${RELEASED[@]}")
  else
    TARGETS=("${ADAPTER}")
  fi
  for name in "${TARGETS[@]}"; do
    fetch_adapter "${name}"
    base="$(base_of "${ADAPTER_DIR}/${name}")"
    [ -n "${base}" ] || continue
    if [ "${WITH_BASE:-0}" = 1 ]; then
      fetch_base "${base}"
    elif [ ! -f "${HF_HOME}/hub/${base}/config.json" ]; then
      echo "        needs base ${base} (not local yet) -- add WITH_BASE=1, or:"
      echo "        MODEL=${base} bash $0"
    fi
  done

  # Ready-to-paste serve line for the last adapter handled.
  name="${TARGETS[-1]}"
  base="$(base_of "${ADAPTER_DIR}/${name}")"
  family="$(family_of "${name}")"
  if [ -n "${base}" ] && [ -n "${family}" ]; then
    echo
    echo "serve it:"
    echo "  MODEL=${base} \\"
    echo "  LORA=$(adapter_name_of "${name}")=${ADAPTER_DIR#"${REPO_ROOT}/"}/${name} \\"
    echo "  FAMILY=${family} bash scripts/serve_vlm.sh"
  fi
fi
