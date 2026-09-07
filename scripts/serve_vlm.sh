#!/usr/bin/env bash
# Serve a VLM over vLLM's OpenAI-compatible API on :8000, where the configs'
# base_url points.
#
#   bash scripts/serve_vlm.sh                                   # default model, zero-shot
#   MODEL=<model> bash scripts/serve_vlm.sh                     # any other VLM
#   MODEL=<base> LORA=<name>=<path> bash scripts/serve_vlm.sh   # fine-tuned action model
#
#   MODEL=         hub id or local dir (default Qwen/Qwen3.5-2B). A bare hub id also
#                  resolves against HF_HOME/hub/ so a pre-downloaded copy is used
#   LORA=          adapters as comma-separated name=path; the name is what clients request
#                  (scripts/run_real_mvtoken.py --model <name>)
#   PORT=8000      listen port
#   GPU=           CUDA_VISIBLE_DEVICES
#   TP=1           tensor-parallel size; raise it only if the model needs several GPUs
#   GPU_UTIL=0.9   memory fraction
#   FAMILY=        qwen3_5 | internvl3_5 | gemma4 - required with LORA; picks the chat
#                  template the adapter was trained with
#   CHAT_TEMPLATE= override the auto-selected template
#   MAX_LEN=8192   max-model-len
#   TEMPERATURE=0  server-side default sampling temperature
#   ENFORCE_EAGER=1  skip CUDA graph capture (saves memory on large models)
#   DRY_RUN=1      print the command instead of running it
#
# With LORA set, three failure modes are handled that otherwise produce no error, only
# wrong results:
#
#   1. The chat template must reproduce what TRAINING rendered, not the base model's own.
#      Qwen3.5's official template emits an empty think block even with
#      enable_thinking=false, while the training template emits nothing - a 4-token
#      difference the model will happily answer through. The chat_template.jinja saved into
#      a training output dir is the base model's copy and is equally wrong. This script
#      mounts the aligned copy from models/chat_templates/.
#   2. InternVL needs tie_word_embeddings forced off, or vLLM drops the checkpoint's
#      lm_head and uses embed_tokens as the output layer: fluent text, broken semantics,
#      never emits an end token. It also needs --chat-template-content-format openai, or
#      vLLM joins image placeholders with newlines the template deliberately omits.
#   3. A port already serving another vLLM silently absorbs the requests - you get another
#      model's answers rather than a connection error. So probe the port first.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # <repo>/scripts
REPO_ROOT="$(dirname "$HERE")"

PORT="${PORT:-8000}"
TP="${TP:-1}"
GPU_UTIL="${GPU_UTIL:-0.9}"
MAX_LEN="${MAX_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
VENV="${VLLM_VENV:-${REPO_ROOT}/.venv-vllm}"
EXTRA_ARGS=()
LORA_ARGS=()

# Default = the fine-tuning backbone, so a bare `bash scripts/serve_vlm.sh` serves
# something usable. Pass MODEL= for anything else.
MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
SERVED_NAME="${MODEL}"          # what clients ask for; keep it stable below

# A pre-downloaded copy (scripts/model/download_vlm_model.sh) lives under HF_HOME/hub/<id>,
# which is not the layout vLLM resolves on its own -- point it there when present. The
# served name stays the id the user typed, so configs can name the model, not a path.
export HF_HOME="${HF_HOME:-${REPO_ROOT}/models/huggingface}"
if [ ! -e "${MODEL}" ] && [ -d "${HF_HOME}/hub/${MODEL}" ]; then
  MODEL="${HF_HOME}/hub/${MODEL}"
fi

# ── Adapters ────────────────────────────────────────────────────────────────
# vLLM loads the entire base model before it parses --lora-modules, so a wrong path costs
# minutes before it surfaces. Fail here instead, naming the offender.
if [ -n "${LORA:-}" ]; then
  IFS=',' read -ra _items <<< "$LORA"
  MISSING=()
  for item in "${_items[@]}"; do
    item="$(echo "$item" | xargs)"; [ -z "$item" ] && continue
    [ -f "${item#*=}/adapter_model.safetensors" ] || MISSING+=("${item%%=*}  ->  ${item#*=}")
    LORA_ARGS+=("$item")
  done
  if [ ${#MISSING[@]} -gt 0 ]; then
    echo "ERROR: no adapter_model.safetensors under these paths:" >&2
    printf '  %s\n' "${MISSING[@]}" >&2
    exit 1
  fi
fi

# ── Chat template: required for adapters, which were trained against a specific one ──
if [ ${#LORA_ARGS[@]} -gt 0 ] || [ -n "${FAMILY:-}" ] || [ -n "${CHAT_TEMPLATE:-}" ]; then
  if [ -z "${CHAT_TEMPLATE:-}" ]; then
    case "${FAMILY:-}" in
      qwen3_5)  CHAT_TEMPLATE="${REPO_ROOT}/models/chat_templates/qwen3_5_nothink.jinja" ;;
      gemma4)   CHAT_TEMPLATE="${REPO_ROOT}/models/chat_templates/gemma4n.jinja" ;;
      internvl3_5) CHAT_TEMPLATE="${REPO_ROOT}/models/chat_templates/internvl3_5.jinja" ;;
      "") echo "ERROR: serving an adapter needs FAMILY=qwen3_5|internvl3_5|gemma4" >&2
          echo "  it selects the chat template the adapter was trained with" >&2; exit 1 ;;
      *)  echo "ERROR: FAMILY must be qwen3_5 | internvl3_5 | gemma4, got: $FAMILY" >&2; exit 1 ;;
    esac
  fi
  [ -f "$CHAT_TEMPLATE" ] || { echo "ERROR: chat template not found: $CHAT_TEMPLATE" >&2; exit 1; }
  EXTRA_ARGS+=(--chat-template "${CHAT_TEMPLATE}")
  if [ "${FAMILY:-}" = internvl3_5 ]; then
    # InternVL3.5-HF's config.json has no tie_word_embeddings field, and the defaults
    # disagree: transformers says False, vLLM says True. vLLM then discards the
    # checkpoint's lm_head and uses embed_tokens instead. Unrelated to LoRA or images.
    EXTRA_ARGS+=(--hf-overrides '{"tie_word_embeddings": false, "text_config": {"tie_word_embeddings": false}}')
    # If content-format is inferred as string, vLLM joins image placeholders with newlines.
    EXTRA_ARGS+=(--chat-template-content-format openai)
  fi
fi
[ ${#LORA_ARGS[@]} -gt 0 ] && EXTRA_ARGS+=(--enable-lora --max-lora-rank "${MAX_LORA_RANK}" --lora-modules "${LORA_ARGS[@]}")

[ -f "${VENV}/bin/activate" ] || {
  echo "ERROR: vLLM venv not found: ${VENV}" >&2
  echo "  build it: bash ${HERE}/setup.sh serve   or set VLLM_VENV=/path/to/venv" >&2; exit 1; }

# A port already serving another vLLM would silently absorb requests meant for this one.
if command -v curl >/dev/null 2>&1 && curl -s -m 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "ERROR: port ${PORT} is already serving." >&2
  echo "  It would silently absorb requests meant for this server." >&2
  curl -s -m 2 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | head -c 300 >&2; echo >&2
  exit 1
fi

# CUDA JIT host compiler: some boxes ship a gcc whose cc1plus is missing, which breaks
# tilelang / flashinfer. Prefer the shim built by the training setup, else the newest
# g++-N that actually compiles.
_probe() { echo 'int main(){return 0;}' | "$1" -x c++ - -o /dev/null >/dev/null 2>&1; }
if [ -x "${REPO_ROOT}/third_party/.cc-shim/g++" ] && _probe "${REPO_ROOT}/third_party/.cc-shim/g++"; then
  _cxx="${REPO_ROOT}/third_party/.cc-shim/g++"; _cc="${REPO_ROOT}/third_party/.cc-shim/gcc"
elif ! _probe gcc; then
  for _cand in g++-13 g++-12 g++-11 g++-10; do
    command -v "$_cand" >/dev/null 2>&1 && _probe "$_cand" \
      && { _cxx="$(command -v "$_cand")"; _cc="$(command -v "${_cand/g++/gcc}")"; break; }
  done
fi
if [ -n "${_cxx:-}" ]; then
  export CC="${CC:-$_cc}" CXX="${CXX:-$_cxx}" CUDAHOSTCXX="${CUDAHOSTCXX:-$_cxx}"
  export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--ccbin $_cxx}"
fi

[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="${GPU}"
# shellcheck disable=SC1090
source "${VENV}/bin/activate"

CMD=(
  vllm serve "${MODEL}"
  --dtype bfloat16
  --gpu-memory-utilization "${GPU_UTIL}"
  --max-model-len "${MAX_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --tensor-parallel-size "${TP}"
  --override-generation-config "{\"temperature\": ${TEMPERATURE}, \"top_p\": 1.0, \"top_k\": -1}"
  --trust-remote-code
  --host "${HOST:-0.0.0.0}"
  --port "${PORT}"
  --served-model-name "${SERVED_NAME}"
)
[ ${#EXTRA_ARGS[@]} -gt 0 ] && CMD+=("${EXTRA_ARGS[@]}")
[ "${ENFORCE_EAGER:-0}" = "1" ] && CMD+=(--enforce-eager)

SEP="────────────────────────────────────────────────────────────────"
echo "$SEP"
echo "  vLLM      http://0.0.0.0:${PORT}   GPU ${CUDA_VISIBLE_DEVICES:-<all>}  tp ${TP}  util ${GPU_UTIL}"
echo "  model     ${SERVED_NAME}"
[ "${MODEL}" != "${SERVED_NAME}" ] && echo "  path      ${MODEL}"
[ -n "${CHAT_TEMPLATE:-}" ] && echo "  template  ${CHAT_TEMPLATE##*/}"
[ ${#LORA_ARGS[@]} -gt 0 ] && for m in "${LORA_ARGS[@]}"; do printf "  lora      %-24s %s\n" "${m%%=*}" "${m#*=}"; done
echo "$SEP"

if [ -n "${DRY_RUN:-}" ]; then printf '[dry-run]'; printf ' %q' "${CMD[@]}"; echo; exit 0; fi
exec "${CMD[@]}"
