#!/usr/bin/env bash
# Batch-evaluate a served MVTOKEN LoRA in ManiSkill: N episodes over consecutive seeds,
# then print the success rate. Each episode is a full closed-loop rollout
# (scripts/run_maniskill_mvtoken.py), so this is the real task metric, not token accuracy.
#
#   bash scripts/maniskill/eval_batch.sh <config> <model> <n_episodes> [max_steps] [tag]
#
# Example:
#   bash scripts/maniskill/eval_batch.sh \
#     configs/robot_maniskill.yaml qwen3_5_2b_showharness_sim 20
#
# To evaluate on the training layout, pass the protocol through to the runner:
#   RUN_ARGS="--traj-id random --layout wide" bash scripts/maniskill/eval_batch.sh ...
set -uo pipefail

CFG="${1:?usage: eval_batch.sh <config> <model> <n> [max_steps] [tag]}"
MODEL="${2:?}"
N="${3:-20}"
MAX_STEPS="${4:-80}"
TAG="${5:-$MODEL}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-.venv/bin/python}"  # ManiSkill needs its own env: export PY=<maniskill env python>
OUT="${OUT:-/tmp/ms_eval}"
mkdir -p "$OUT"
LOG="$OUT/${TAG}.log"
: > "$LOG"

cd "$ROOT"
# Extra flags handed to the runner verbatim, e.g. RUN_ARGS="--traj-id random --layout wide".
read -r -a EXTRA <<<"${RUN_ARGS:-}"
echo "[eval] $TAG | cfg=$(basename "$CFG") | model=$MODEL | $N episodes | max_steps=$MAX_STEPS${RUN_ARGS:+ | $RUN_ARGS}"
ok=0
for i in $(seq 0 $((N - 1))); do
  # episode_index shifts the env seed, so each episode is a different layout.
  line=$(MUJOCO_GL=egl "$PY" -u scripts/run_maniskill_mvtoken.py \
      --robot-config "$CFG" --version v3 --model "$MODEL" \
      --episode-index "$i" --max-steps "$MAX_STEPS" --prompt-log-every 0 \
      ${EXTRA[@]+"${EXTRA[@]}"} \
      2>/dev/null | grep -E "^(Episode success|Steps|End reason):")
  s=$(grep -c "Episode success: True" <<<"$line")
  steps=$(sed -n 's/^Steps: //p' <<<"$line")
  reason=$(sed -n 's/^End reason: //p' <<<"$line")
  ok=$((ok + s))
  printf '  ep %02d: success=%s steps=%s reason=%s\n' \
    "$i" "$([ "$s" -eq 1 ] && echo True || echo False)" "${steps:-?}" "${reason:-?}" | tee -a "$LOG"
done
echo "[eval] $TAG RESULT: $ok/$N ($((100 * ok / N))%)" | tee -a "$LOG"
