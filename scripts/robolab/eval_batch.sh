#!/usr/bin/env bash
# Batch-evaluate a served MVTOKEN LoRA on RoboLab tasks: N episodes per task, then print
# the per-task and overall success rate. Each episode is a full closed-loop rollout
# (scripts/run_robolab_mvtoken.py), so this is the real task metric, not token accuracy.
#
#   bash scripts/robolab/eval_batch.sh <model> <n_episodes> [task ...]
#
# Example:
#   bash scripts/robolab/eval_batch.sh <adapter-name> 10 \
#       BananaInBowlTask BagelOnPlateTask
#
# Unlike scripts/maniskill/eval_batch.sh this runs ONE process per TASK rather than one per
# episode: Isaac Sim's cold start plus env construction costs tens of seconds, so
# scripts/run_robolab_mvtoken.py --episodes N reuses the app and the env across episodes (exactly
# what RoboLab's own run_evaluation does).
#
# Requires: the served action model (scripts/serve_vlm.sh), and Isaac Sim's EULA already accepted
# (OMNI_KIT_ACCEPT_EULA=YES, see docs/simulators.md).
set -uo pipefail

MODEL="${1:?usage: eval_batch.sh <model> <n_episodes> [task ...]}"
N="${2:-10}"
shift 2 || true
TASKS=("$@")
if [ ${#TASKS[@]} -eq 0 ]; then
  TASKS=(BananaInBowlTask)
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-.venv/bin/python}"  # RoboLab needs its own env: export PY=<robolab venv python>
CFG="${CFG:-configs/robot_robolab.yaml}"
VERSION="${VERSION:-v3}"
MAX_STEPS="${MAX_STEPS:-80}"
OUT="${OUT:-/tmp/robolab_eval}"
mkdir -p "$OUT"
LOG="$OUT/${MODEL}.log"
: > "$LOG"

cd "$ROOT"
echo "[eval] model=$MODEL | cfg=$(basename "$CFG") | $N episodes x ${#TASKS[@]} task(s) | max_steps=$MAX_STEPS"
total_ok=0
total_n=0
for task in "${TASKS[@]}"; do
  line=$("$PY" -u scripts/run_robolab_mvtoken.py \
      --robot-config "$CFG" --version "$VERSION" --model "$MODEL" \
      --task "$task" --episodes "$N" --max-steps "$MAX_STEPS" --prompt-log-every 0 \
      2>/dev/null | grep -E "^Success rate:")
  # "Success rate: k/N"
  ok=$(sed -n 's|^Success rate: \([0-9]*\)/.*|\1|p' <<<"$line")
  ok="${ok:-0}"
  total_ok=$((total_ok + ok))
  total_n=$((total_n + N))
  printf '  %-40s %s/%s\n' "$task" "$ok" "$N" | tee -a "$LOG"
done
if [ "$total_n" -gt 0 ]; then
  echo "[eval] $MODEL RESULT: $total_ok/$total_n ($((100 * total_ok / total_n))%)" | tee -a "$LOG"
fi
