# GUMI — GUI-based Manipulation Interface

The semantic action space is discrete and directly operable, so each unit maps to a key or
button: humans and agents demonstrate tasks by playing the robot, without teleoperation
hardware. Every step records a ready `(observation, action)` pair — exactly what the policy
sees at inference — so a demonstration is a training sequence with no post-processing.

| Launcher | Port | |
|---|---|---|
| `collect_rollouts_web.py` | 8600 | Single arm (franka / piper / sim) |
| `collect_rollouts_web_dual.py` | 8620 | Dual arm: one step = one synchronized `(a_left, a_right)` pair, idle side `STILL` |
| `gpt_web_operator.py` | 8630 | A VLM plays either one over HTTP, with a supervisor dashboard |

Both collectors take `--sim` for a synthetic rig — no hardware, no ROS; the operator
inherits whichever one it connects to.

## Collect

```bash
.venv/bin/python gumi/collect_rollouts_web.py rollouts/web --sim
.venv/bin/python gumi/collect_rollouts_web_dual.py rollouts/dual --sim
```

Open `http://<host>:8600/` and click the page once so it takes keys. `?` shows the key
reference. The servers are unauthenticated — bind them to a trusted network only.

Three ways to drive, all recording the same pairs:

| | |
|---|---|
| buttons / keys | one token per press, hold to repeat — `POST /api/move` |
| the sequence box | a whole line at once (`w*3 a g`) — `POST /api/step` |
| an agent | the same `/api/step` |

## Let an agent drive

**Browser agent** — paste the operator prompt into a computer-use agent pointed at the UI:
[`web_operator.txt`](../prompts/web_operator.txt) (8600) or
[`web_operator_dual.txt`](../prompts/web_operator_dual.txt) (8620). It loops
screenshot → action → verify the step counter, then stops and saves. **The prompts' key
tables are authoritative — change them together with the UI.**

**Hosted VLM** — no browser, reads the cameras over HTTP and posts tokens itself:

```bash
.venv/bin/python gumi/gpt_web_operator.py --target-url http://localhost:8620
```

Open `http://localhost:8630`. It starts **paused**: use **Step once**, then **Run**. Add
`--once --dry-run` for a safe API check that decides and traces but executes nothing.

The model comes from `--vlm-backend` (a `vlm_backends` profile in `--robot-config`, default
`chatgpt`); `--confidence-threshold`, `--image-max-side`, `--reasoning-effort` and
`--no-auto-save` are the switches worth knowing. Each session writes an append-only trace
(prompt, per-step JSONL, the exact frames sent) under `data/gpt_operator_traces/`.

It pauses on malformed or low-confidence output, repeated/oscillating actions, stale-image
decisions, dual-arm motion during a proximity warning, and completion claims while the save
gate is shut. **Pause cannot recall a command already sent** — keep the e-stop reachable.

## Intervene mid-rollout

The same key bindings drive the `dagger` plugin: during an autonomous rollout, a human key
preempts the in-flight VLM decision, the runner executes the human intent instead, and the
correction is recorded.
