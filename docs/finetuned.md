# Fine-tuned mode

The fine-tuned mode replaces the planner/controller stack with one small
fine-tuned VLM. There is no subgoal planner and no stage machinery: each step
sends the current camera frames plus a fixed-format prompt and receives exactly
one action token — `MV_FWD/BACK/LEFT/RIGHT/UP/DOWN`, `GRASP`, `RELEASE`, or
`DONE`. `DONE` ends the rollout; otherwise it runs to `max_steps`. The model is
a base VLM plus a LoRA adapter fine-tuned on GUMI demonstrations (real teleop
and/or real2sim data — see `docs/simulators.md`).

## Training contracts — must not drift

The served adapter saw exactly one input shape. Reproduce it, or the policy
degrades silently — no error, it just gets worse. The contracts, and where the
code enforces them:

- **Prompt version** (`--version`, required). Prompts live in
  `prompts/<version>/` and are the trained interface; the runtime template must
  match the adapter field-for-field.

  `v3` is **unified**: one single-arm prompt (`mvtoken_generator_lite.txt`) for
  both embodiments. The released corpus renders its Franka *and* its AgileX
  episodes through it — the rigs are reconciled in the data, by swapping
  `MV_FWD`/`MV_BACK` on the egocentric side, not by giving each its own prompt.
  **Every released checkpoint is v3**, so `--version v3` serves them directly,
  on either rig.

  `v4` is purpose-built for the AgileX rig: a per-embodiment single-arm pair
  (`--franka` exocentric / `--piper` egocentric, which describes the agentview
  as first-person) plus the dual-arm schemes below. No released checkpoint is
  trained on it — it is the interface to fine-tune your own adapter against.

  **The v3 reconciliation is not free at deploy time.** The AgileX episodes
  were rendered with `MV_FWD`/`MV_BACK` swapped, so a model trained on that mix
  emits swapped tokens on the AgileX rig — an emitted `MV_BACK` means *drive
  forward*. See **Execution-boundary token swap** below.

  For Franka the two are byte-identical (`prompts/v3/mvtoken_generator_lite.txt`
  == `prompts/v4/franka_mvtoken_lite.txt`), so a Franka or sim checkpoint
  reproduces under either version; only the `--piper` path actually differs.
- **Move history**: the prompt lists up to the 5 most recent `MV_*` moves,
  newest first, with `GRASP`/`RELEASE` excluded (`RECENT_MOVES_MAX` in
  `core/runners/mvtoken.py`, matching the training converter's window).
- **Execution-boundary token swap**: a checkpoint co-trained on the mixed
  Franka + AgileX corpus speaks the swapped depth convention on the AgileX rig.
  The rig's config declares it per checkpoint —
  `vlm_backends.<name>.execution_token_swap: [MV_FWD, MV_BACK]` in
  `configs/robot_piper_ft.yaml` — and `core.launch.install_execution_token_swap`
  applies it to that controller instance, so the pair is exchanged on the way
  into `controller.step` and **nowhere else**. `recent_moves` and the episode log
  keep the raw model token on purpose: the move history the adapter was trained
  on was written in the swapped convention too, so echoing the raw output back is
  what keeps the input distribution intact. A misspelled unit is refused at
  startup rather than silently disabling the swap (which would read as a bad
  policy, not a config error). Adapters you train from a single rig's rollouts
  need no declaration; `finetuned_local` ships without one.

- **No-think chat template**: requests disable model thinking and use
  temperature 0 (`VLMClient.complete_action_token`); the model answers with the bare
  token.
- **No gripper-state field**: the mvtoken lite prompts render only `{task}` and
  `{recent_moves}` — the gripper is read off the wrist view. The runner still
  computes a commanded `open`/`closed`
  (`core/runners/mvtoken.py::_gripper_state`) and hands it to `.format()`, where
  it is dropped for want of a placeholder; only the zero-shot
  `prompts/controller.txt` consumes that field.

To confirm nothing drifted, every `--prompt-log-every` steps the runner writes
`controller_prompts/<step>.txt`: the exact media parts sent (with per-frame
pixel fingerprints verified against the saved PNGs) and the full prompt text.

## Serving a checkpoint

Serve the base model with vLLM and register the LoRA adapter:

```bash
MODEL=<base-model> FAMILY=<family> LORA=<adapter-name>=<path> bash scripts/serve_vlm.sh
```

`FAMILY` picks the jinja that reproduces the adapter's training-time rendering (training
itself uses no jinja; LlamaFactory renders the conversation). `<adapter-name>` is what
clients ask for (`--model <adapter-name>`); the released checkpoints use
`<model>_showharness_<split>`, where the split is `ft` for the real-robot corpus and `sim`
for the simulation corpus:

| checkpoint | `MODEL` | `FAMILY` | adapter name | config backend |
| --- | --- | --- | --- | --- |
| Qwen3.5-2B | `Qwen/Qwen3.5-2B` | `qwen3_5` | `qwen3_5_2b_showharness_ft` | `qwen3_5_2b` |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | `qwen3_5` | `qwen3_5_4b_showharness_ft` | `qwen3_5_4b` |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` | `qwen3_5` | `qwen3_5_9b_showharness_ft` | `qwen3_5_9b` |
| InternVL3.5-2B | `OpenGVLab/InternVL3_5-2B-HF` | `internvl3_5` | `internvl3_5_2b_showharness_ft` | `internvl3_5_2b` |
| Gemma-4-E4B | `google/gemma-4-E4B-it` | `gemma4` | `gemma4_e4b_showharness_ft` | `gemma4_e4b` |

The last two columns are deliberately different names: the adapter name is what
the server registers and what `--model` asks for, while the config backend is the
short profile key you pass to `--vlm-backend`. That split lets every config offer
the same backend names while each points at its own weights.

The simulation-only policy is one adapter, not one per simulator: the released `sim` split
mixes RoboLab and ManiSkill, and both simulators share the 2 cm translation quantum and the
camera transform, so a single policy covers them. `configs/robot_maniskill.yaml` and
`configs/robot_robolab.yaml` both select it:

| checkpoint | `MODEL` | `FAMILY` | adapter name | config backend |
| --- | --- | --- | --- | --- |
| Qwen3.5-2B (RoboLab + ManiSkill) | `Qwen/Qwen3.5-2B` | `qwen3_5` | `qwen3_5_2b_showharness_sim` | `qwen3_5_2b` |

So a full command reads:

```bash
MODEL=OpenGVLab/InternVL3_5-2B-HF FAMILY=internvl3_5 \
  LORA=internvl3_5_2b_showharness_ft=<path> bash scripts/serve_vlm.sh
```

Use that script rather than calling `vllm serve` directly. It mounts the jinja that reproduces the
adapter's training-time rendering — the base model's own template does not, and the
mismatch is silent — and adds the flags InternVL needs to load its `lm_head` correctly. It also
checks the adapter paths and the port before vLLM spends minutes loading the base model.

The runners talk to any OpenAI-compatible endpoint.
`configs/robot_franka_ft.yaml` and
`configs/robot_piper_ft.yaml` default to `vlm_backend:
qwen3_5_2b` and carry one profile per released checkpoint, so serving
another one is a one-line switch. For your own adapter use the
`finetuned_local` profile and rename its `model`. Either way, set the
profile's `base_url` (or pass `--vlm-url http://<host>:<port>/v1`); `--model
<adapter-name>` overrides the adapter for one run.

## Running

Real robot, single arm (`--version` is required; every multi-frame flag must
match the served adapter's training):

```bash
python scripts/run_real_mvtoken.py --version v3 --model <adapter>
python scripts/run_real_mvtoken.py --version v4 --franka --model <adapter>
python scripts/run_real_mvtoken.py --version v3 --mock-robot --mock-cameras --max-steps 5 --no-show  # dry run
```

Every step's latency decomposition (camera read, decision incl. the VLM
call, arm motion, absolute start time) is written into the rollout's
`steps.jsonl`; summarize runs with
`python scripts/trajectory/step_timing.py <rollout-dir-or-parent>`.

Between rollouts the runner gates on Enter so the operator can reset the
scene; with `plugins: dagger: true` in the config the operator can take over
mid-rollout with the teleop keys, and `auto_release: true` opens the gripper
when a close measures empty.

Real robot, dual arm (Piper; the `arms:` block in
`configs/robot_piper_ft.yaml`). Pick exactly one scheme, matching the
adapter's training:

```bash
python scripts/run_real_dual_mvtoken.py --version v4 --once    # one call answers "<left> <right>"  (default choice)
python scripts/run_real_dual_mvtoken.py --version v4 --twice   # two VLM calls per step; right does not see left
python scripts/run_real_dual_mvtoken.py --version v4 --chain   # one image encoding, two answers; right sees left
```

`--once` is the scheme our dual-arm comparison settled on; the other two are kept
as alternatives. Whichever you pick has to be the one the served adapter was
trained under.

Simulators: `scripts/run_maniskill_mvtoken.py` and
`scripts/run_robolab_mvtoken.py` run the same loop in sim (`docs/simulators.md`); the
prompt-version and multi-frame flags are identical across all entry points.

## Measured step latency

All numbers below were measured on one machine — a single RTX 5090, vLLM on
localhost, two 256x256 views, the v3 lite prompt — so they are directly
comparable. Real-rig rows come from actual rollouts via
`scripts/trajectory/step_timing.py`; the serving breakdown comes from
replaying a recorded rollout's frames against the same server. Medians.

### Per-step budget on the real rig

| median, ms | 0.8B | 2B | 4B | 9B | Gemma-E4B |
| --- | --- | --- | --- | --- | --- |
| camera read | 3 | 18 | *3* | 3 | 10 |
| decision (client encode + call + parse) | 52 | 62 | *78* | 103 | 151 |
| &nbsp;&nbsp;of which server round trip | 27 | 39 | 58 | 79 | 131 |
| &nbsp;&nbsp;of which client PNG+base64 encode | 22 | 22 | 21 | 22 | 22 |
| bookkeeping (live view, logging) | 26 | 28 | *27* | 27 | 28 |
| **gap between two executions** | **81** | **108** | ***108*** | **133** | **189** |
| arm motion (one token) | 225 | 224 | *224* | 224 | 226 |
| **full step period** | **306** | **333** | ***332*** | **357** | **416** |
| **closed-loop rate** | 3.3 Hz | 3.0 Hz | *3.0 Hz* | 2.8 Hz | 2.4 Hz |
| timed steps | 41 | 24 | — | 31 | 51 |

*Italic = derived from the same model's serving measurement plus the
motion/bookkeeping constants, for the one checkpoint without a timed rollout.
The method was validated on Gemma before its rollout existed: predicted 153 /
183 / 407 ms for decision / gap / period against 151 / 189 / 416 measured.*

### Where the server time goes

Isolated with synthetic requests against each server (fresh pixels per
request, so nothing is served from the prefix cache):

| median, ms | 0.8B | 2B | 4B | 9B | Gemma-E4B |
| --- | --- | --- | --- | --- | --- |
| fixed serving overhead (HTTP, scheduling, tokenize, 1 decode) | 9.3 | 9.1 | 13.5 | 18.8 | 13.9 |
| vision encode + prefill, two views | 9.6 | 14.0 | 18.4 | 18.2 | **79.5** |
| per extra decoded token | 1.3 | 1.8 | 2.6 | 4.0 | 4.2 |

Three things worth knowing before optimizing anything:

- **The serving stack is not the bottleneck.** vLLM's fixed overhead is
  9-19 ms, 3-5% of a step. Going in-process would buy back at most that.
- **Gemma-E4B is the outlier.** Its vision tower costs 79.5 ms for two
  views — 4-5x the Qwen of comparable size, and 85% of its own inference
  time. The Qwen line grows gently with scale (9.6 -> 18.2 ms).
- **The client's PNG encode costs ~22 ms on every model**, more than the
  entire server round trip for 0.8B. It is pure format choice, not model
  cost.

### Against a chunked policy (pi0.5)

Measured on the same GPU with the same two-view input, in-process JAX
(`sample_actions`, five denoising steps), so it is the counterpart of the
server round trip above — pi0.5 pays no HTTP or image-encoding cost:

| pi0.5, 5 denoise steps | one inference | per action |
| --- | --- | --- |
| action_horizon 1 | 66.2 ms | 66.2 ms |
| action_horizon 4 | 50.8 ms | 12.7 ms |
| action_horizon 16 | 51.2 ms | 3.2 ms |

Horizon barely moves the cost (+0.4 ms from 4 to 16 actions): almost all of
it is the 3B backbone's prefix vision encoding, which happens once per
inference regardless of how many actions come out. So a chunked policy
cannot get cheaper by emitting fewer actions — its floor is ~51 ms per
inference.

That makes the comparison depend entirely on the granularity you hold fixed:

| | per inference | per action, all executed | per action, re-observing each time |
| --- | --- | --- | --- |
| this repo, 2B | 39 ms | 39 ms | **39 ms** |
| pi0.5, horizon 16 | 51 ms | 3.2 ms | **51 ms** |

Amortized over an executed chunk, pi0.5 is an order of magnitude cheaper per
action. At equal closed-loop granularity — a fresh observation before every
action, which is what this repo does — it is more expensive than the 2B and
0.8B checkpoints here. Its speed advantage comes from executing open-loop
between observations, not from a faster inference engine.

## Collecting training data (GUMI)

GUMI is the browser teleoperation suite in `gumi/`:

- `collect_rollouts_web.py` — single arm (Franka / Piper / sim): buttons, keyboard,
  and the `/api/step` sequence API that agents post to.
- `collect_rollouts_web_dual.py` — dual-arm compose-and-commit UI; every step
  records one synchronized left/right action pair.
- `gpt_web_operator.py` — a VLM operator plus supervisor dashboard that drives
  either collector autonomously through `/api/step`.

Each serves a small web app on its own port and runs against the real rig or a
synthetic world (`--sim`, no hardware). See `gumi/README.md`. Keyboard (pygame)
collectors record the same rollout format directly:
`scripts/trajectory/collect_rollouts.py` (Franka) and
`scripts/trajectory/collect_rollouts_piper.py` (Piper). Simulated
demonstrations come from `scripts/trajectory/real2sim/`.

All collectors share one convention with deployment: the frame is stored before
its token executes, and the recorded gripper state is the commanded one — so
training inputs and inference inputs line up frame-for-frame.

## Converting for training

The rollout-to-SFT converter is `train/data_preparation/rollouts_to_alpaca.py`:
it renders each recorded step through the same `prompts/<version>/` template the
runtime uses and emits one LLaMA-Factory sample per step. See [train/README.md](../train/README.md)
for the full path from rollouts to a served adapter.
