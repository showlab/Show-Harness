<div align="center">

<img src="assets/show-harness-logo.svg" alt="Show-Harness" width="380">

### Just a VLM Agent Can Play Robots

<p align="center">
<a href="https://chenanno.github.io/">Yanzhe Chen</a><sup>*</sup> ·
<a href="https://www.baizechen.site/">Zechen Bai</a><sup>*</sup> ·
<a href="https://caozhijun.top/">Zhijun Cao</a><sup>*</sup> ·
<a href="https://wenzhengzeng.github.io/">Wenzheng Zeng</a><sup>*</sup> ·
<a href="https://qhlin.me/">Kevin Qinghong Lin</a><br>
<a href="https://linyq17.github.io/">Yiqi Lin</a> ·
<a href="https://ethanliang99.github.io/">Guoqiang Liang</a> ·
<a href="https://kevinskwk.github.io/">Kevin Yuchen Ma</a> ·
<a href="https://github.com/ceilingFan456/">Qiming Huang</a> ·
<a href="https://sites.google.com/view/showlab">Mike Zheng Shou</a><sup>&dagger;</sup>
</p>

<p align="center"><sup>*</sup> equal contribution &nbsp;·&nbsp; <sup>&dagger;</sup> corresponding author</p>

**Show Lab @ National University of Singapore**

<!-- TODO before release: fill in the paper and X links; confirm the project-page URL. -->
📄 [Paper](#) &nbsp;|&nbsp; 🤗 [Models](https://huggingface.co/showlab/Show-Harness-VLMs) &nbsp;|&nbsp; 📊 [Dataset](https://huggingface.co/datasets/showlab/Show-Harness-Data) &nbsp;|&nbsp; 🌐 [Project Page](https://showlab.github.io/Show-Harness/) &nbsp;|&nbsp; 💬 [X (Twitter)](#)

</div>

---

## Layout

| Path | What it is |
| --- | --- |
| `core/` | The interaction loop: runners for both modes, config layering, logging, the shared action vocabulary, and the provider-agnostic VLM client + roles (`core/vlm/`) |
| `plugins/` | Harness plugins — each mounts on one stage of the loop and is toggled from the `plugins:` config block ([plugins/README.md](plugins/README.md)) |
| `interpreters/` | Embodiment interpreters: Franka (impedance), AgileX Piper (joint streaming), ManiSkill / Isaac-Lab sims |
| `gumi/` | GUMI: browser teleoperation + agent operators; every step is recorded as a ready (observation, action) training pair |
| `configs/` | Layered configs: shipped defaults + your site identity + optional overlays ([configs/README.md](configs/README.md)) |
| `prompts/` | Controller prompts (zero-shot) and the versioned prompt contracts of fine-tuned checkpoints |
| `scripts/` | Rig bring-up, calibration capture, serving, data collection |
| `train/` | The fine-tuning pipeline: data conversion, dataset registration, LoRA configs ([train/README.md](train/README.md)) |
| `models/` | Chat templates, downloaded adapters, the HuggingFace cache ([models/README.md](models/README.md)) |
| `docs/` | Per-rig runbooks and the fine-tuned mode guide |

## Environments

Separate venvs, because their pins conflict. Start with `base`; add the rest only when
you need them. `bash scripts/setup.sh` with no arguments prints which already exist.

| | build it with | what it is for |
| --- | --- | --- |
| `.venv` | `bash scripts/setup.sh base` | the harness: collect, run a robot, drive a served VLM |
| `.venv-vllm` | `bash scripts/setup.sh serve` | serving a VLM locally (`scripts/serve_vlm.sh`) |

`bash scripts/setup.sh base --real` adds the Franka/Piper hardware layer (RealSense, ROS
shims, teleop window). Zero-shot and sim work do not need it.

Serving is its own process, so the harness talks to any OpenAI-compatible endpoint — a
hosted model, or a colleague's server — without `.venv-vllm` existing at all.

Training is self-contained under [train/](train/) and builds its own venvs against
upstream LLaMA-Factory (`bash train/scripts/setup_llamafactory.sh`); nothing in the
sections above depends on it.

## Collect demonstrations with GUMI

GUMI maps every action unit to a key or button, so a human — or a GUI-driving
agent — demonstrates a task by playing the robot in the browser, and every step
is recorded as a training-ready (observation, action) pair. It drives the real
rigs below; a synthetic tabletop world (`--sim`) lets you try the interface
before any hardware is set up:

```bash
bash scripts/setup.sh base
.venv/bin/python gumi/collect_rollouts_web.py data/rollouts_demo --sim
# open http://localhost:8600 and drive the gripper with WASD / arrow keys
```

The same servers run against the real Franka/Piper rigs (drop `--sim`), and the
same key bindings power live human takeover during autonomous rollouts. See
[gumi/README.md](gumi/README.md) for the keyboard UI, the dual-arm UI, and the
agent operators.

## Run a real robot

1. Copy `configs/site/franka.yaml.example` to `configs/site/franka.yaml` and
   fill in your robot address and camera serials (Piper:
   `site/piper_arms.yaml.example`).
2. Copy `configs/secrets.env.example` to `configs/secrets.env` and add an API
   key for the backend you use (`GEMINI_API_KEY` by default), or serve a local
   VLM with `scripts/serve_vlm.sh`.
3. Calibrate the safety floor and begin pose for your table — the shipped
   values are examples, and every autonomous run refuses to descend below the
   calibrated floor.
4. Preflight — checks the environment, your site config and calibration, the
   VLM backend (live), and that the robot and cameras answer:
   `python scripts/check_setup.py --robot-config configs/robot_franka.yaml`
5. `python scripts/run_real.py --robot-config configs/robot_franka.yaml`

The full walkthroughs are in [docs/franka.md](docs/franka.md) and
[docs/piper.md](docs/piper.md); simulators in
[docs/simulators.md](docs/simulators.md).

## Two modes, one interface

**Zero-shot** — a frontier VLM operates the full plugin harness with no
robot-specific training (`scripts/run_real.py`, `scripts/run_real_dual.py`).

**Fine-tuned** — a small VLM fine-tuned on GUMI demonstrations emits one
action token per step, planner-free (`scripts/run_real_mvtoken.py`). The real-robot
configs default to `vlm_backend: qwen3_5_2b` (the `qwen3_5_2b_showharness_ft`
adapter); serve your own checkpoint
instead and select it with `vlm_backend: finetuned_local`. See
[docs/finetuned.md](docs/finetuned.md), including the training contracts
that must not drift.

Switching embodiments changes only the interpreter and its
`configs/primitives_<embodiment>.yaml`; the model-facing vocabulary and prompts
stay the same.

## Released checkpoints and data

Five LoRA adapters trained on the real corpus, one per backbone, at
[showlab/Show-Harness-VLMs](https://huggingface.co/showlab/Show-Harness-VLMs):
`qwen3_5_0_8b`, `qwen3_5_2b`, `qwen3_5_4b`, `qwen3_5_9b`, `gemma4_e4b`; plus
`qwen3_5_2b_sim`, one simulation policy covering both simulators. The
demonstrations they were trained on are at
[showlab/Show-Harness-Data](https://huggingface.co/datasets/showlab/Show-Harness-Data)
(real Franka/Piper rollouts plus RoboLab and ManiSkill).

Fetch an adapter with the base model it needs, serve it, drive the robot:

```bash
ADAPTER=qwen3_5_2b WITH_BASE=1 bash scripts/model/download_vlm_model.sh
MODEL=Qwen/Qwen3.5-2B \
  LORA=qwen3_5_2b_showharness_ft=models/Show-Harness-VLMs/qwen3_5_2b \
  FAMILY=qwen3_5 bash scripts/serve_vlm.sh          # activates .venv-vllm itself

python scripts/run_real_mvtoken.py --robot-config configs/robot_franka_ft.yaml
```

`FAMILY` picks a jinja template from `models/chat_templates/`. Training never reads one —
LlamaFactory renders the conversation itself — so these exist only to make vLLM reproduce
that rendering at serve time. A base model's own template does not, and the mismatch fails
silently — see [models/README.md](models/README.md).

To fine-tune your own, [train/](train/) takes rollouts (yours or the released set) to a
LoRA on any of the three supported families.

## Plugins

Each plugin mounts on one stage of the loop, is toggled by one boolean, and
leaves the loop byte-identical when disabled.

| Stage | Plugin (paper) | Code |
| --- | --- | --- |
| Perception | Multi-View Guidance | view-role prompt scaffolding + `core/prompting/wrist_marker.py` (`plugins/view_select` on the dual rig) |
| Perception | Proprioception | `plugins/proprioception` |
| Reasoning | Subtask Planning | `plugins/subgoal` |
| Reasoning | Situated Planning | `plugins/deepplan` |
| Reasoning | Action Chunking | `plugins/action_chunk` |
| Reasoning | Adaptive Step | `plugins/variable_step` |
| Reasoning | Visual Prompt | `plugins/affordance` |
| Action | Action History | `plugins/mem_text` |
| Action | Failure Recovery | `plugins/recovery` (+ `plugins/auto_release` in the fine-tuned mode) |

`plugins/README.md` documents the contract for writing your own.

## Citation

```bibtex
@article{chen2026showharness,
  title   = {Show-Harness: Just a VLM Agent Can Play Robots},
  author  = {Chen, Yanzhe and Bai, Zechen and Cao, Zhijun and Zeng, Wenzheng and
             Lin, Kevin Qinghong and Lin, Yiqi and Liang, Guoqiang and
             Ma, Kevin Yuchen and Huang, Qiming and Shou, Mike Zheng},
  journal = {arXiv preprint},
  year    = {2026},
}
```
