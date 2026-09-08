# MVTOKEN training

Turn rollouts collected by this repo into a training set, then train an mvtoken LoRA.

The trainer is upstream [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), pinned to
a commit that has been trained on, with its source unmodified. The two things mvtoken needs
on top live outside it in [llamafactory_extensions/](llamafactory_extensions/README.md), and
are mounted only for gemma4 or camera dropout — Qwen3.5 and InternVL run on stock upstream.

This repo supplies the data and the interface: rollouts, the prompt templates under
`prompts/<version>/`, and the converter that turns the two into training samples.

## Install

```bash
bash train/scripts/setup_llamafactory.sh              # qwen3_5 / internvl3_5
bash train/scripts/setup_llamafactory.sh --gemma4     # add this to also train gemma4
```

Clones upstream into `third_party/` and builds the venv (the torch / fla / tilelang pins and
the gcc shim are all baked in). Needs [uv](https://docs.astral.sh/uv/).

There is nothing to configure afterwards: every script derives its paths from the repo
layout. Already have a LLaMA-Factory checkout? Point at it once and setup symlinks it into
`third_party/`, after which everything resolves the same way:

```bash
LF_ROOT=/path/to/LLaMA-Factory bash train/scripts/setup_llamafactory.sh
```

## Data

Either download the released set, or convert your own rollouts. Both land in `train/data/`
and register the same way, so a run can mix them.

```bash
# released Show-Harness data (real + sim), downloaded and registered
bash train/scripts/download_dataset.sh

# your own rollouts -> training set, registered
SRC=$PWD/rollouts/my_env/my_task NAME=my_data \
    bash train/scripts/prepare_dataset.sh
```

Registration is one dataset per call. To train on several, list them comma-separated the
way LlamaFactory expects — `dataset: showharness_real,showharness_sim` in the yaml.

Already have a converted sample file? Register it directly:

```bash
python train/data_preparation/register_dataset.py my_data --samples /path/to/rollouts.json
```

`SRC` points at a **parent** directory: one subdirectory per task, each holding `rollout_000/`,
`rollout_001/`, … Each task uses the `task_text` from its own `metadata.json` — that is the
language signal multi-task training needs. Pass `TASK=...` to give every rollout the same one.

## Train

```bash
# 1. get data (above)

# 2. copy a template, fill the three <FILL_ME>
cp train/configs/qwen3_5_2b_lora.yaml train/configs/my_run.yaml

# 3. train
CONFIG=train/configs/my_run.yaml GPU=1,3 bash train/scripts/train.sh
```

Datasets live in `train/data/<NAME>/`, checkpoints in `train/saves/<model>/robot/<name>/`.
The yamls' `output_dir` is relative; `train.sh` pins it under `train/` so runs survive the
`rm -rf third_party/LlamaFactory` in [Upgrading upstream](#upgrading-upstream). Pass
`OUTPUT_DIR=/somewhere/else` for one run, or an absolute `output_dir:` in the yaml.

Common switches:

```bash
CAMERA_DROPOUT=0.15         # randomly blank camera views during training
MODEL_PATH=/path/to/weights # local weights instead of a hub download
WANDB_ENTITY=your_team      # project comes from WANDB_PROJECT
DRY_RUN=1                   # run the checks, print the command, do not train
```

## Serve

Serving lives on the inference side, in [`scripts/serve_vlm.sh`](../scripts/serve_vlm.sh):

```bash
MODEL=Qwen/Qwen3.5-2B FAMILY=qwen3_5 \
  LORA=qwen3_5_2b_showharness_ft=/path/to/adapter bash scripts/serve_vlm.sh
```

`FAMILY` selects the jinja that reproduces this run's `template:` at serve time. Adapter names follow
`<model>_showharness_<split>` for the released checkpoints (`ft` = the real-robot corpus,
`sim` = RoboLab and ManiSkill trained together) — see
[docs/finetuned.md](../docs/finetuned.md) for the full table.

## Layout

| | |
| --- | --- |
| `scripts/setup_llamafactory.sh` | **once**: clone (or symlink) upstream into `third_party/`, build the venv(s) |
| `scripts/llamafactory_env.sh` | **every run**: derive paths from the repo layout, validate, locate the gcc shim. Sourced by the scripts below, never run directly |
| `scripts/download_dataset.sh` | released data → `train/data/` → registered |
| `scripts/prepare_dataset.sh` | your rollouts → training set → registered |
| `scripts/train.sh` | training entry point, shared by all three families |
| `data_preparation/rollouts_to_alpaca.py` | the converter: one action step → one Alpaca sample |
| `data_preparation/register_dataset.py` | write a dataset into `dataset_info.json`; entry shape inferred from the samples |
| `data_preparation/generate_{subgoals,affordance}.py` | VLM-generated subgoals / grasp hints for the converter's `--use-*` modes (needs a running VLM server) |
| `configs/*.yaml` | one LoRA template per family, all keys accepted by upstream |
| `llamafactory_extensions/` | the layer outside upstream, mounted on demand |

## The prompt is the interface

Templates under `prompts/<version>/` feed both the converter (`--version`) and the runtime.
A field missing on one side — `recent_moves`, or a `{task}` rendered differently — degrades
the policy with no error. Changing the prompt version changes the interface: reconvert the
data and retrain.

`v3` is the unified prompt: one single-arm template for both embodiments, and what every
released checkpoint was trained on — convert with `--version v3` and the result drops
straight onto the released adapters. `v4` splits the single-arm prompt per embodiment
(`--franka` / `--piper`) and adds the dual-arm schemes; use it to fine-tune an adapter of
your own for the AgileX rig. The Franka side of v4 is byte-identical to v3, so for Franka
and sim the choice is cosmetic.

`--franka` / `--piper` must match what the LoRA was trained with. The two rigs observe from
opposite orientations, so `MV_FWD` and `MV_BACK` are swapped between them.

**Do not serve the base model's own chat template.** The `chat_template.jinja` saved into a
training output directory is the base model's copy, not what training rendered. Qwen3.5's
official template emits an empty think block after the assistant turn even with
`enable_thinking=false`, while LF's `qwen3_5_nothink` emits nothing — a 4-token difference
the model answers through without complaint. `models/chat_templates/` holds the aligned copies, which
`scripts/serve_vlm.sh` mounts for you.

## Things that bite

- **Re-running data prep overwrites `rollouts.json`.** Doing that mid-training desynchronizes
  LF's fingerprint from the data on disk. Stop the run before rebuilding.
- **Image paths end up absolute.** A locally converted set points straight into this repo's
  `rollouts/`; a downloaded one ships relative paths and `register_dataset.py` pins them at
  registration, because LlamaFactory has one global `media_dir` and two relative datasets
  would collide. Either way, moving or cleaning the image tree surfaces mid-training, so
  `train.sh` samples the first 200 entries before starting; `SKIP_CHECKS=1` opts out.
- **Single GPU + DeepSpeed needs `FORCE_TORCHRUN=1`**, otherwise `llamafactory-cli` refuses to
  start. `train.sh` sets it when it sees one GPU and a `deepspeed:` key.
- **To keep the multi-GPU lr curve on one GPU**, double `gradient_accumulation_steps`.
- **gemma4 trains in a second venv.** The released adapters were produced under two
  different transformers -- 5.7.0 for qwen3_5/internvl3_5, 5.12.1 for gemma4 (the model
  cards in LF's output dirs record it), and 5.12.1 is past LF's own bound, hence
  `DISABLE_VERSION_CHECK=1`. Add `--gemma4` to `setup_llamafactory.sh` to build it; `train.sh`
  routes on the family it infers from the config's `template:` (override with `FAMILY=`).
- **`overwrite_cache: true` is not optional.** LF's dataset fingerprint only hashes path
  strings, so images overwritten in place go unnoticed and stale cache gets trained on.
- **InternVL: stay on the image slot.** Its `<video>` slot does not fuse frames (each is its
  own 256 tokens), and LF trains with ImageNet normalization while vLLM serves with CLIP's.
- **wandb project/entity are env vars**, not yaml keys. The yaml's `project` is transformers'
  Trackio field, unrelated to wandb.

## Upgrading upstream

The pin lives in `train/scripts/setup_llamafactory.sh` as `LF_UPSTREAM_PIN`. Upstream refactors have
broken Qwen3.5 support more than once, so actually run a training on the new version before
moving it. `third_party/LlamaFactory/` is gitignored — delete it and rerun setup to switch;
`train/data/` lives in this repo and is unaffected.
