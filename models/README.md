# `models/`

Everything the serving side reads at runtime. Only `chat_templates/` is tracked in git;
weights and caches are local (see the repo `.gitignore`).

```
models/
├── chat_templates/     tracked — the jinja templates vLLM must be started with
├── Show-Harness-VLMs/       the released adapters (mirror of the HF repo)
└── huggingface/        HF_HOME: base-model downloads (hub/, xet/)
```

## `chat_templates/`

Serving-side only. Training never reads a jinja file — LlamaFactory renders the
conversation from its own `template:` registry — but vLLM needs one, so these are
hand-aligned jinja that reproduce that rendering. A base model's own template does not:
Qwen3.5's official one emits `<think>\n\n</think>\n\n` after the assistant turn even with
`enable_thinking=false`, 4 tokens the adapter never saw, and the model answers through the
mismatch without complaint. `scripts/serve_vlm.sh` mounts the right file based on `FAMILY`:

| `FAMILY` | serve with | reproduces LF's |
| --- | --- | --- |
| `qwen3_5` | `qwen3_5_nothink.jinja` | `template: qwen3_5_nothink` |
| `gemma4` | `gemma4n.jinja` | `template: gemma4n` |
| `internvl3_5` | `internvl3_5.jinja` | `template: intern_vl` |

## `Show-Harness-VLMs/`

Local copy of [`showlab/Show-Harness-VLMs`](https://huggingface.co/showlab/Show-Harness-VLMs) —
inference-only LoRA adapters, one folder per backbone. Each carries `adapter_config.json`,
`adapter_model.safetensors`, tokenizer files, and the chat template it must be served with.

| folder | base model | epochs | train loss |
| --- | --- | --- | --- |
| `qwen3_5_0_8b` | `Qwen/Qwen3.5-0.8B` | 40 | 0.0535 |
| `qwen3_5_2b` | `Qwen/Qwen3.5-2B` | 40 | 0.0534 |
| `qwen3_5_4b` | `Qwen/Qwen3.5-4B` | 40 | 0.0575 |
| `qwen3_5_9b` | `Qwen/Qwen3.5-9B` | 40 | 0.0615 |
| `gemma4_e4b` | `google/gemma-4-E4B-it` | 40 | 0.0626 |
| `qwen3_5_2b_sim` | `Qwen/Qwen3.5-2B` | 30 | 0.0477 |

The first five are the same recipe on the same mix: single-arm MVTOKEN, `02_exchange_token`
(Franka and Piper cameras face opposite ways, so the training data swaps `MV_FWD`/`MV_BACK`
for one embodiment — deployment swaps back). `qwen3_5_2b_sim` is the simulation-only policy,
trained on the released `sim` split (RoboLab + ManiSkill together) — one adapter for both
simulators, served as `qwen3_5_2b_showharness_sim`.

Fetch one, or all of them:

```bash
ADAPTER=qwen3_5_2b bash scripts/model/download_vlm_model.sh   # + WITH_BASE=1 for its base
ADAPTER=all        bash scripts/model/download_vlm_model.sh
```

## Serving one

```bash
MODEL=Qwen/Qwen3.5-2B \
LORA=qwen3_5_2b_showharness_ft=models/Show-Harness-VLMs/qwen3_5_2b \
FAMILY=qwen3_5 \
bash scripts/serve_vlm.sh
```

`LORA`'s name is what clients request, and it is the value a robot config's
`vlm_backend` points at.

`MODEL` takes a hub id or a directory. A bare hub id resolves against
`huggingface/hub/<id>` first, so pre-fetching keeps the download out of server start-up:

```bash
MODEL=Qwen/Qwen3.5-2B bash scripts/model/download_vlm_model.sh
```

Otherwise vLLM downloads it itself on first start. A base model already on disk elsewhere
works too — pass its path as `MODEL` directly.

To get both in one go, `ADAPTER=<name> WITH_BASE=1` downloads the adapter and the base its
`adapter_config.json` names, then prints the serve command for it.
