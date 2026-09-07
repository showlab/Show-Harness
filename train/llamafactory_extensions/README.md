# llamafactory_extensions

A layer on top of upstream [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
Upstream source is never modified. mvtoken training needs two things it does not provide;
both attach from the outside, which is why this repo depends on upstream rather than a fork.

**Nothing is installed by default.** Qwen3.5 and InternVL training runs on stock upstream —
without `PYTHONPATH` pointing here, this directory is never even loaded.

## What it adds

| | Installed when | How |
| --- | --- | --- |
| `gemma4_unified` registration | `FAMILY=gemma4` | calls upstream's own `_register_composite_model()` |
| camera dropout | `CAMERA_DROPOUT>0` | wraps upstream's collator `__call__` |

**gemma4_unified** — upstream already ships `Gemma4Plugin` and the `gemma4`/`gemma4n`
templates; it just has no entry for this `model_type`'s module layout. Released gemma-4
weights put the vision backbone at `model.vision_embedder` (not `vision_tower`) and have a
projection-only audio side. Without the entry, `freeze_vision_tower` finds nothing to
freeze. `COMPOSITE_MODELS` is a module-level dict and `_register_composite_model()` is
upstream's own writer for it. If upstream ever adds the entry, this detects it and yields.

**camera dropout** — replaces a random subset of a sample's camera views with black frames
during training. Multi-camera policies tend to collapse onto whichever view is easiest to
read, then fall apart when that view goes uninformative or is missing at deployment. The
hook runs before upstream's `__call__` sees the features, so it only edits inputs. Frames
are replaced at the same size, so the visual token count is unchanged and the sequence
length computed during preprocessing still holds.

The switch is the `MVTOKEN_CAMERA_DROPOUT` env var rather than a yaml key: upstream's
`DataArguments` rejects unknown fields, so a custom yaml key would be refused outright.

## Why sitecustomize

Multi-GPU runs are re-launched by upstream's launcher through torchrun, and those workers
know nothing about what the parent process imported. torchrun does pass the environment
through, so putting this directory on `PYTHONPATH` makes every worker load
`sitecustomize.py` at interpreter startup — which is where the extensions attach.

`sitecustomize.py` first chains to any pre-existing module of that name. Python imports
only the first `sitecustomize` it finds, and we get there first via `PYTHONPATH`; silently
shadowing someone else's would be a hard bug to track down.

## Not ported

The fork also patched `Gemma4Plugin`'s audio branch (`padding="max_length"` →
`"longest"`). mvtoken is vision-only and never reaches that branch, and patching it from
the outside would mean wrapping the whole method. Revisit if audio-capable gemma4n
training is ever needed.

## Check it

```bash
PYTHONPATH=train/llamafactory_extensions MVTOKEN_EXT_DEBUG=1 MVTOKEN_CAMERA_DROPOUT=0.15 \
  python -c "
from llamafactory.model.model_utils.visual import COMPOSITE_MODELS
from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq as C
print('gemma4_unified:', 'gemma4_unified' in COMPOSITE_MODELS)
print('camera dropout:', getattr(C, '_mvtoken_camera_dropout_installed', False))"
```
