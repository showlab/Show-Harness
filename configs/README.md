# Configs

One robot config per (rig × mode):

| File | Rig / mode | Entry point |
| --- | --- | --- |
| `robot_franka.yaml` | Franka, zero-shot | `scripts/run_real.py` |
| `robot_franka_ft.yaml` | Franka, fine-tuned | `scripts/run_real_mvtoken.py` |
| `robot_piper.yaml` | Piper dual-arm, zero-shot | `scripts/run_real_dual.py` (single arm: `scripts/run_real.py --arm`) |
| `robot_piper_ft.yaml` | Piper, fine-tuned | `scripts/run_real_mvtoken.py` / `scripts/run_real_dual_mvtoken.py` |
| `robot_maniskill.yaml`, `robot_robolab.yaml` | simulators, fine-tuned | `scripts/run_maniskill_mvtoken.py` / `scripts/run_robolab_mvtoken.py` |
| `primitives_<embodiment>.yaml` | action-unit → motion mapping (axis signs, step, yaw) | read by the interpreters |

## Layering

A robot config composes other files around its own body:

```yaml
defaults:                                   # merged UNDER the body (body wins)
  - site/franka.yaml                        #   your rig identity (required)
  - {path: backends/internal.yaml, optional: true}
overlays:                                   # merged OVER the body (overlay wins)
  - {path: experiments/current_franka.yaml, optional: true}
```

- **The body** holds the shippable defaults: task, plugin toggles,
  hyperparameters (step sizes, budgets), safety values, and the public VLM
  backend profiles. Calibration scripts write into the body, and the body wins
  over `defaults:`, so a captured value always takes effect.
- **`site/`** holds what only your lab can know: the robot's address and
  camera serials (`site/franka.yaml`), per-arm poses and floors
  (`site/piper_arms.yaml`). Copy the shipped `.example` files and fill them
  in; the real-rig configs refuse to run without them. Sim configs pull no
  site layer — there is no hardware to describe.
- **`backends/` and `experiments/`** are optional layers for
  organization-internal endpoint catalogs and the currently active experiment
  state. They are absent from the public release, where every config resolves
  to its paper defaults.
- **Secrets never live in YAML.** Profiles name an environment variable
  (`api_key_env`), populated from the gitignored `secrets.env` (copy
  `secrets.env.example`; an optional `secrets.local.env` per-machine overlay
  is read on top, and shell-exported variables win over both).

Every knob has exactly one defaulting site: the yaml body, else the per-key
helper in `core/launch.py`.

## Choosing the VLM

`vlm_backend:` names one profile out of the config's own `vlm_backends:` map — endpoint,
model name, decode budget, chat-template kwargs. Nothing else in the config changes when
you switch.

| Config | Default | Also defined |
| --- | --- | --- |
| `robot_franka.yaml`, `robot_piper.yaml` | `gemini` | `chatgpt` (+ `local` on Franka) |
| `*_ft.yaml` | `qwen3_5_2b` | the other released adapters, `finetuned_local`, hosted profiles |
| `robot_maniskill.yaml`, `robot_robolab.yaml` | `qwen3_5_2b` (that sim's adapter) | `finetuned_local` |

The fine-tuned profiles are **placeholders naming a served adapter**, not weights: a
profile's `model:` must equal the name the server registered (`LORA=<name>=<path>` in
`scripts/serve_vlm.sh`). Released checkpoints are `<model>_showharness_<split>`, where the
split is `ft` for the real-robot corpus and `maniskill` / `robolab` for simulation. For
your own adapter, use `finetuned_local` and rename its `model:` to whatever you
registered. See [docs/finetuned.md](../docs/finetuned.md).

Hosted profiles name an environment variable via `api_key_env` rather than a key.

## Safety values are yours to calibrate

`z_floors:` (Franka) and `arms.*.z_floor_m` (Piper) are hard minimum
end-effector heights — the interpreter refuses to descend below the active
one. The shipped numbers fit the authors' tables, not yours: capture your own
with `scripts/franka/capture_z_floor.sh --name <n> --write` (Piper:
`scripts/piper/`) before the first autonomous run. The same applies to begin
poses and to the `primitives_*.yaml` axis signs if your camera mounting
differs.
