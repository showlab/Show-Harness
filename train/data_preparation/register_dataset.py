#!/usr/bin/env python3
"""Register a converted mvtoken dataset into LlamaFactory's ``data/dataset_info.json``.

A training yaml resolves ``dataset: <name>`` through that file, so a converted
rollouts.json has to be registered before it can be trained on. The entry shape is
inferred from the samples rather than passed in:

  * ``{instruction, input, output, images}``   -> alpaca + images
  * ``{instruction, input, output, videos}``   -> alpaca + videos
  * ``{messages, images}``                     -> sharegpt + images

Writing is idempotent: an existing entry of the same name is replaced, everything else
in the file keeps its order.

Register one dataset per call; to train on several, list them comma-separated in the
yaml's ``dataset:`` field, the way LlamaFactory expects.

Usage:
    python train/data_preparation/register_dataset.py robolab_0816_12task \
        --samples train/data/robolab_0816_12task/rollouts.json

    python train/data_preparation/register_dataset.py my_set --samples ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


SHAREGPT_TAGS = {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
}


def _first_sample(path: Path) -> dict:
    """Read one sample, tolerating both a JSON array and JSON Lines."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"ERROR: empty sample file: {path}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        first_line = next((ln for ln in text.splitlines() if ln.strip()), "")
        return json.loads(first_line)
    if isinstance(data, list):
        if not data:
            raise SystemExit(f"ERROR: no samples in {path}")
        return data[0]
    return data


def _media_of(sample: dict) -> list:
    """Flattened media paths of a sample (the videos field may nest one level)."""
    out = []
    for key in ("images", "videos"):
        for item in sample.get(key, []) or []:
            out.extend(item if isinstance(item, list) else [item])
    return out


def _has_relative_media(sample: dict) -> bool:
    media = _media_of(sample)
    return bool(media) and not os.path.isabs(media[0])


def _resolve_media_paths(src: Path, dst: Path) -> int:
    """Resolve relative media paths against src's directory, write to dst, return count."""
    root = src.parent
    samples = json.loads(src.read_text(encoding="utf-8"))
    n = 0
    for s in samples:
        for key in ("images", "videos"):
            if key not in s:
                continue
            fixed = []
            for item in s[key]:
                if isinstance(item, list):
                    fixed.append([str((root / p).resolve()) if not os.path.isabs(p) else p for p in item])
                    n += len(item)
                else:
                    fixed.append(str((root / item).resolve()) if not os.path.isabs(item) else item)
                    n += 1
            s[key] = fixed
    dst.write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
    return n


def _build_entry(file_name: str, sample: dict) -> dict:
    """Infer the dataset_info entry shape from one sample."""
    media = "videos" if "videos" in sample else "images"

    if "messages" in sample:
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", media: media},
            "tags": dict(SHAREGPT_TAGS),
        }

    if "instruction" not in sample:
        raise SystemExit(
            f"ERROR: sample has neither 'messages' nor 'instruction'; fields: {sorted(sample)}"
        )

    return {
        "file_name": file_name,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            media: media,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register a converted dataset into LlamaFactory's dataset_info.json."
    )
    parser.add_argument("name", help="dataset name; this is what the yaml's dataset: refers to")
    parser.add_argument("--samples", type=Path, required=True,
                        help="sample file produced by the converter (rollouts.json)")
    parser.add_argument("--lf-root", type=Path, default=os.environ.get("LF_ROOT"),
                        help="LlamaFactory checkout root (defaults to $LF_ROOT)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the entry without writing it")
    args = parser.parse_args()

    if args.lf_root is None:
        raise SystemExit("ERROR: no --lf-root and no $LF_ROOT; source train/scripts/llamafactory_env.sh first")

    lf_root = Path(args.lf_root).resolve()
    info_path = lf_root / "data" / "dataset_info.json"
    if not info_path.is_file():
        raise SystemExit(f"ERROR: {info_path} not found - is LF_ROOT correct?")

    samples = args.samples.resolve()
    if not samples.is_file():
        raise SystemExit(f"ERROR: sample file not found: {samples}")

    # LF resolves file_name against its own data/. Our datasets live in this repo
    # (train/data/), so write an absolute path: LF accepts it, and the data does
    # not disappear when third_party/ is cleaned.
    data_dir = lf_root / "data"
    try:
        file_name = str(samples.relative_to(data_dir))
    except ValueError:
        file_name = str(samples)

    # Distributable datasets store image paths relative to the sample file. LF has a
    # single global media_dir, so mixing two such datasets makes their roots collide.
    # Pin the paths at registration time and write a resolved copy: from then on the
    # dataset behaves like any other, and mixing or changing cwd is a non-issue.
    # The original file is untouched and stays distributable.
    resolved = None
    if _has_relative_media(_first_sample(samples)):
        resolved = lf_root / "data" / "resolved" / f"{args.name}.json"
        file_name = str(resolved.relative_to(lf_root / "data"))

    entry = _build_entry(file_name, _first_sample(samples))

    info = json.loads(info_path.read_text(encoding="utf-8"))
    action = "replacing" if args.name in info else "adding"

    print(f"{action} dataset_info.json entry [{args.name}]:")
    print(json.dumps({args.name: entry}, ensure_ascii=False, indent=2))

    if args.dry_run:
        if resolved:
            print(f"(would resolve relative image paths into {resolved})")
        print("(--dry-run, nothing written)")
        return

    if resolved:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        n = _resolve_media_paths(samples, resolved)
        print(f"relative image paths -> resolved {n} into {resolved}")

    info[args.name] = entry
    # No trailing newline: matches the file as it exists in LF, avoiding a spurious diff.
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {info_path}")


if __name__ == "__main__":
    main()
