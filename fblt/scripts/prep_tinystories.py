"""
Downloads TinyStories from HuggingFace, shuffles deterministically (seed=42),
splits 95/5 train/heldout, and writes raw UTF-8 byte streams separated by
'\n' plus data/tinystories/manifest.json with per-story index.

Usage:
    python fblt/scripts/prep_tinystories.py [--out-dir data/tinystories] [--seed 42]
"""

import argparse
import hashlib
import json
import os
import random
import sys

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/tinystories")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-stories", type=int, default=0, help="Limit number of stories (0 = all)")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed. Run: pip install datasets", file=sys.stderr)
        return 1

    print("Downloading TinyStories from HuggingFace...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    print(f"Loaded {len(ds)} stories")

    rng = random.Random(args.seed)

    stories = []
    for i, row in enumerate(ds):
        if args.max_stories and i >= args.max_stories:
            break
        text = row.get("text", "")
        if not text or not text.strip():
            continue
        data = text.encode("utf-8", errors="replace")
        stories.append({"data": data, "index": i})

    print(f"Kept {len(stories)} non-empty stories")

    rng.shuffle(stories)

    n_train = int(len(stories) * 0.95)
    split = {"train": stories[:n_train], "heldout": stories[n_train:]}

    streams = {}
    index = {}
    for split_name, items in split.items():
        buf = bytearray()
        entries = []
        for item in items:
            offset = len(buf)
            buf += item["data"]
            buf += b"\n"
            entries.append(
                {
                    "offset": offset,
                    "length": len(item["data"]),
                    "path": f"story_{item['index']}",
                }
            )
        streams[split_name] = bytes(buf)
        index[split_name] = entries

    os.makedirs(args.out_dir, exist_ok=True)

    out_paths = {}
    for name in ("train", "heldout"):
        p = os.path.join(args.out_dir, f"{name}.bin")
        with open(p, "wb") as fh:
            fh.write(streams[name])
        out_paths[name] = p

    manifest = {
        "seed": args.seed,
        "source": "roneneldan/TinyStories",
        "total_stories": len(ds),
        "kept_stories": len(stories),
        "split_counts": {
            "train_stories": len(split["train"]),
            "heldout_stories": len(split["heldout"]),
        },
        "bytes": {
            name: {
                "path": out_paths[name],
                "sha256": hashlib.sha256(streams[name]).hexdigest(),
                "size_bytes": os.path.getsize(out_paths[name]),
            }
            for name in ("train", "heldout")
        },
        "index": index,
    }

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    tr = manifest["bytes"]["train"]["size_bytes"]
    ho = manifest["bytes"]["heldout"]["size_bytes"]
    print(f"train.bin   {tr / 1e6:.2f} MB ({manifest['split_counts']['train_stories']} stories)")
    print(f"heldout.bin {ho / 1e6:.2f} MB ({manifest['split_counts']['heldout_stories']} stories)")
    print(f"Total: {(tr + ho) / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
