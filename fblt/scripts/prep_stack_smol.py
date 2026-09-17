"""
Reads data/raw/the-stack-smol/data/c/data.json (JSONL), filters degenerate
files, exact-hash dedups content, deterministic-shuffles (seed=42), splits
95/5, and writes raw UTF-8 byte streams separated by a single '\n' plus
data/manifest.json (sha256 of both bins, seed, kept/rejected counts with
reasons, per-file index {offset,length,domain,path}).
"""

import argparse
import hashlib
import json
import os
import random
import sys

SEED = 42

MIN_FILE_BYTES = 100
MAX_FILE_BYTES = 100 * 1024

MIN_ALPHANUM_FRACTION = 0.1
MAX_AVG_LINE_LENGTH = 512


def iter_rows(path):
    opener = open
    if path.endswith(".gz"):
        import gzip

        opener = gzip.open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def domain_of(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".h":
        return "h"
    return "c"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/the-stack-smol/data/c/data.json")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    rng = random.Random(SEED)

    rejected = {"too_small": 0, "too_large": 0, "alphanum": 0, "line_length": 0, "duplicate": 0}
    kept = []
    seen_hashes = set()
    total = 0

    for row in iter_rows(args.input):
        total += 1
        content = row.get("content") or ""
        data = content.encode("utf-8", errors="replace")
        size = row.get("size", len(data))
        path = row.get("path", "")

        if size < MIN_FILE_BYTES or len(data) < MIN_FILE_BYTES:
            rejected["too_small"] += 1
            continue
        if size > MAX_FILE_BYTES or len(data) > MAX_FILE_BYTES:
            rejected["too_large"] += 1
            continue
        alpha = row.get("alphanum_fraction")
        if alpha is not None and alpha < MIN_ALPHANUM_FRACTION:
            rejected["alphanum"] += 1
            continue
        avg_line = row.get("avg_line_length")
        if avg_line is not None and avg_line > MAX_AVG_LINE_LENGTH:
            rejected["line_length"] += 1
            continue

        h = hashlib.sha256(data).hexdigest()
        if h in seen_hashes:
            rejected["duplicate"] += 1
            continue
        seen_hashes.add(h)

        kept.append({"data": data, "path": path, "domain": domain_of(path)})

    rng.shuffle(kept)

    n_train = int(len(kept) * 0.95)
    split = {"train": kept[:n_train], "heldout": kept[n_train:]}

    streams = {}
    index = {}
    for split_name, files in split.items():
        buf = bytearray()
        entries = []
        for f in files:
            offset = len(buf)
            buf += f["data"]
            buf += b"\n"
            entries.append(
                {
                    "offset": offset,
                    "length": len(f["data"]),
                    "domain": f["domain"],
                    "path": f["path"],
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

    # Per-domain held-out streams for the bpb_c / bpb_h breakout
    for dom in ("c", "h"):
        buf = bytearray()
        for e in index["heldout"]:
            if e["domain"] == dom:
                buf += streams["heldout"][e["offset"] : e["offset"] + e["length"]]
                buf += b"\n"
        p = os.path.join(args.out_dir, f"heldout_{dom}.bin")
        with open(p, "wb") as fh:
            fh.write(bytes(buf))
        out_paths[f"heldout_{dom}"] = p

    manifest = {
        "seed": SEED,
        "source": args.input,
        "total_rows": total,
        "kept_files": len(kept),
        "rejected": rejected,
        "split_counts": {
            "train_files": len(split["train"]),
            "heldout_files": len(split["heldout"]),
        },
        "bytes": {
            name: {
                "path": out_paths[name],
                "sha256": hashlib.sha256(streams[name]).hexdigest() if name in streams else None,
                "size_bytes": os.path.getsize(out_paths[name]),
            }
            for name in ("train", "heldout")
        },
        "index": index,
    }
    # sha256 of derived domain bins
    for dom in ("c", "h"):
        p = os.path.join(args.out_dir, f"heldout_{dom}.bin")
        hh = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                hh.update(chunk)
        manifest["bytes"][f"heldout_{dom}"] = {
            "path": p,
            "sha256": hh.hexdigest(),
            "size_bytes": os.path.getsize(p),
        }

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    tr = manifest["bytes"]["train"]["size_bytes"]
    ho = manifest["bytes"]["heldout"]["size_bytes"]
    print(f"kept {len(kept)}/{total} files (rejected: {rejected})")
    print(f"train.bin   {tr / 1e6:.2f} MB")
    print(f"heldout.bin {ho / 1e6:.2f} MB ({manifest['split_counts']['heldout_files']} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
