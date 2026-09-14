"""
Deterministic sample corpus generator for fBLT.

Discovers all .c/.h source files in the repo, splits them 95/5 at the file
level using a hash-based deterministic assignment, concatenates each split
into raw .bin files, and writes a JSON manifest with SHA-256 checksums.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

EXCLUDE_DIRS = {
    "obj",
    "obj-cuda",
    "bin",
    "bin-cuda",
    "data",
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "node_modules",
}

SEP = b"\n"


def find_source_files():
    """Walk repo root for .c/.h files, excluding build/data dirs."""
    files = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in (".c", ".h"):
            continue
        # check if any parent is excluded
        rel = path.relative_to(REPO_ROOT)
        parts = rel.parts
        if any(p in EXCLUDE_DIRS for p in parts):
            continue
        files.append(rel)
    files.sort(key=lambda p: p.as_posix())
    return files


def split_files(files, seed):
    """Deterministic file-level split using sha256 hash.

    The seed is mixed into the hash so different seeds produce different
    splits while remaining fully reproducible for a given seed + file list.
    """
    train, heldout = [], []
    for rel in files:
        h = hashlib.sha256(f"{seed}:{rel.as_posix()}".encode()).digest()[:8]
        val = int.from_bytes(h, "little") % 100
        if val < 95:
            train.append(rel)
        else:
            heldout.append(rel)
    return train, heldout


def read_file_bytes(rel):
    """Read file as raw bytes with UTF-8 replace decoding."""
    return (REPO_ROOT / rel).read_bytes()


def build_stream(file_list, size_bytes):
    """Concatenate files in sorted order, capping at size_bytes.

    Returns (bytes, list_of_included_rels, total_source_bytes).
    Truncates mid-file cleanly to hit the cap exactly.
    """
    buf = bytearray()
    included = []
    total_source = 0
    for rel in file_list:
        data = read_file_bytes(rel)
        remaining = size_bytes - len(buf)
        if remaining <= 0:
            break
        # +1 for separator (except first file)
        needed = len(data) + (1 if buf else 0)
        if needed <= remaining:
            if buf:
                buf += SEP
            buf += data
            included.append(rel)
            total_source += len(data)
        else:
            # truncate to fit exactly
            can_use = remaining - (1 if buf else 0)
            if can_use > 0:
                if buf:
                    buf += SEP
                buf += data[:can_use]
                included.append(rel)
                total_source += can_use
            break
    if len(buf) < size_bytes:
        actual = len(buf)
        print(f"warning: pool smaller than cap ({actual} < {size_bytes} bytes)")
    return bytes(buf), included, total_source


def get_git_commit():
    """Get HEAD commit hash, fall back to 'unknown'."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    print("warning: could not determine git commit")
    return "unknown"


def generate_corpus(out_dir, train_size, heldout_size, seed):
    """Generate train.bin, heldout.bin, and manifest. Returns manifest dict."""
    files = find_source_files()
    train_files, heldout_files = split_files(files, seed)

    train_bytes, train_included, train_src = build_stream(train_files, train_size)
    heldout_bytes, heldout_included, heldout_src = build_stream(heldout_files, heldout_size)

    out_path = REPO_ROOT / out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    train_bin = out_path / "train.bin"
    heldout_bin = out_path / "heldout.bin"
    train_bin.write_bytes(train_bytes)
    heldout_bin.write_bytes(heldout_bytes)

    manifest = {
        "git_commit": get_git_commit(),
        "source_files": [
            {"path": rel.as_posix(), "size": (REPO_ROOT / rel).stat().st_size} for rel in files
        ],
        "train_files": [
            {"path": rel.as_posix(), "size": (REPO_ROOT / rel).stat().st_size}
            for rel in train_included
        ],
        "heldout_files": [
            {"path": rel.as_posix(), "size": (REPO_ROOT / rel).stat().st_size}
            for rel in heldout_included
        ],
        "train_sha256": hashlib.sha256(train_bytes).hexdigest(),
        "heldout_sha256": hashlib.sha256(heldout_bytes).hexdigest(),
        "train_size_bytes": len(train_bytes),
        "heldout_size_bytes": len(heldout_bytes),
        "seed": seed,
    }

    manifest_path = out_path / "sample_corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return manifest


def check_corpus(out_dir, train_size, heldout_size, seed):
    """Regenerate into temp dir, compare sha256 against manifest."""
    manifest_path = REPO_ROOT / out_dir / "sample_corpus_manifest.json"
    if not manifest_path.exists():
        print(f"error: manifest not found at {manifest_path}")
        return 1
    existing = json.loads(manifest_path.read_text())

    tmp = tempfile.mkdtemp()
    try:
        manifest = generate_corpus(tmp, train_size, heldout_size, seed)
        if manifest["train_sha256"] != existing["train_sha256"]:
            print("error: train.bin sha256 mismatch")
            print(f"  expected: {existing['train_sha256']}")
            print(f"  got:      {manifest['train_sha256']}")
            return 1
        if manifest["heldout_sha256"] != existing["heldout_sha256"]:
            print("error: heldout.bin sha256 mismatch")
            print(f"  expected: {existing['heldout_sha256']}")
            print(f"  got:      {manifest['heldout_sha256']}")
            return 1
        print("check passed: outputs match manifest")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="Deterministic sample corpus generator")
    ap.add_argument("--train-size-mb", type=float, default=2.0, help="max train corpus size in MB")
    ap.add_argument(
        "--heldout-size-mb", type=float, default=0.2, help="max heldout corpus size in MB"
    )
    ap.add_argument("--output-dir", default="data", help="output directory relative to repo root")
    ap.add_argument(
        "--check", action="store_true", help="drift guard: regenerate and compare sha256"
    )
    ap.add_argument("--seed", type=int, default=0, help="seed for hash-based file split")
    args = ap.parse_args()

    train_bytes = int(args.train_size_mb * 1024 * 1024)
    heldout_bytes = int(args.heldout_size_mb * 1024 * 1024)

    if args.check:
        return check_corpus(args.output_dir, train_bytes, heldout_bytes, args.seed)

    manifest = generate_corpus(args.output_dir, train_bytes, heldout_bytes, args.seed)

    n_source = len(manifest["source_files"])
    n_train = len(manifest["train_files"])
    n_heldout = len(manifest["heldout_files"])
    tb = manifest["train_size_bytes"]
    hb = manifest["heldout_size_bytes"]
    print(
        f"{n_source} files discovered, "
        f"{n_train} train / {n_heldout} heldout, "
        f"train {tb / 1e6:.2f} MB, heldout {hb / 1e6:.2f} MB"
    )
    print(
        f"output: {args.output_dir}/train.bin, "
        f"{args.output_dir}/heldout.bin, "
        f"{args.output_dir}/sample_corpus_manifest.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
