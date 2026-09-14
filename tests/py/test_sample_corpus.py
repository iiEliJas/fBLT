import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

SCRIPT = REPO_ROOT / "fblt" / "scripts" / "build_sample_corpus.py"

MANIFEST_KEYS = {
    "git_commit",
    "source_files",
    "train_files",
    "heldout_files",
    "train_sha256",
    "heldout_sha256",
    "train_size_bytes",
    "heldout_size_bytes",
    "seed",
}


def run_corpus(tmp_path, extra_args=None):
    cmd = [sys.executable, str(SCRIPT), "--output-dir", str(tmp_path)]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)


# ------------------------------------------------------------------
# Test 1: Script produces non-empty output files and valid manifest
#
def test_corpus_outputs_exist(tmp_path):
    result = run_corpus(tmp_path)
    assert result.returncode == 0, result.stderr

    train = tmp_path / "train.bin"
    heldout = tmp_path / "heldout.bin"
    manifest = tmp_path / "sample_corpus_manifest.json"

    assert train.exists() and train.stat().st_size > 0
    assert heldout.exists() and heldout.stat().st_size > 0
    assert manifest.exists()

    data = json.loads(manifest.read_text())
    assert isinstance(data, dict)
    assert MANIFEST_KEYS.issubset(data.keys())
    assert isinstance(data["train_files"], list)
    assert isinstance(data["heldout_files"], list)
    assert data["train_size_bytes"] > 0
    assert data["heldout_size_bytes"] > 0


# ------------------------------------------------------------------
# Test 2: Deterministic output across repeated runs
#
def test_deterministic_output(tmp_path):
    run_corpus(tmp_path)
    train_sha_1 = (tmp_path / "train.bin").read_bytes()
    heldout_sha_1 = (tmp_path / "heldout.bin").read_bytes()

    run_corpus(tmp_path)
    train_sha_2 = (tmp_path / "train.bin").read_bytes()
    heldout_sha_2 = (tmp_path / "heldout.bin").read_bytes()

    assert train_sha_1 == train_sha_2
    assert heldout_sha_1 == heldout_sha_2


# ------------------------------------------------------------------
# Test 3: No file-level overlap between train and heldout splits
#
def test_no_byte_overlap(tmp_path):
    result = run_corpus(tmp_path)
    assert result.returncode == 0, result.stderr

    manifest = json.loads((tmp_path / "sample_corpus_manifest.json").read_text())
    train_paths = {f["path"] for f in manifest["train_files"]}
    heldout_paths = {f["path"] for f in manifest["heldout_files"]}

    overlap = train_paths & heldout_paths
    assert not overlap, f"files appear in both splits: {overlap}"


# ------------------------------------------------------------------
# Test 4: --check mode detects drift when manifest is corrupted
#
def test_check_detects_drift(tmp_path):
    # initial generation
    result = run_corpus(tmp_path)
    assert result.returncode == 0, result.stderr

    # --check should pass on unmodified outputs
    result = run_corpus(tmp_path, extra_args=["--check"])
    assert result.returncode == 0, result.stderr

    # corrupt manifest: tamper with train_sha256
    manifest_path = tmp_path / "sample_corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    original_train_sha = manifest["train_sha256"]
    manifest["train_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    result = run_corpus(tmp_path, extra_args=["--check"])
    assert result.returncode == 1

    # restore manifest and corrupt heldout_sha256 instead
    manifest = json.loads(manifest_path.read_text())
    manifest["train_sha256"] = original_train_sha
    manifest["heldout_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    result = run_corpus(tmp_path, extra_args=["--check"])
    assert result.returncode == 1
