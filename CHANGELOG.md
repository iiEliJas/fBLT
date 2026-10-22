# Changelog

All notable changes to fBLT are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

## [1.0.0] - 2026-10-22

### Added
- Best-checkpoint saving on eval metric (`--save-best`, `--best-metric`, `--save-best-optim`)
- Metrics JSONL event stream (`--metrics-log`) and optional W&B integration
- LR warmup + cosine/min-lr schedule (`--warmup-steps`, `--lr-schedule`, `--min-lr`)
- Training telemetry: param count, ms/step, B/s, TFLOP/s, MFU (`--peak-tflops`)
- Inference `--top-k` and `--num-samples`
- Profiling wrapper script and docs
- Checkpoint metadata v2 (JSON header) with v1 backward compatibility
- Selective weight decay (norms excluded by default; `--weight-decay-norms`)
- macOS CPU CI job
- Packaging metadata and extras; robust binary resolution
- Finetune recipe config
- Ablation configs under `configs/ablations/`

### Changed
- README layout (`run/` -> `csrc/`), known limitations and support matrix
- AGENTS.md profiling link (`docs/Plan.md` -> `docs/PROFILING.md`)

### Fixed
- CLI reference stale tool paths and omitted built tools documented

