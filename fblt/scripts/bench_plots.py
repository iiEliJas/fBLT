"""
Benchmark plots.

Renders PNGs into graphs/:

  Toy-scale (from bench/results.jsonl, phase 6_infer):
    nfe_quality_frontier.png  decoder NFEs/byte vs agreement with greedy
    enc_dec_map.png           encoder/global vs decoder NFEs per byte
    acceptance.png            drafted-byte acceptance rate by config
    latency.png               mean wall-clock per prompt by config

  At-scale (from bench/results.jsonl, phase 7_cuda_bench):
    cuda_speedup.png          CPU vs CUDA meso-scale forward/backward

Usage:
    source .venv/bin/activate
    python3 tools/bench_plots.py [--results bench/results.jsonl] [--out graphs]
"""

import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Shared style ──────────────────────────────────────────────────────

MODEL_COLORS = {"plain": "#4477aa", "bltd": "#ee6677"}
METHOD_MARKERS = {
    "greedy": "o",
    "selfspec": "s",
    "blockdiff": "^",
    "blockdv": "D",
}
PRETTY = {
    "greedy": "greedy (baseline)",
    "selfspec": "BLT-S k={k}",
    "blockdiff": "BLT-D B={B} a={t}",
    "blockdv": "BLT-DV B={B} a={t}",
    "blockdv_onestep": "BLT-DV one-step B=8",
    "blockdv_eb": "BLT-DV-EB B=8 g={t}",
    "blockdiff_eb": "BLT-D-EB B=8 g={t}",
}

ARM_COLORS = {
    "plain": "#4477aa",
    "base": "#55aadd",
    "dec1": "#77ccee",
    "dec3": "#99ddff",
    "xlast": "#bbeeff",
    "late": "#ee6677",
    "hit": "#ee9988",
    "entp": "#cc4455",
    "e256": "#aaaaaa",
}
ARM_ORDER = ["plain", "late", "hit", "dec1", "base", "xlast", "dec3", "entp", "e256"]
ARM_LABELS = {
    "plain": "plain (no diff.)",
    "base": "baseline 2/2/2",
    "dec1": "dec-layers=1",
    "dec3": "dec-layers=3",
    "xlast": "cross-attn=last",
    "late": "mask-late-step 14k",
    "hit": "t-hi-start 0.8",
    "entp": "entropy patches",
    "e256": "E=256 H=512 10k",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="bench/results.jsonl")
    p.add_argument("--phase", default="6_infer")
    p.add_argument("--out", default="graphs")
    return p.parse_args(argv)


# ── Toy-scale parsers ────────────────────────────────────────────────


def load_rows(path, phase):
    rows = []
    if not os.path.exists(path):
        raise SystemExit(f"missing results file: {path}")
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("phase") != phase:
                continue
            rows.append(rec)
    return rows


def split_tag(tag):
    # tag format: <model>[_cuda]_<method>[_eb]_B<k|B>_<a|g><T>
    # Strip optional _cuda infix before parsing method.
    parts = tag.split("_")
    if len(parts) >= 3 and parts[1] == "cuda":
        model = parts[0]
        rest = "_".join(parts[2:])
    else:
        model, _, rest = tag.partition("_")
    parts = rest.split("_")
    method = parts[0]
    is_eb = "eb" in parts[1:] or any(
        p.startswith("g") and p[1:].replace(".", "").isdigit() for p in parts[1:]
    )
    k = B = None
    thresh = None
    for part in parts[1:]:
        if part.startswith("k") and part[1:].isdigit():
            k = int(part[1:])
        elif part.startswith("B") and part[1:].isdigit():
            B = int(part[1:])
        elif (part.startswith("a") or part.startswith("g")) and part[1:]:
            try:
                thresh = float(part[1:])
            except ValueError:
                pass
    suffix = "_eb" if is_eb and method in ("blockdiff", "blockdv") else ""
    return model, method, suffix, k, B, thresh


def short_name(model, method, suffix, k, B, thresh):
    mp = "P" if model == "plain" else "D"
    if method == "greedy":
        return mp + "-greedy"
    if method == "selfspec":
        return f"{mp}-S{k}"
    if method == "blockdiff":
        if suffix:
            return f"{mp}-BD-EB{thresh:g}"
        return f"{mp}-BD{B}"
    if method == "blockdv":
        if thresh == 0.0:
            return mp + "-DV1step"
        if suffix:
            return f"{mp}-DV-EB{thresh:g}"
        return f"{mp}-DV{B}"
    return method


def pretty_name(model, method, suffix, k, B, thresh):
    key = method + suffix
    tmpl = PRETTY.get(key, key)
    label = model.upper() + " "
    if "{k}" in tmpl:
        return label + tmpl.format(k=k)
    if "{B}" in tmpl:
        return label + tmpl.format(B=B, t=thresh)
    return label + tmpl


# ── At-scale parsers ─────────────────────────────────────────────────

# Fixed method order from BLTD_CFGS_ALL in infer_bench.c.
BLTD_METHODS = [
    "greedy",
    "selfspec_k4",
    "selfspec_k8",
    "selfspec_k16",
    "blockdiff_B4_a0.70",
    "blockdiff_B8_a0.70",
    "blockdiff_B16_a0.70",
    "blockdv_B4_a0.70",
    "blockdv_B8_a0.70",
    "blockdv_B16_a0.70",
    "blockdv_onestep_B8_a0.00",
    "blockdv_eb_B8_g1.00",
    "blockdv_eb_B8_g2.00",
    "blockdiff_eb_B8_g1.00",
    "blockdv_hetv_B8_a0.70_hetv",
    "blockdv_bal_B8_a0.70_bal",
    "blockdv_adapt_B8_a0.70_adapt",
    "blockdv_adapt_bal_B8_a0.70_bal_adapt",
]


def parse_inference_log():
    """Return {arm: {method: {dec, enc, accept, agree, lat_ms}}}."""
    results = {}
    path = "logs/s6_infer.log"
    if not os.path.exists(path):
        return results
    current_arm = None
    in_data = False
    row_idx = 0
    for line in open(path):
        line = line.rstrip()
        m = re.match(r"^=== (\w+) ===$", line)
        if m and "done" not in line:
            current_arm = m.group(1)
            in_data = False
            row_idx = 0
            results.setdefault(current_arm, {})
            continue
        if line.startswith("---"):
            in_data = True
            continue
        if not in_data or not current_arm:
            continue
        if not line.strip():
            continue
        if line.startswith("tag ") or line.startswith("[INFER"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            dec = float(parts[1])
            enc = float(parts[2])
            accept = float(parts[3])
            agree = float(parts[4])
            lat = float(parts[5])
        except (ValueError, IndexError):
            continue
        if row_idx < 4:
            plain_methods = ["greedy", "selfspec_k4", "selfspec_k8", "selfspec_k16"]
            method = f"plain_{plain_methods[row_idx]}"
        else:
            bltd_idx = row_idx - 4
            if bltd_idx < len(BLTD_METHODS):
                method = f"bltd_{BLTD_METHODS[bltd_idx]}"
            else:
                method = f"unknown_{bltd_idx}"
        results[current_arm][method] = {
            "dec": dec,
            "enc": enc,
            "accept": accept,
            "agree": agree,
            "lat_ms": lat,
        }
        row_idx += 1
    return results


def parse_cuda_bench():
    """Return {tag: averaged_metrics_dict} from results.jsonl phase 7_cuda_bench.

    Multiple runs per tag are averaged.  A warning is printed showing the
    min/max range for each metric so the reader knows the variance that was
    collapsed.
    """
    raw = {}  # tag -> list of metrics dicts
    path = "bench/results.jsonl"
    if not os.path.exists(path):
        return {}
    for line in open(path):
        r = json.loads(line)
        if r.get("phase") != "7_cuda_bench":
            continue
        raw.setdefault(r["tag"], []).append(r.get("metrics", {}))

    results = {}
    for tag, runs in raw.items():
        if len(runs) == 1:
            results[tag] = runs[0]
            continue
        # Collect all metric keys across runs
        keys = set()
        for m in runs:
            keys.update(m.keys())
        # Average each metric; track min/max for the warning
        averaged = {}
        for k in sorted(keys):
            vals = [m[k] for m in runs if k in m]
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            averaged[k] = avg
            lo, hi = min(vals), max(vals)
            if lo != hi:
                print(
                    f"  warn {tag}: {k} range [{lo:.4g} .. {hi:.4g}] (avg {avg:.4g}, n={len(vals)})"
                )
        results[tag] = averaged
    return results


# ── Toy-scale plots ──────────────────────────────────────────────────


def plot_toy_quality_frontier(parsed, base, out_dir):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Filter out greedy (baseline point plotted separately as vertical line)
    pts = [r for r in parsed if r["method"] != "greedy"]

    # Cluster-aware label staggering: group points whose x values are within
    # ~0.3 of each other, then fan labels out with increasing magnitude and
    # alternating up/down direction within each cluster.
    pts_sorted = sorted(pts, key=lambda r: (r["dec"], r["agree"]))
    clusters = []  # list of lists of (index_in_pts_sorted, point)
    for j, r in enumerate(pts_sorted):
        placed = False
        for cl in clusters:
            if abs(r["dec"] - cl[-1][1]["dec"]) < 0.3:
                cl.append((j, r))
                placed = True
                break
        if not placed:
            clusters.append([(j, r)])

    # Assign offsets per cluster
    offsets = {}  # point index -> (x_off, y_off)
    for cl in clusters:
        for rank, (j, r) in enumerate(cl):
            # Magnitude increases with rank; direction alternates
            mag = 7 + rank * 4
            y_dir = 1 if rank % 2 == 0 else -1
            # Use different x-directions for variety
            x_dir = [7, -14, 12, -10, 7, -14, 12, -10][rank % 8]
            offsets[j] = (x_dir, y_dir * mag)

    for j, r in enumerate(pts_sorted):
        ax.scatter(
            r["dec"],
            r["agree"],
            c=MODEL_COLORS[r["model"]],
            marker=METHOD_MARKERS[r["method"]],
            s=90,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
        xyoff = offsets.get(j, (7, 5))
        ax.annotate(
            r["short"],
            (r["dec"], r["agree"]),
            textcoords="offset points",
            xytext=xyoff,
            fontsize=7.5,
        )

    ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.axvline(
        base["dec"], color="gray", lw=0.8, ls=":", alpha=0.6, label="greedy baseline dec NFE"
    )
    ax.set_xlabel("decoder forward passes per generated byte")
    ax.set_ylabel("agreement with same-model greedy output")
    ax.set_title(
        "Quality/cost frontier of generation methods\n"
        "(toy 300k-param models, held-out C code prompts)"
    )
    ax.set_ylim(-0.05, 1.2)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "nfe_quality_frontier.png"), dpi=150)
    plt.close(fig)


def plot_toy_enc_dec_map(parsed, out_dir):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    OFFS = [
        (7, 5),
        (7, -11),
        (7, 12),
        (7, -4),
        (-14, 14),
        (7, 19),
        (-14, -10),
        (12, -14),
        (-14, 4),
        (12, 8),
    ]
    for i, r in enumerate(parsed):
        ax.scatter(
            r["dec"],
            r["enc"],
            c=MODEL_COLORS[r["model"]],
            marker=METHOD_MARKERS[r["method"]],
            s=90,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
        ax.annotate(
            r["short"],
            (r["dec"], r["enc"]),
            textcoords="offset points",
            xytext=OFFS[i % len(OFFS)],
            fontsize=7.5,
        )
    ax.scatter(
        [1], [1], marker="*", s=260, c="green", edgecolors="black", zorder=4, label="baseline (1,1)"
    )
    ax.set_xlabel("decoder NFEs / byte")
    ax.set_ylabel("encoder+global NFEs / byte")
    ax.set_title("Where each method spends its forward passes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "enc_dec_map.png"), dpi=150)
    plt.close(fig)


def plot_toy_acceptance(parsed, out_dir):
    drafted = [r for r in parsed if r["acc"] >= 0.0 and r["method"] != "blockdiff"]
    drafted.sort(key=lambda r: r["acc"])
    labels = [r["short"] for r in drafted]
    colors = [MODEL_COLORS[r["model"]] for r in drafted]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.barh(
        range(len(drafted)),
        [r["acc"] for r in drafted],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(drafted)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(1.0, color="gray", lw=0.8, ls="--")
    for i, r in enumerate(drafted):
        ax.text(r["acc"] + 0.01, i, f"{r['acc']:.2f}", va="center", fontsize=8)
    ax.set_xlabel("drafted bytes accepted by verification")
    ax.set_title("Speculative acceptance")
    ax.set_xlim(0, 1.08)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "acceptance.png"), dpi=150)
    plt.close(fig)


def plot_toy_latency(parsed, out_dir):
    order = sorted(parsed, key=lambda r: r["lat_ms"])
    labels = [r["short"] for r in order]
    colors = [MODEL_COLORS[r["model"]] for r in order]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(
        range(len(order)),
        [r["lat_ms"] for r in order],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=8)
    for i, r in enumerate(order):
        ax.text(r["lat_ms"] * 1.02, i, f"{r['lat_ms']:.0f}", va="center", fontsize=8)
    ax.set_xlabel("mean wall-clock ms per prompt (64 new bytes)")
    ax.set_title("Latency (note: KV-cache use differs between paths)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "latency.png"), dpi=150)
    plt.close(fig)


def plot_inference_nfe(out_dir):
    data = parse_inference_log()
    arms = [a for a in ARM_ORDER if a in data]
    if not arms:
        print(
            "  skip inference_nfe: no per-arm inference data exists "
            "(logs/s6_infer.log was never generated; the 6_infer phase "
            "in results.jsonl benchmarks a single checkpoint across "
            "inference methods, but has no arm dimension — it cannot "
            "provide per-arm NFE/acceptance breakdowns) — needs a "
            "future benchmarking pass"
        )
        return

    method = "bltd_blockdv_onestep_B8_a0.00"
    dec_vals, enc_vals, acc_vals, labels, colors = [], [], [], [], []
    for arm in arms:
        row = data[arm].get(method)
        if not row:
            continue
        dec_vals.append(row["dec"])
        enc_vals.append(row["enc"])
        acc_vals.append(row["accept"])
        labels.append(ARM_LABELS.get(arm, arm))
        colors.append(ARM_COLORS.get(arm, "#888888"))

    if not dec_vals:
        print("  skip inference_nfe: no onestep data in per-arm inference results")
        return

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(y, dec_vals, color=colors, edgecolor="black", linewidth=0.5, label="decoder NFE/byte")
    ax.barh(
        y,
        enc_vals,
        left=dec_vals,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.5,
        label="encoder NFE/byte",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("forward passes per generated byte")
    ax.set_title(
        "Inference: total NFE by configuration\n(BLT-DV onestep B=8, 3.5M-param models, CUDA)"
    )
    for i, (d, e, a) in enumerate(zip(dec_vals, enc_vals, acc_vals)):
        total = d + e
        ax.text(total + 0.02, i, f"{total:.2f} (acc {a:.0%})", va="center", fontsize=7.5)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "inference_nfe.png"), dpi=150)
    plt.close(fig)
    print("  wrote inference_nfe.png")


def plot_quality_speed(out_dir):
    train = {}  # at_scale data not available (per-arm inference data missing)
    infer = parse_inference_log()
    method = "bltd_blockdv_onestep_B8_a0.00"

    arms, bpbs, nfes, accs, colors = [], [], [], [], []
    for arm in ARM_ORDER:
        if arm not in train or arm not in infer:
            continue
        row = infer[arm].get(method)
        if not row:
            continue
        arms.append(arm)
        bpbs.append(train[arm]["bpb_avg"])
        nfes.append(row["dec"] + row["enc"])
        accs.append(row["accept"])
        colors.append(ARM_COLORS.get(arm, "#888888"))

    if not arms:
        print(
            "  skip quality_speed: no per-arm inference data exists "
            "(logs/s6_infer.log was never generated; the 6_infer phase "
            "in results.jsonl benchmarks a single checkpoint across "
            "inference methods, but has no arm dimension — it cannot "
            "provide per-arm NFE/acceptance breakdowns) — needs a "
            "future benchmarking pass"
        )
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sizes = [max(40, a * 300) for a in accs]
    ax.scatter(bpbs, nfes, s=sizes, c=colors, edgecolors="black", linewidths=0.6, zorder=3)
    OFFS = [
        (8, 4),
        (8, -8),
        (-10, 8),
        (8, 12),
        (-10, -8),
        (8, -12),
        (-10, 12),
        (8, 16),
        (-10, -12),
        (14, -4),
    ]
    for i, arm in enumerate(arms):
        lbl = ARM_LABELS.get(arm, arm)
        ax.annotate(
            lbl,
            (bpbs[i], nfes[i]),
            textcoords="offset points",
            xytext=OFFS[i % len(OFFS)],
            fontsize=7.5,
        )
    ax.set_xlabel("training BPB (lower = better quality)")
    ax.set_ylabel("inference NFE/byte (lower = faster)")
    ax.set_title("Quality vs speed trade-off\n(point size = acceptance rate, 3.5M-param models)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "quality_speed.png"), dpi=150)
    plt.close(fig)
    print("  wrote quality_speed.png")


def plot_cuda_speedup(out_dir):
    bench = parse_cuda_bench()
    if not bench:
        print("  skip cuda_speedup: no data")
        return

    # CPU reference: read from the same bench dict via _cpu tags
    # (meso_pipeline_{seq}_cpu / meso_pipeline_bf16_{seq}_cpu).
    # Config: E=64, H=128, L=2 (BENCH_E/BENCH_HID/BENCH_L in cuda_bench.c),
    # fixed stride-4 patching (scenario_build).
    seqs = [256, 1024]
    cuda_fwd, cuda_bwd = [], []
    bf16_fwd, bf16_bwd = [], []
    cpu_fwd, cpu_bwd = [], []
    for s in seqs:
        tag = f"meso_pipeline_{s}_cuda"
        if tag in bench:
            m = bench[tag]
            cuda_fwd.append(m["fwd_ms"])
            cuda_bwd.append(m["bwd_ms"])
        else:
            cuda_fwd.append(0)
            cuda_bwd.append(0)
        bf16_tag = f"meso_pipeline_bf16_{s}_cuda"
        if bf16_tag in bench:
            m = bench[bf16_tag]
            bf16_fwd.append(m["fwd_ms"])
            bf16_bwd.append(m["bwd_ms"])
        else:
            bf16_fwd.append(0)
            bf16_bwd.append(0)
        cpu_tag = f"meso_pipeline_{s}_cpu"
        if cpu_tag in bench:
            m = bench[cpu_tag]
            cpu_fwd.append(m["fwd_ms"])
            cpu_bwd.append(m["bwd_ms"])
        else:
            cpu_fwd.append(0)
            cpu_bwd.append(0)

    x = np.arange(len(seqs))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Forward
    ax1.bar(
        x - width, cpu_fwd, width, label="CPU", color="#aaaaaa", edgecolor="black", linewidth=0.5
    )
    ax1.bar(
        x, cuda_fwd, width, label="CUDA fp32", color="#ee6677", edgecolor="black", linewidth=0.5
    )
    ax1.bar(
        x + width,
        bf16_fwd,
        width,
        label="CUDA bf16",
        color="#44aa77",
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"seq={s}" for s in seqs])
    ax1.set_ylabel("forward time (ms)")
    ax1.set_title("Forward pass")
    ax1.legend(fontsize=8)
    ax1.set_yscale("log")
    for i, (cv, cc, cb) in enumerate(zip(cpu_fwd, cuda_fwd, bf16_fwd)):
        ax1.text(i - width, cv * 1.2, f"{cv:.0f}", ha="center", fontsize=7)
        ax1.text(i, cc * 1.2, f"{cc:.1f}", ha="center", fontsize=7)
        ax1.text(i + width, cb * 1.2, f"{cb:.1f}", ha="center", fontsize=7)
    ax1.grid(axis="y", alpha=0.25)

    # Backward
    ax2.bar(
        x - width, cpu_bwd, width, label="CPU", color="#aaaaaa", edgecolor="black", linewidth=0.5
    )
    ax2.bar(
        x, cuda_bwd, width, label="CUDA fp32", color="#ee6677", edgecolor="black", linewidth=0.5
    )
    ax2.bar(
        x + width,
        bf16_bwd,
        width,
        label="CUDA bf16",
        color="#44aa77",
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"seq={s}" for s in seqs])
    ax2.set_ylabel("backward time (ms)")
    ax2.set_title("Backward pass")
    ax2.legend(fontsize=8)
    ax2.set_yscale("log")
    for i, (cv, cc, cb) in enumerate(zip(cpu_bwd, cuda_bwd, bf16_bwd)):
        ax2.text(i - width, cv * 1.2, f"{cv:.0f}", ha="center", fontsize=7)
        ax2.text(i, cc * 1.2, f"{cc:.1f}", ha="center", fontsize=7)
        ax2.text(i + width, cb * 1.2, f"{cb:.1f}", ha="center", fontsize=7)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "CPU vs CUDA fp32 vs CUDA bf16 pipeline timing (E=64 L=2, fixed stride-4)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cuda_speedup.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote cuda_speedup.png")


# ── main ─────────────────────────────────────────────────────────────


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    # Toy-scale plots from results.jsonl.
    rows = [r for r in load_rows(args.results, args.phase) if r.get("name") == "gen_methods"]
    if rows:
        latest = {}
        for r in rows:
            latest[r["tag"]] = r
        all_rows = list(latest.values())

        # Filter to CUDA backend + entropy patching only.
        # CPU tags lack _cuda infix (always entropy-patched by default).
        # CUDA fixp tags end with _fixp (fixed patching — different NFE).
        # This collapses ~66 tags down to ~22, matching BENCHMARKS.md's
        # "Entropy patching (default)" table and eliminating CPU/CUDA
        # label collisions (identical NFE coordinates across backends).
        n_before = len(all_rows)
        rows = [r for r in all_rows if "_cuda" in r["tag"] and not r["tag"].endswith("_fixp")]
        n_dropped = n_before - len(rows)
        print(
            f"  toy-scale: kept {len(rows)} CUDA+entropy tags, "
            f"dropped {n_dropped} (CPU + CUDA fixp)"
        )

        parsed = []
        for r in rows:
            model, method, suffix, k, B, thresh = split_tag(r["tag"])
            m = r["metrics"]
            parsed.append(
                dict(
                    model=model,
                    method=method,
                    k=k,
                    B=B,
                    thresh=thresh,
                    dec=m["dec_nfe_per_byte"],
                    enc=m["enc_nfe_per_byte"],
                    acc=m["acceptance"],
                    agree=m["agreement"],
                    lat_ms=r["latency"]["mean"] * 1000.0,
                    label=pretty_name(model, method, suffix, k, B, thresh),
                    short=short_name(model, method, suffix, k, B, thresh),
                )
            )
        base = next(r for r in parsed if r["method"] == "greedy")

        plot_toy_quality_frontier(parsed, base, args.out)
        plot_toy_enc_dec_map(parsed, args.out)
        plot_toy_acceptance(parsed, args.out)
        plot_toy_latency(parsed, args.out)
        print(f"  wrote 4 toy-scale PNGs to {args.out}/")
    else:
        print("  skip toy-scale plots: no 6_infer data")

    # At-scale plots from CUDA bench.
    plot_inference_nfe(args.out)
    plot_quality_speed(args.out)
    plot_cuda_speedup(args.out)

    print(f"done — all graphs in {args.out}/")


if __name__ == "__main__":
    main()
