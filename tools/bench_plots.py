"""
inference-benchmark plots.

Reads bench/results.jsonl (phase "6_infer", written by bin/infer_bench)
and renders four PNGs into runs/bench_graphs/:

  nfe_quality_frontier.png  decoder NFEs/byte vs agreement with greedy
  enc_dec_map.png           encoder/global vs decoder NFEs per byte
  acceptance.png            drafted-byte acceptance rate by config
  latency.png               mean wall-clock per prompt by config

Usage:
    source .venv/bin/activate
    python3 tools/bench_plots.py [--results bench/results.jsonl]
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="bench/results.jsonl")
    p.add_argument("--phase", default="6_infer")
    p.add_argument("--out", default="runs/bench_graphs")
    return p.parse_args(argv)


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
    # tag format: <model>_<method>[_eb]_B<k|B>_<a|g><T>  (greedy/selfspec simpler)
    model, _, rest = tag.partition("_")
    parts = rest.split("_")
    method = parts[0]
    is_eb = "eb" in parts[1:] or any(p.startswith("g") and p[1:].replace(".","").isdigit()
                                     for p in parts[1:])
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
    # Compact unique tag for annotations.
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


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    rows = [r for r in load_rows(args.results, args.phase)
            if r.get("name") == "gen_methods"]
    if not rows:
        raise SystemExit("no 6_infer/gen_methods rows found")
    # bench_write_json appends across runs; keep the LATEST row per tag.
    latest = {}
    for r in rows:
        latest[r["tag"]] = r
    rows = list(latest.values())

    parsed = []
    for r in rows:
        model, method, suffix, k, B, thresh = split_tag(r["tag"])
        m = r["metrics"]
        parsed.append(dict(
            model=model, method=method, k=k, B=B, thresh=thresh,
            dec=m["dec_nfe_per_byte"], enc=m["enc_nfe_per_byte"],
            acc=m["acceptance"], agree=m["agreement"],
            lat_ms=r["latency"]["mean"] * 1000.0,
            label=pretty_name(model, method, suffix, k, B, thresh),
            short=short_name(model, method, suffix, k, B, thresh),
        ))

    base = next(r for r in parsed if r["method"] == "greedy")

    # ------------------------------------------------------------------
    # 1. Quality frontier: decoder NFEs/byte vs agreement
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    OFFS = [(7, 5), (7, -11), (7, 12), (7, -4), (-14, 14), (7, 19)]
    for i, r in enumerate(parsed):
        if r["method"] == "greedy":
            continue
        ax.scatter(r["dec"], r["agree"],
                   c=MODEL_COLORS[r["model"]],
                   marker=METHOD_MARKERS[r["method"]], s=90,
                   edgecolors="black", linewidths=0.6, zorder=3)
        ax.annotate(r["short"], (r["dec"], r["agree"]),
                    textcoords="offset points", xytext=OFFS[i % len(OFFS)],
                    fontsize=7.5)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.axvline(base["dec"], color="gray", lw=0.8, ls=":", alpha=0.6,
               label="greedy baseline dec NFE")
    ax.set_xlabel("decoder forward passes per generated byte")
    ax.set_ylabel("agreement with same-model greedy output")
    ax.set_title("Quality/cost frontier of generation methods\n"
                 "(toy 300k-param models, held-out C code prompts)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "nfe_quality_frontier.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Efficiency map: encoder/global vs decoder NFEs
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    OFFS = [(7, 5), (7, -11), (7, 12), (7, -4), (-14, 14), (7, 19)]
    for i, r in enumerate(parsed):
        ax.scatter(r["dec"], r["enc"],
                   c=MODEL_COLORS[r["model"]],
                   marker=METHOD_MARKERS[r["method"]], s=90,
                   edgecolors="black", linewidths=0.6, zorder=3)
        ax.annotate(r["short"], (r["dec"], r["enc"]),
                    textcoords="offset points", xytext=OFFS[i % len(OFFS)],
                    fontsize=7.5)
    ax.scatter([1], [1], marker="*", s=260, c="green",
               edgecolors="black", zorder=4, label="baseline (1,1)")
    ax.set_xlabel("decoder NFEs / byte")
    ax.set_ylabel("encoder+global NFEs / byte")
    ax.set_title("Where each method spends its forward passes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "enc_dec_map.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Acceptance rates (verified methods only: raw BLT-D drafts have no
    #    verification step, so "acceptance" is meaningless for them)
    # ------------------------------------------------------------------
    drafted = [r for r in parsed
               if r["acc"] >= 0.0 and r["method"] != "blockdiff"]
    drafted.sort(key=lambda r: r["acc"])
    labels = [r["short"] for r in drafted]
    colors = [MODEL_COLORS[r["model"]] for r in drafted]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.barh(range(len(drafted)), [r["acc"] for r in drafted],
                   color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(drafted)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(1.0, color="gray", lw=0.8, ls="--")
    for i, r in enumerate(drafted):
        ax.text(r["acc"] + 0.01, i, f"{r['acc']:.2f}", va="center",
                fontsize=8)
    ax.set_xlabel("drafted bytes accepted by verification")
    ax.set_title("Speculative acceptance")
    ax.set_xlim(0, 1.08)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "acceptance.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Wall-clock latency (informational: cache use differs by path)
    # ------------------------------------------------------------------
    order = sorted(parsed, key=lambda r: r["lat_ms"])
    labels = [r["short"] for r in order]
    colors = [MODEL_COLORS[r["model"]] for r in order]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(range(len(order)), [r["lat_ms"] for r in order],
            color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=8)
    for i, r in enumerate(order):
        ax.text(r["lat_ms"] * 1.02, i, f"{r['lat_ms']:.0f}", va="center",
                fontsize=8)
    ax.set_xlabel("mean wall-clock ms per prompt (64 new bytes)")
    ax.set_title("Latency (note: KV-cache use differs between paths)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "latency.png"), dpi=150)
    plt.close(fig)

    print(f"wrote 4 PNGs to {args.out}/")


if __name__ == "__main__":
    main()
