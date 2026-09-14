"""
Reads the JSON produced by bench/harness.c,
filters by phase/tag, and renders a Markdown comparison table.
Optionally produces a matplotlib PNG (e.g. BPB vs FLOPs, quality vs
memory-bandwidth)

Usage:
    python3 tools/bench_report.py                                # all runs
    python3 tools/bench_report.py --phase 5.1_ngram              # filter
    python3 tools/bench_report.py --baseline baseline_v1         # delta column
    python3 tools/bench_report.py --x flops_per_byte --y bpb --plot plot.png

Rows are keyed by (name, tag)
"""

import argparse
import json
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Render benchmark JSONL results as Markdown/PNG.")
    p.add_argument(
        "--results",
        default="bench/results.jsonl",
        help="path to results.jsonl (default: bench/results.jsonl)",
    )
    p.add_argument(
        "--phase",
        action="append",
        default=[],
        help="keep only these experiment groups (repeatable)",
    )
    p.add_argument("--tag", action="append", default=[], help="keep only these tags (repeatable)")
    p.add_argument("--name", action="append", default=[], help="keep only these names (repeatable)")
    p.add_argument(
        "--baseline", default=None, help="tag of the baseline run; adds %%delta columns vs it"
    )
    p.add_argument(
        "--sort",
        default=None,
        metavar="KEY",
        help="sort rows by latency field or metric key (e.g. mean, bpb)",
    )
    p.add_argument(
        "--x",
        default=None,
        metavar="KEY",
        help="metric key for the plot x-axis (latency field or metric)",
    )
    p.add_argument(
        "--y",
        default=None,
        metavar="KEY",
        help="metric key for the plot y-axis (latency field or metric)",
    )
    p.add_argument(
        "--plot",
        default=None,
        metavar="PNG",
        help="write a scatter PNG to this path (requires --x/--y and matplotlib)",
    )
    p.add_argument(
        "--compare",
        action="append",
        default=[],
        help="tag to include in a grouped BPB-vs-throughput bar chart "
        "(repeatable; --baseline tag is auto-included and highlighted)",
    )
    p.add_argument("--title", default=None, metavar="TEXT", help="title for the --compare chart")
    return p.parse_args(argv)


def load_runs(path):
    runs = []
    try:
        f = open(path, "r", encoding="utf-8")
    except OSError:
        sys.exit(f"error: cannot open {path}")
    with f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"warning: skipping malformed line {lineno}: {e}", file=sys.stderr)
                continue
            runs.append(rec)
    if not runs:
        sys.exit(f"error: no valid records in {path}")
    return runs


def keep(rec, phases, tags, names):
    if phases and rec.get("phase") not in phases:
        return False
    if tags and rec.get("tag") not in tags:
        return False
    if names and rec.get("name") not in names:
        return False
    return True


def dedupe_keep_latest(runs):
    """Append-only history: last record per (name, tag) wins."""
    latest = {}
    for rec in runs:
        latest[(rec.get("name", ""), rec.get("tag", ""))] = rec
    return list(latest.values())


LATENCY_FIELDS = ["n", "mean", "stddev", "min", "max", "p50", "p90", "p99"]


def get_value(rec, key):
    """Fetch a value by latency field name or metric key. None if missing."""
    lat = rec.get("latency", {})
    if key in lat:
        return lat[key]
    return rec.get("metrics", {}).get(key)


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, int):
        return str(v)
    if v == 0:
        return "0"
    if abs(v) >= 1000 or abs(v) < 0.001:
        return f"{v:.3e}"
    return f"{v:.6g}"


def fmt_delta(new, base, lower_is_better=True):
    """Formatted %delta of new vs base. None-safe."""
    if new is None or base is None or base == 0:
        return "-"
    pct = (new - base) / abs(base) * 100.0
    arrow = ""
    improved = (new < base) if lower_is_better else (new > base)
    if new != base:
        arrow = " (+)" if improved else " (-)"
    return f"{pct:+.2f}%{arrow}"


def main(argv=None):
    args = parse_args(argv)
    runs = [r for r in load_runs(args.results) if keep(r, args.phase, args.tag, args.name)]
    if not runs:
        sys.exit("error: filters matched no records")
    rows = dedupe_keep_latest(runs)

    # Collect metric keys
    metric_keys = []
    for r in rows:
        for k in r.get("metrics", {}):
            if k not in metric_keys:
                metric_keys.append(k)

    # Baseline lookup by tag.
    base_rec = None
    if args.baseline:
        for r in rows:
            if r.get("tag") == args.baseline:
                base_rec = r
        if base_rec is None:
            sys.exit(f"error: baseline tag '{args.baseline}' not found in filtered rows")

    if args.sort:

        def sort_key(r):
            v = get_value(r, args.sort)
            return float("inf") if v is None else v

        rows.sort(key=sort_key)

    # ---- Markdown table -------------------------------------------------------
    header = ["name", "tag", "n", "mean_ms", "p50_ms", "p90_ms", "p99_ms"] + [
        f"{k}*" for k in metric_keys
    ]
    sep = ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]

    for r in rows:
        lat = r.get("latency", {})
        cells = [
            r.get("name", ""),
            r.get("tag", ""),
            str(lat.get("n", "-")),
            fmt(lat.get("mean", 0) * 1e3),  # seconds -> ms
            fmt(lat.get("p50", 0) * 1e3),
            fmt(lat.get("p90", 0) * 1e3),
            fmt(lat.get("p99", 0) * 1e3),
        ] + [fmt(get_value(r, k)) for k in metric_keys]

        if base_rec is not None and r is not base_rec:
            bl = base_rec.get("latency", {})
            cells.append(fmt_delta(lat.get("mean"), bl.get("mean"), lower_is_better=True))
            for k in metric_keys:
                # Metrics are assumed lower-is-better too (BPB, NFEs, FLOPs/byte);
                # throughput-style metrics should be inverted at logging time
                # or read the sign accordingly.
                cells.append(fmt_delta(get_value(r, k), get_value(base_rec, k)))
        elif base_rec is not None:
            cells.extend(["(baseline)"] * (1 + len(metric_keys)))

        lines.append("| " + " | ".join(cells) + " |")

    print("\n".join(lines))
    if metric_keys:
        print()
        print("* metric values (domain metrics: BPB, NFEs, GB/s, ...)")
    if base_rec is not None:
        print()
        print(
            f"delta vs baseline tag '{args.baseline}': "
            "(+) improvement, (-) regression; latency/metrics assumed lower-is-better."
        )

    # ---- Optional scatter plot -----------------------------------------------
    if args.plot and not args.compare:
        if not (args.x and args.y):
            sys.exit("error: --plot requires both --x KEY and --y KEY")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            sys.exit("error: matplotlib not available; cannot write plot")
        xs = [get_value(r, args.x) for r in rows]
        ys = [get_value(r, args.y) for r in rows]
        pts = [(x, y, r) for x, y, r in zip(xs, ys, rows) if x is not None and y is not None]
        if not pts:
            sys.exit(f"error: no rows have both '{args.x}' and '{args.y}'")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter([p[0] for p in pts], [p[1] for p in pts])
        for x, y, r in pts:
            ax.annotate(
                r.get("tag", ""), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points"
            )
        ax.set_xlabel(args.x)
        ax.set_ylabel(args.y)
        ax.set_title("benchmark comparison")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nplot written to {args.plot}")

    # ---- Grouped BPB-vs-throughput bar chart ----------------------------------
    if args.compare:
        render_compare(rows, base_rec, args)


def render_compare(rows, base_rec, args):
    """Grouped bar chart: held-out BPB (left) and throughput (right) for the
    selected tags. Baseline bar is grey, the best-BPB tag is gold, everything
    else steel blue. Requires --compare TAG (repeatable); use --baseline TAG to
    highlight/reference a run and --title TEXT to label the figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("error: matplotlib not available; cannot write chart")

    wanted = list(dict.fromkeys(args.compare))  # dedupe, keep order
    by_tag = {r["tag"]: r for r in rows}
    tags = [t for t in wanted if t in by_tag]
    missing = [t for t in wanted if t not in by_tag]
    if missing:
        print(f"warning: --compare tags not found: {', '.join(missing)}", file=sys.stderr)
    baseline_tag = args.baseline
    if baseline_tag and baseline_tag not in tags and baseline_tag in by_tag:
        tags.insert(0, baseline_tag)
    if len(tags) < 2:
        sys.exit("error: need at least two comparable rows for --compare")

    def metric(tag, key):
        v = get_value(by_tag[tag], key)
        return 0.0 if v is None else float(v)

    bpbs = [metric(t, "bpb") for t in tags]
    thr = [metric(t, "throughput_bytes_per_sec") for t in tags]  # B/s

    best_i = min(range(len(tags)), key=lambda i: bpbs[i])
    colors = []
    for i, t in enumerate(tags):
        if t == baseline_tag:
            colors.append("#888888")
        elif i == best_i:
            colors.append("#e6a817")
        else:
            colors.append("#4878a8")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(7, 0.9 * len(tags)), 4.6))

    bars1 = ax1.bar(tags, bpbs, color=colors)
    ax1.set_ylabel("held-out BPB (lower is better)")
    ax1.set_ylim(min(bpbs) * 0.985, max(bpbs) * 1.005)
    for b, v in zip(bars1, bpbs):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)

    bars2 = ax2.bar(tags, thr, color=colors)
    ax2.set_ylabel("train throughput (bytes/s, higher is better)")
    ax2.set_ylim(0, max(thr) * 1.18)
    for b, v in zip(bars2, thr):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelrotation=35)
        for lbl in ax.get_xticklabels():
            lbl.set_horizontalalignment("right")
            lbl.set_fontsize(8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(args.title or "ablation comparison", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = args.plot or "compare.png"
    fig.savefig(out, dpi=150)
    print(f"\ncomparison chart written to {out}")


if __name__ == "__main__":
    main()
