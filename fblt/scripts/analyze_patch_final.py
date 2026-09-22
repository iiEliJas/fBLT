"""
Recompute patch-final bucket aggregates from the TSV dump written by
build/patch_final_split, verify them against the tool's machine line, and
render a Markdown report of per-row top-1 accuracy split by patch
finality, closure type, and patch-length bucket.

Machine-line key mapping: bare <finality>_<closure> is the row count,
the _c suffix is the correct count (final_nat=count, final_nat_c=correct).

Usage:
    python3 fblt/scripts/analyze_patch_final.py --tsv dump.tsv
    python3 fblt/scripts/analyze_patch_final.py --tsv dump.tsv \
        --machine machine.txt --out report.md
"""

import argparse
import math
import re
import sys
from collections import namedtuple

CLOSURES = ("nat", "max", "bnd")
FINALITIES = ("final", "nonfinal")
BUCKETS = ("1-2", "3-4", "5-8", "9-16", "17+")
Z_95 = 1.959963984540054
KV_RE = re.compile(r"(\w+)=(\d+)")
MACHINE_KEYS = (
    "final_nat final_max final_bnd final_nat_c final_max_c final_bnd_c"
    " nonfinal_nat nonfinal_max nonfinal_bnd nonfinal_nat_c nonfinal_max_c nonfinal_bnd_c"
    " total_correct total_rows windows"
).split()

Aggregates = namedtuple(
    "Aggregates", ["cells", "buckets", "total_correct", "total_rows", "windows"]
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Verify and report patch-final accuracy splits from a TSV dump."
    )
    p.add_argument("--tsv", required=True, help="TSV dump written by build/patch_final_split")
    p.add_argument("--machine", default=None, help="file containing the C tool's machine line")
    p.add_argument("--out", default=None, help="write the Markdown report here instead of stdout")
    p.add_argument("--title", default="Patch-final row accuracy", help="report title")
    p.add_argument("--min-n", type=int, default=100, help="flag table cells with fewer rows")
    p.add_argument(
        "--gap-threshold",
        type=float,
        default=5.0,
        help="flag closures whose final-vs-nonfinal gap reaches this many percentage points",
    )
    return p.parse_args(argv)


def bucket_for(patch_len):
    if patch_len <= 2:
        return "1-2"
    if patch_len <= 4:
        return "3-4"
    if patch_len <= 8:
        return "5-8"
    if patch_len <= 16:
        return "9-16"
    return "17+"


def warn_skipped(lineno, reason):
    print(f"WARNING: skipping malformed line {lineno}: {reason}", file=sys.stderr)


def parse_row(fields, lineno):
    if len(fields) != 8:
        warn_skipped(lineno, f"expected 8 fields, got {len(fields)}")
        return None
    for idx in (0, 1, 2, 3, 4, 6, 7):
        try:
            int(fields[idx])
        except ValueError:
            warn_skipped(lineno, f"non-integer value in column {idx + 1}: {fields[idx]!r}")
            return None
    closure = fields[5]
    if closure not in CLOSURES:
        warn_skipped(lineno, f"unknown closure {closure!r}")
        return None
    return int(fields[0]), int(fields[4]), closure, int(fields[6]), int(fields[7])


def parse_tsv(path):
    """Return Aggregates with cell values as [correct, count], or None if unreadable."""
    cells = {(f, c): [0, 0] for f in FINALITIES for c in CLOSURES}
    buckets = {(f, c, b): [0, 0] for f in FINALITIES for c in CLOSURES for b in BUCKETS}
    windows = set()
    total_correct = 0
    total_rows = 0
    try:
        handle = open(path, "r", encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"ERROR: cannot read TSV {path}: {exc}", file=sys.stderr)
        return None
    with handle:
        seen_header = False
        for lineno, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            if not seen_header:
                seen_header = True
                continue
            parsed = parse_row(raw.rstrip("\r\n").split("\t"), lineno)
            if parsed is None:
                continue
            window, patch_len, closure, is_final, correct = parsed
            finality = "final" if is_final else "nonfinal"
            bucket = bucket_for(patch_len)
            cells[(finality, closure)][0] += correct
            cells[(finality, closure)][1] += 1
            buckets[(finality, closure, bucket)][0] += correct
            buckets[(finality, closure, bucket)][1] += 1
            total_correct += correct
            total_rows += 1
            windows.add(window)
    return Aggregates(cells, buckets, total_correct, total_rows, len(windows))


def machine_expected(aggs):
    expected = {}
    for finality in FINALITIES:
        for closure in CLOSURES:
            correct, count = aggs.cells[(finality, closure)]
            key = f"{finality}_{closure}"
            expected[key] = count
            expected[f"{key}_c"] = correct
    expected["total_correct"] = aggs.total_correct
    expected["total_rows"] = aggs.total_rows
    expected["windows"] = aggs.windows
    return expected


def load_machine_kv(path):
    """Return {key: int} parsed from the first line containing final_nat=, or None."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "final_nat=" in line:
                    return {k: int(v) for k, v in KV_RE.findall(line)}
    except OSError as exc:
        print(f"ERROR: cannot read machine file {path}: {exc}", file=sys.stderr)
        return None
    print(f"ERROR: no line containing 'final_nat=' found in {path}", file=sys.stderr)
    return None


def verify_machine(kv, expected):
    failures = []
    for key in MACHINE_KEYS:
        machine_value = kv.get(key)
        shown = "<missing>" if machine_value is None else str(machine_value)
        recomputed = expected[key]
        if machine_value != recomputed:
            failures.append(f"MISMATCH {key}: machine={shown} recomputed={recomputed}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return False
    print("[VERIFY] machine line matches TSV aggregates", file=sys.stderr)
    return True


def wilson(correct, count):
    """Return (acc, lo, hi) as fractions using the 95% Wilson score interval."""
    phat = correct / count
    denom = 1.0 + Z_95 * Z_95 / count
    center = (phat + Z_95 * Z_95 / (2.0 * count)) / denom
    variance = phat * (1.0 - phat) / count + Z_95 * Z_95 / (4.0 * count * count)
    half = Z_95 * math.sqrt(variance) / denom
    return phat, max(0.0, center - half), min(1.0, center + half)


def format_acc(correct, count):
    if count == 0:
        return "acc=— (—)"
    acc, lo, hi = wilson(correct, count)
    return f"acc={acc * 100:.2f}% ({lo * 100:.2f}%–{hi * 100:.2f}%)"


def format_cell(correct, count, min_n):
    if count == 0:
        text = "(no samples)"
    else:
        text = f"n={count} {format_acc(correct, count)}"
    if count < min_n:
        text += " ⚠small-n"
    return text


def format_summary(closure, cells, gap_threshold):
    final_correct, final_count = cells[("final", closure)]
    nonfinal_correct, nonfinal_count = cells[("nonfinal", closure)]
    final_part = f"final n={final_count} {format_acc(final_correct, final_count)}"
    nonfinal_part = f"nonfinal n={nonfinal_count} {format_acc(nonfinal_correct, nonfinal_count)}"
    if final_count == 0 or nonfinal_count == 0:
        gap_part = "gap=—"
    else:
        final_acc = wilson(final_correct, final_count)[0]
        nonfinal_acc = wilson(nonfinal_correct, nonfinal_count)[0]
        gap = (nonfinal_acc - final_acc) * 100.0
        gap_part = f"gap={gap:.2f}pp"
        if abs(gap) >= gap_threshold:
            gap_part += " ⚠gap"
    return f"{closure}: {final_part} | {nonfinal_part} | {gap_part}"


def render_report(args, aggs):
    headings = {
        "final": "Patch-final rows (final byte of patch)",
        "nonfinal": "Non-final rows",
    }
    lines = [
        f"# {args.title}",
        "",
        f"Source: `{args.tsv}` — {aggs.total_rows} rows, {aggs.windows} windows",
        "",
    ]
    for finality in FINALITIES:
        lines.append(f"## {headings[finality]}")
        lines.append("")
        lines.append("| Length | nat | max | bnd |")
        lines.append("| --- | --- | --- | --- |")
        for bucket in BUCKETS:
            cells = []
            for closure in CLOSURES:
                correct, count = aggs.buckets[(finality, closure, bucket)]
                cells.append(format_cell(correct, count, args.min_n))
            lines.append(f"| {bucket} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("## Summary")
    lines.append("")
    for closure in CLOSURES:
        lines.append(format_summary(closure, aggs.cells, args.gap_threshold))
    lines.append("")
    lines.append("## Small-n caveats")
    lines.append("")
    caveats = []
    for finality in FINALITIES:
        for closure in CLOSURES:
            count = aggs.cells[(finality, closure)][1]
            if count < args.min_n:
                caveats.append(f"- {finality}/{closure}: n={count}")
    if caveats:
        lines.append(f"Cells below n={args.min_n} are unstable and flagged ⚠small-n above:")
        lines.append("")
        lines.extend(caveats)
    else:
        lines.append(f"- none (all cells n>={args.min_n})")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    aggs = parse_tsv(args.tsv)
    if aggs is None:
        return 1
    if args.machine:
        kv = load_machine_kv(args.machine)
        if kv is None:
            return 1
        if not verify_machine(kv, machine_expected(aggs)):
            return 1
    report = render_report(args, aggs)
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as handle:
                handle.write(report)
        except OSError as exc:
            print(f"ERROR: cannot write report {args.out}: {exc}", file=sys.stderr)
            return 1
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
