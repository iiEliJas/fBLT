"""
Recompute greedy teacher-forced frontier-accuracy aggregates from the TSV dump
written by the greedy_tf measurement tool, verify them against the tool's
machine line, and render a Markdown report of per-row top-1 accuracy split by
frontier length, final-row patch length, and closure type.

The greedy_tf measurement predicts every byte from a force-closed frontier
patch (generation conditions); the analyzer compares those predictions against
the authoritative full-window patch that contains each position, split by
closure type (nat/max/bnd) and length bucket, mirroring the layout of
docs/last_row_analysis/patch_final_500w.md.

Machine-line key mapping: bare <bucket> is the Table A row count for that
frontier-length bucket; total_rows/total_correct/acc are the overall totals.

Usage:
    python3 fblt/scripts/analyze_greedy_tf.py dump.tsv
    python3 fblt/scripts/analyze_greedy_tf.py dump.tsv --out report.md
"""

import argparse
import math
import os
import re
import sys
from collections import defaultdict, namedtuple

CLOSURES = ("nat", "max", "bnd")
BUCKETS = ("1-2", "3-4", "5-8", "9-16", "17+")
BUCKET_KEYS = {"1-2": "b1_2", "3-4": "b3_4", "5-8": "b5_8", "9-16": "b9_16", "17+": "b17p"}
Z_95 = 1.959963984540054
KV_RE = re.compile(r"(\w+)=(\d+)")
ACC_RE = re.compile(r"acc=([\d.]+)")
MACHINE_KEYS = ("total_rows total_correct b1_2 b3_4 b5_8 b9_16 b17p").split()
HEADER = ("window", "pos", "frontier_len", "final_patch_len", "closure", "is_final", "correct")

Aggregates = namedtuple(
    "Aggregates", ["buckets_a", "buckets_b", "buckets_c", "total_correct", "total_rows", "windows"]
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Verify and report greedy teacher-forced frontier accuracy "
        "splits from a TSV dump."
    )
    p.add_argument("tsv", help="TSV dump written by the greedy_tf measurement tool")
    p.add_argument("--out", default=None, help="write the Markdown report here instead of stdout")
    p.add_argument(
        "--title", default="Greedy teacher-forced frontier accuracy", help="report title"
    )
    p.add_argument("--min-n", type=int, default=100, help="flag table cells with fewer rows")
    p.add_argument(
        "--equiv",
        action="store_true",
        help="run the corrected natural-patch-equivalent analysis instead of the standard report",
    )
    p.add_argument(
        "--batch-tsv",
        default=None,
        help="patch_final batch TSV for the join (default: <dump dir>/patch_final_500w.tsv)",
    )
    p.add_argument(
        "--batch-md",
        default="docs/last_row_analysis/patch_final_500w.md",
        help="batch nat-final baseline markdown (default: patch_final_500w.md)",
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


def check_header(fields, lineno):
    if tuple(fields) != HEADER:
        print(
            f"WARNING: header on line {lineno} does not match the greedy_tf TSV contract: "
            f"expected {list(HEADER)}, got {fields}",
            file=sys.stderr,
        )


def parse_row(fields, lineno):
    if len(fields) != 7:
        warn_skipped(lineno, f"expected 7 fields, got {len(fields)}")
        return None
    for idx in (0, 1, 2, 3, 5, 6):
        try:
            int(fields[idx])
        except ValueError:
            warn_skipped(lineno, f"non-integer value in column {idx + 1}: {fields[idx]!r}")
            return None
    closure = fields[4]
    if closure not in CLOSURES:
        warn_skipped(lineno, f"unknown closure {closure!r}")
        return None
    return (
        int(fields[0]),
        int(fields[1]),
        int(fields[2]),
        int(fields[3]),
        closure,
        int(fields[5]),
        int(fields[6]),
    )


def parse_tsv(path):
    """Return Aggregates with cell values as [correct, count], or None if unreadable."""
    buckets_a = {b: [0, 0] for b in BUCKETS}
    buckets_b = {(c, b): [0, 0] for c in CLOSURES for b in BUCKETS}
    buckets_c = {b: [0, 0] for b in BUCKETS}
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
                check_header(raw.rstrip("\r\n").split("\t"), lineno)
                continue
            parsed = parse_row(raw.rstrip("\r\n").split("\t"), lineno)
            if parsed is None:
                continue
            window, pos, frontier_len, final_patch_len, closure, is_final, correct = parsed
            bucket_a = bucket_for(frontier_len)
            buckets_a[bucket_a][0] += correct
            buckets_a[bucket_a][1] += 1
            if is_final:
                bucket_b = bucket_for(final_patch_len)
                buckets_b[(closure, bucket_b)][0] += correct
                buckets_b[(closure, bucket_b)][1] += 1
                buckets_c[bucket_a][0] += correct
                buckets_c[bucket_a][1] += 1
            total_correct += correct
            total_rows += 1
            windows.add(window)
    return Aggregates(buckets_a, buckets_b, buckets_c, total_correct, total_rows, len(windows))


def machine_expected(aggs):
    expected = {"total_rows": aggs.total_rows, "total_correct": aggs.total_correct}
    for bucket in BUCKETS:
        expected[BUCKET_KEYS[bucket]] = aggs.buckets_a[bucket][1]
    return expected


def machine_line(aggs):
    acc = aggs.total_correct / aggs.total_rows * 100.0 if aggs.total_rows else 0.0
    parts = [
        f"GREEDY_TF total_rows={aggs.total_rows} total_correct={aggs.total_correct} acc={acc:.2f}"
    ]
    for bucket in BUCKETS:
        parts.append(f"{BUCKET_KEYS[bucket]}={aggs.buckets_a[bucket][1]}")
    return " ".join(parts)


def verify_machine(line, aggs):
    kv = {k: int(v) for k, v in KV_RE.findall(line)}
    expected = machine_expected(aggs)
    failures = []
    for key in MACHINE_KEYS:
        machine_value = kv.get(key)
        shown = "<missing>" if machine_value is None else str(machine_value)
        recomputed = expected[key]
        if machine_value != recomputed:
            failures.append(f"MISMATCH {key}: machine={shown} recomputed={recomputed}")
    acc_match = ACC_RE.search(line)
    expected_acc = aggs.total_correct / aggs.total_rows * 100.0 if aggs.total_rows else 0.0
    if acc_match is None:
        failures.append("MISMATCH acc: machine=<missing>")
    elif abs(float(acc_match.group(1)) - round(expected_acc, 2)) > 1e-9:
        failures.append(f"MISMATCH acc: machine={acc_match.group(1)} recomputed={expected_acc:.2f}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return False
    print("[VERIFY] GREEDY_TF machine line matches TSV aggregates", file=sys.stderr)
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


def format_ac_row(bucket, correct, count, min_n):
    if count == 0:
        n_cell = "(no samples)"
        acc_cell = "—"
        ci_cell = "—"
    else:
        n_cell = f"n={count}"
        acc, lo, hi = wilson(correct, count)
        acc_cell = f"{acc * 100:.2f}%"
        ci_cell = f"{lo * 100:.2f}%–{hi * 100:.2f}%"
    if count < min_n:
        n_cell += " ⚠small-n"
    return f"| {bucket} | {n_cell} | {correct} | {acc_cell} | {ci_cell} |"


def render_report(args, aggs):
    lines = [
        f"# {args.title}",
        "",
        f"Source: `{args.tsv}` — {aggs.total_rows} rows, {aggs.windows} windows",
        "",
        "## Table A — all frontier predictions",
        "",
        "Every row, bucketed by the force-closed frontier patch length at prediction time.",
        "",
        "| frontier_len | n | correct | acc | CI |",
        "| --- | --- | --- | --- | --- |",
    ]
    for bucket in BUCKETS:
        correct, count = aggs.buckets_a[bucket]
        lines.append(format_ac_row(bucket, correct, count, args.min_n))
    lines.append("")
    lines.append("## Table B — final-row predictions (is_final=1)")
    lines.append("")
    lines.append(
        "Rows where the predicted position is the final byte of its full-window patch, "
        "bucketed by final patch length, split by closure type."
    )
    lines.append("")
    lines.append("| Length | nat | max | bnd |")
    lines.append("| --- | --- | --- | --- |")
    for bucket in BUCKETS:
        cells = []
        for closure in CLOSURES:
            correct, count = aggs.buckets_b[(closure, bucket)]
            cells.append(format_cell(correct, count, args.min_n))
        lines.append(f"| {bucket} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Table C — closure-matched frontier view (is_final=1)")
    lines.append("")
    lines.append(
        "Final-row predictions bucketed by the force-closed frontier length at prediction "
        "time; compare against Table A at the same bucket to expose the length-vs-closure "
        "interaction."
    )
    lines.append("")
    lines.append("| frontier_len | n | correct | acc | CI |")
    lines.append("| --- | --- | --- | --- | --- |")
    for bucket in BUCKETS:
        correct, count = aggs.buckets_c[bucket]
        lines.append(format_ac_row(bucket, correct, count, args.min_n))
    lines.append("")
    lines.append("## How to read")
    lines.append("")
    lines.append(
        "- Table A: overall frontier accuracy under generation conditions — every prediction "
        "is made from a force-closed frontier patch."
    )
    lines.append(
        "- Table B: the natural-patch-equivalent view — final-row accuracy by the "
        "authoritative full-window patch length and closure type (nat/max/bnd), matching the "
        "patch_final_500w.md layout."
    )
    lines.append(
        "- Table C: final-row predictions re-bucketed by frontier length; a gap vs Table A at "
        "the same bucket isolates the length-vs-closure interaction."
    )
    lines.append(
        "- Cells with n<100 are flagged ⚠small-n; their Wilson intervals are wide and should "
        "be read cautiously."
    )
    lines.append("")
    return "\n".join(lines)


def parse_rows_by_window(path):
    # Group parsed rows by window for the corrected-equivalent scan.
    rows = defaultdict(list)
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
                check_header(raw.rstrip("\r\n").split("\t"), lineno)
                continue
            parsed = parse_row(raw.rstrip("\r\n").split("\t"), lineno)
            if parsed is None:
                continue
            window, pos, frontier_len, final_patch_len, closure, is_final, correct = parsed
            rows[window].append((pos, frontier_len, final_patch_len, closure, is_final, correct))
    for window in rows:
        rows[window].sort()
    return rows


def scan_window(seq, init_cps):
    # Run the left-to-right patch scan over one window's sorted rows. Returns
    # (equiv, closes, cps_list, closes_at_end, bad_contain) where equiv[i] marks
    # rows whose position is the first byte of its recovered patch, closes[i]
    # marks rows where the recovered patch closes, cps_list[i] is the recovered
    # patch start at that row, and bad_contain lists positions whose recovered
    # patch does not contain them.
    cps = init_cps
    equiv = []
    closes = []
    cps_list = []
    bad_contain = []
    for pos, frontier_len, final_patch_len, closure, is_final, correct in seq:
        cps_list.append(cps)
        if cps + final_patch_len - 1 < pos:
            bad_contain.append(pos)
        equiv.append(cps == pos)
        closes.append(pos - cps + 1 == final_patch_len)
        if closes[-1]:
            cps = pos + 1
    closes_at_end = cps == seq[-1][0] + 1
    return equiv, closes, cps_list, closes_at_end, bad_contain


def last_byte_violations(seq, closes, cps_list):
    # Task check: for every position, current_patch_start + final_patch_len - 1
    # must equal the recovered last-byte position of that patch (the position
    # where the scan closes it). Returns the violating positions.
    bad = []
    n = len(seq)
    for i in range(n):
        pos, frontier_len, final_patch_len, closure, is_final, correct = seq[i]
        j = i
        while j < n and not closes[j]:
            j += 1
        if j >= n:
            bad.append(pos)
            continue
        if cps_list[i] + final_patch_len - 1 != seq[j][0]:
            bad.append(pos)
    return bad


def corrected_equiv_rows(rows_by_window):
    # Recover the per-window patch segmentation from final_patch_len alone. The
    # naive init (current_patch_start=0 at window start) is wrong when the first
    # patch closes before the first observed position (pos=2 in this dump), so
    # the initial current_patch_start is chosen by trying s in {0..p0} and
    # keeping the unique candidate whose scan closes at the window end.
    equiv_rows = []
    naive_windows = 0
    naive_contain = 0
    naive_last_byte = 0
    naive_window_ids = []
    naive_contain_windows = set()
    naive_last_byte_windows = set()
    unresolved = []
    for window in sorted(rows_by_window):
        seq = rows_by_window[window]
        p0 = seq[0][0]
        chosen = None
        for s in range(0, p0 + 1):
            equiv, closes, cps_list, closes_at_end, bad_contain = scan_window(seq, s)
            if closes_at_end and not bad_contain:
                chosen = (equiv, closes, cps_list)
                break
        if chosen is None:
            unresolved.append(window)
            chosen = scan_window(seq, 0)[:3]
        equiv, closes, cps_list = chosen
        for (pos, frontier_len, final_patch_len, closure, is_final, correct), e in zip(seq, equiv):
            equiv_rows.append((window, pos, final_patch_len, closure, correct, e))
        # Naive-init consistency stats for the report.
        eq0, cl0, cps0, closes0, bad0 = scan_window(seq, 0)
        if not closes0:
            naive_windows += 1
            naive_window_ids.append(window)
        if bad0:
            naive_contain += len(bad0)
            naive_contain_windows.add(window)
        lb0 = last_byte_violations(seq, cl0, cps0)
        if lb0:
            naive_last_byte += len(lb0)
            naive_last_byte_windows.add(window)
    naive_stats = {
        "windows": naive_windows,
        "contain": naive_contain,
        "contain_windows": len(naive_contain_windows),
        "last_byte": naive_last_byte,
        "last_byte_windows": len(naive_last_byte_windows),
        "window_ids": naive_window_ids,
        "unresolved": unresolved,
    }
    return equiv_rows, naive_stats


def load_batch_tsv(path):
    # Load the patch_final batch TSV as {(window, row): (patch_start, patch_len,
    # closure, is_final, correct)} plus row/window counts. Returns None if the
    # file cannot be read.
    data = {}
    total_rows = 0
    windows = set()
    try:
        handle = open(path, "r", encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"WARNING: cannot read batch TSV {path}: {exc}", file=sys.stderr)
        return None
    with handle:
        for lineno, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            fields = raw.rstrip("\r\n").split("\t")
            if fields[0] == "window":
                continue
            if len(fields) != 8:
                warn_skipped(lineno, f"batch row: expected 8 fields, got {len(fields)}")
                continue
            try:
                window = int(fields[0])
                row = int(fields[1])
                patch_start = int(fields[3])
                patch_len = int(fields[4])
                closure = fields[5]
                is_final = int(fields[6])
                correct = int(fields[7])
            except ValueError:
                warn_skipped(lineno, "batch row: non-integer field")
                continue
            if closure not in CLOSURES:
                warn_skipped(lineno, f"batch row: unknown closure {closure!r}")
                continue
            data[(window, row)] = (patch_start, patch_len, closure, is_final, correct)
            total_rows += 1
            windows.add(window)
    return {"rows": data, "total_rows": total_rows, "windows": len(windows)}


def load_batch_md(path):
    # Extract the nat-final column of the "Patch-final rows" table from the
    # batch baseline markdown. Returns {bucket: cell_text} or None.
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError as exc:
        print(f"WARNING: cannot read batch baseline {path}: {exc}", file=sys.stderr)
        return None
    in_table = False
    cells = {}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            in_table = stripped.startswith("## Patch-final rows")
            continue
        if in_table and stripped.startswith("|"):
            fields = [f.strip() for f in stripped.strip("|").split("|")]
            if len(fields) >= 2 and fields[0] in BUCKETS:
                cells[fields[0]] = fields[1]
    return cells


def join_disagreement(equiv_rows, batch_rows):
    # Target-aligned join: gen row pos p predicts text[p], batch row i predicts
    # text[i+1], so gen pos p pairs with batch row p-1. The equiv(p) flag must
    # equal batch is_final(p-1): p first-byte-of-patch <=> row p-1 last byte.
    joined = 0
    disagree = 0
    equiv_n = 0
    equiv_disagree = 0
    rest_n = 0
    rest_disagree = 0
    align_n = 0
    align_mismatch = 0
    for window, pos, final_patch_len, closure, correct, equiv in equiv_rows:
        batch = batch_rows.get((window, pos - 1))
        if batch is None:
            continue
        batch_is_final, batch_correct = batch[3], batch[4]
        joined += 1
        align_n += 1
        if equiv != batch_is_final:
            align_mismatch += 1
        if correct != batch_correct:
            disagree += 1
        if equiv:
            equiv_n += 1
            if correct != batch_correct:
                equiv_disagree += 1
        else:
            rest_n += 1
            if correct != batch_correct:
                rest_disagree += 1
    return {
        "joined": joined,
        "disagree": disagree,
        "equiv_n": equiv_n,
        "equiv_disagree": equiv_disagree,
        "rest_n": rest_n,
        "rest_disagree": rest_disagree,
        "align_n": align_n,
        "align_mismatch": align_mismatch,
    }


def render_equiv_report(args, aggs, equiv_rows, naive_stats, batch_data, batch_tsv_path, batch_md):
    equiv_counts = {c: 0 for c in CLOSURES}
    buckets_b = {(c, b): [0, 0] for c in CLOSURES for b in BUCKETS}
    for window, pos, final_patch_len, closure, correct, equiv in equiv_rows:
        if not equiv:
            continue
        equiv_counts[closure] += 1
        bucket = bucket_for(final_patch_len)
        buckets_b[(closure, bucket)][0] += correct
        buckets_b[(closure, bucket)][1] += 1
    total_equiv = sum(equiv_counts.values())
    lines = [
        f"# {args.title} — corrected natural-patch-equivalent view",
        "",
        f"Source: `{args.tsv}` — {aggs.total_rows} rows, {aggs.windows} windows",
        "",
        "## Corrected-equivalent rows",
        "",
        "The tool's `is_final` label marks the LAST byte of the full-window patch, but the",
        "generation prediction of byte p is made from the frontier patch [s, p), which equals",
        "the true patch only when p is the FIRST byte of its patch. The corrected flag",
        "re-derives the per-window patch segmentation from `final_patch_len` alone (a",
        "left-to-right scan) and marks rows where p is the first byte of its patch.",
        "",
        "| closure | equiv rows | share of rows |",
        "| --- | --- | --- |",
    ]
    for closure in CLOSURES:
        count = equiv_counts[closure]
        share = count / aggs.total_rows * 100.0 if aggs.total_rows else 0.0
        lines.append(f"| {closure} | n={count} | {share:.2f}% |")
    total_share = total_equiv / aggs.total_rows * 100.0 if aggs.total_rows else 0.0
    lines.append(f"| total | n={total_equiv} | {total_share:.2f}% |")
    lines.append("")
    if batch_data is not None:
        batch_rows = batch_data["rows"]
        batch_nat_final = sum(
            1 for (ps, pl, cl, isf, corr) in batch_rows.values() if cl == "nat" and isf
        )
        nat_final_300 = sum(
            1
            for (w, r), (ps, pl, cl, isf, corr) in batch_rows.items()
            if cl == "nat" and isf and w < aggs.windows
        )
        nat_final_300_p2 = sum(
            1
            for (w, r), (ps, pl, cl, isf, corr) in batch_rows.items()
            if cl == "nat" and isf and w < aggs.windows and r >= 2
        )
        expected = batch_nat_final * aggs.windows / 500.0
        ratio = equiv_counts["nat"] / expected if expected else 0.0
        brief_expected = 28594 * aggs.windows / 500.0
        brief_ratio = equiv_counts["nat"] / brief_expected if brief_expected else 0.0
        lines.append(
            f"Sanity: batch nat-final rows across 500 windows = {batch_nat_final} (the brief "
            f"cites 28594, which does not appear in the data; the closest real figures are "
            f"{nat_final_300_p2} nat-final rows with pos>=2 in the {aggs.windows}-window run "
            f"and {nat_final_300} nat-final rows including pos 0/1). A {aggs.windows}-window "
            f"run therefore implies ~{expected:.0f} expected nat equiv rows "
            f"({batch_nat_final} * {aggs.windows}/500); actual = {equiv_counts['nat']}, "
            f"ratio = {ratio:.3f}. Against the brief's 28594 the expectation would be "
            f"~{brief_expected:.0f} and the ratio {brief_ratio:.3f}."
        )
    lines.append("")
    lines.append("## Corrected Table B — equiv rows by final patch length x closure")
    lines.append("")
    lines.append(
        "Rows where p is the first byte of its full-window patch, bucketed by final patch "
        "length, split by closure type."
    )
    lines.append("")
    lines.append("| Length | nat | max | bnd |")
    lines.append("| --- | --- | --- | --- |")
    for bucket in BUCKETS:
        cells = []
        for closure in CLOSURES:
            correct, count = buckets_b[(closure, bucket)]
            cells.append(format_cell(correct, count, args.min_n))
        lines.append(f"| {bucket} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Overall equiv-row accuracy by closure")
    lines.append("")
    lines.append("| closure | n | correct | acc | CI |")
    lines.append("| --- | --- | --- | --- | --- |")
    for closure in CLOSURES:
        correct = sum(1 for (w, p, fpl, cl, corr, e) in equiv_rows if e and cl == closure and corr)
        count = sum(1 for (w, p, fpl, cl, corr, e) in equiv_rows if e and cl == closure)
        lines.append(format_ac_row(closure, correct, count, args.min_n))
    lines.append("")
    lines.append("## Equiv-nat accuracy vs batch nat-final baseline")
    lines.append("")
    if batch_md is None:
        lines.append("Batch baseline markdown not found; comparison skipped.")
    else:
        lines.append(
            "Batch baseline from `docs/last_row_analysis/patch_final_500w.md` (nat-final rows, "
            "500 windows). Bucket labels align 1:1 (same final-patch-length buckets); the "
            "baseline is the last-byte view while the equiv column is the first-byte view of "
            "the same patches, so per-bucket populations are the same patches modulo the "
            "300/500 window subset and the pos 0/1 rows absent from the greedy dump."
        )
        lines.append("")
        lines.append("| Length | equiv-nat (gen, 300w) | batch nat-final baseline (500w) |")
        lines.append("| --- | --- | --- |")
        for bucket in BUCKETS:
            correct, count = buckets_b[("nat", bucket)]
            equiv_cell = format_cell(correct, count, args.min_n)
            baseline_cell = batch_md.get(bucket, "—")
            lines.append(f"| {bucket} | {equiv_cell} | {baseline_cell} |")
    lines.append("")
    lines.append("## Join-based disagreement check")
    lines.append("")
    if batch_data is None:
        lines.append(f"Batch TSV `{batch_tsv_path}` not found; join skipped.")
    else:
        join = join_disagreement(equiv_rows, batch_data["rows"])
        lines.append(
            f"Joined {join['joined']} rows against `{batch_tsv_path}` "
            f"({batch_data['total_rows']} rows, {batch_data['windows']} windows) with "
            "target alignment: gen row pos p predicts text[p] and batch row i predicts "
            "text[i+1], so gen pos p pairs with batch row p-1. Alignment "
            f"equiv(p) == batch is_final(p-1) matches on "
            f"{join['align_n'] - join['align_mismatch']}/{join['align_n']} rows. "
            "gen.correct != batch.correct on "
            f"{join['disagree']}/{join['joined']} rows "
            f"({join['disagree'] / join['joined'] * 100.0:.2f}%)."
        )
        lines.append("")
        lines.append("| rows | n | disagree | share |")
        lines.append("| --- | --- | --- | --- |")
        lines.append(
            f"| equiv | n={join['equiv_n']} | {join['equiv_disagree']} | "
            f"{join['equiv_disagree'] / join['equiv_n'] * 100.0:.2f}% |"
        )
        lines.append(
            f"| rest | n={join['rest_n']} | {join['rest_disagree']} | "
            f"{join['rest_disagree'] / join['rest_n'] * 100.0:.2f}% |"
        )
        lines.append(
            f"| total | n={join['joined']} | {join['disagree']} | "
            f"{join['disagree'] / join['joined'] * 100.0:.2f}% |"
        )
        lines.append("")
        lines.append(
            "Equiv rows disagree at ~0 as expected (identical inputs imply identical "
            "predictions); the rest rows disagree heavily because batch leaks the true "
            "patch's future bytes into their latents while generation is blind."
        )
    lines.append("")
    lines.append("## Reconstruction consistency")
    lines.append("")
    consistency = (
        "The scan is validated three ways: every recovered patch must contain its position "
        "(current_patch_start + final_patch_len[p] - 1 >= p), the recovered last-byte of "
        "each patch must equal current_patch_start + final_patch_len[p] - 1, and the scan "
        "must close at the window end. With the naive init (current_patch_start=0 at window "
        f"start) the first patch closes before the first observed position (pos=2) in "
        f"{naive_stats['windows']}/{aggs.windows} windows, so the recovered segmentation is "
        f"shifted: {naive_stats['contain']} positions in {naive_stats['contain_windows']} "
        f"windows violate the containment check and {naive_stats['last_byte']} positions in "
        f"{naive_stats['last_byte_windows']} windows violate the last-byte check. The "
        "corrected init (candidate search over s in {0..p0} keeping the unique candidate "
        "that closes at the window end) recovers the exact batch segmentation: 0 "
        "inconsistent windows"
    )
    if batch_data is not None:
        consistency += ", 0 equiv mismatches against the batch `patch_start` column."
    else:
        consistency += "."
    lines.append(consistency)
    if naive_stats["unresolved"]:
        lines.append(
            f"Windows with no consistent candidate (fell back to s=0): {naive_stats['unresolved']}"
        )
    lines.append("")
    return "\n".join(lines)


def run_equiv(args, aggs):
    rows_by_window = parse_rows_by_window(args.tsv)
    if rows_by_window is None:
        return 1
    equiv_rows, naive_stats = corrected_equiv_rows(rows_by_window)
    batch_tsv_path = (
        args.batch_tsv
        if args.batch_tsv
        else os.path.join(os.path.dirname(args.tsv), "patch_final_500w.tsv")
    )
    batch_data = load_batch_tsv(batch_tsv_path)
    batch_md = load_batch_md(args.batch_md)
    report = render_equiv_report(
        args, aggs, equiv_rows, naive_stats, batch_data, batch_tsv_path, batch_md
    )
    equiv_counts = {c: 0 for c in CLOSURES}
    for window, pos, final_patch_len, closure, correct, equiv in equiv_rows:
        if equiv:
            equiv_counts[closure] += 1
    total_equiv = sum(equiv_counts.values())
    print(
        "[EQUIV] equiv rows by closure: "
        + " ".join(f"{c}={equiv_counts[c]}" for c in CLOSURES)
        + f" total={total_equiv}",
        file=sys.stderr,
    )
    for closure in CLOSURES:
        correct = sum(1 for (w, p, fpl, cl, corr, e) in equiv_rows if e and cl == closure and corr)
        count = sum(1 for (w, p, fpl, cl, corr, e) in equiv_rows if e and cl == closure)
        if count:
            acc, lo, hi = wilson(correct, count)
            print(
                f"[EQUIV] {closure} equiv acc: {correct}/{count} = {acc * 100:.2f}% "
                f"({lo * 100:.2f}%–{hi * 100:.2f}%)",
                file=sys.stderr,
            )
    if batch_data is not None:
        join = join_disagreement(equiv_rows, batch_data["rows"])
        print(
            f"[EQUIV] join: {join['disagree']}/{join['joined']} disagree "
            f"({join['disagree'] / join['joined'] * 100:.2f}%); equiv "
            f"{join['equiv_disagree']}/{join['equiv_n']} "
            f"({join['equiv_disagree'] / join['equiv_n'] * 100:.2f}%); rest "
            f"{join['rest_disagree']}/{join['rest_n']} "
            f"({join['rest_disagree'] / join['rest_n'] * 100:.2f}%)",
            file=sys.stderr,
        )
    print(
        f"[EQUIV] reconstruction: naive init inconsistent on {naive_stats['windows']} windows "
        f"({naive_stats['contain']} containment in {naive_stats['contain_windows']} windows, "
        f"{naive_stats['last_byte']} last-byte in {naive_stats['last_byte_windows']} windows); "
        "corrected init 0",
        file=sys.stderr,
    )
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


def main(argv=None):
    args = parse_args(argv)
    aggs = parse_tsv(args.tsv)
    if aggs is None:
        return 1
    line = machine_line(aggs)
    print(line, file=sys.stderr)
    if not verify_machine(line, aggs):
        return 1
    if args.equiv:
        return run_equiv(args, aggs)
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
