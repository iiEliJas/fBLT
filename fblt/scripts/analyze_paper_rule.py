"""
Recompute paper-rule decoder cross-attention aggregates from the TSV dump
written by build/eval_paper_rule_split, verify them against the summed shard
machine lines, join the dump against the patch_final baseline on (window, row)
to enforce the no_prev correctness invariant (final-row flips are informational
mixing, not construction errors), compare headline non-final accuracy, and
render a Markdown report.

The headline set is is_final==0 AND no_prev==0 (the paper-rule non-final
aggregate): final bytes keep group j and first-patch non-final bytes keep
group 0 under both conventions, so they are excluded or compared directly.

Machine-line key mapping: bare <finality>_<closure> is the row count, the _c
suffix is the correct count; no_prev counts first-patch non-final rows.

Usage:
    python3 fblt/scripts/analyze_paper_rule.py --tsv dump.tsv \
        --machine machine.txt --out report.md
"""

import argparse
import math
import re
import sys

CLOSURES = ("nat", "max", "bnd")
FINALITIES = ("final", "nonfinal")
BUCKETS = ("1-2", "3-4", "5-8", "9-16", "17+")
POS_BUCKETS = ("0", "1", "2", "3", "4-7", "8-15", "16+")
Z_95 = 1.959963984540054
KV_RE = re.compile(r"(\w+)=(\d+)")
HEADER = (
    "window",
    "row",
    "patch_idx",
    "patch_start",
    "patch_len",
    "closure",
    "is_final",
    "pos_in_patch",
    "no_prev",
    "correct",
)
BASELINE_HEADER = (
    "window",
    "row",
    "patch_idx",
    "patch_start",
    "patch_len",
    "closure",
    "is_final",
    "correct",
)
MACHINE_KEYS = (
    "windows total_rows total_correct"
    " final_nat final_nat_c final_max final_max_c final_bnd final_bnd_c"
    " nonfinal_nat nonfinal_nat_c nonfinal_max nonfinal_max_c nonfinal_bnd nonfinal_bnd_c"
    " no_prev no_prev_c"
).split()
EXPECTED_ROWS = 256000
PUBLISHED_NAT_NONFINAL_ROWS = 198546
PUBLISHED_NAT_NONFINAL_ACC = 98.54


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Verify and report paper-rule decoder cross-attention accuracy."
    )
    p.add_argument("--tsv", required=True, help="TSV dump written by build/eval_paper_rule_split")
    p.add_argument("--machine", default=None, help="file containing the shard machine lines")
    p.add_argument(
        "--baseline-tsv",
        default="runs/tinystories_p7/analyses/patch_final_500w.tsv",
        help="patch_final baseline TSV with the repo-rule group ids",
    )
    p.add_argument("--out", default=None, help="write the Markdown report here instead of stdout")
    p.add_argument(
        "--title",
        default="Paper-rule decoder cross-attention eval",
        help="report title",
    )
    p.add_argument("--min-n", type=int, default=100, help="flag table cells with fewer rows")
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


def pos_bucket_for(pos):
    if pos <= 3:
        return str(pos)
    if pos <= 7:
        return "4-7"
    if pos <= 15:
        return "8-15"
    return "16+"


def warn_skipped(lineno, reason):
    print(f"WARNING: skipping malformed line {lineno}: {reason}", file=sys.stderr)


def check_header(fields, lineno, expected):
    if tuple(fields) != expected:
        print(
            f"WARNING: header on line {lineno} does not match the expected TSV contract: "
            f"expected {list(expected)}, got {fields}",
            file=sys.stderr,
        )


def parse_row(fields, lineno):
    if len(fields) != 10:
        warn_skipped(lineno, f"expected 10 fields, got {len(fields)}")
        return None
    for idx in (0, 1, 2, 3, 4, 6, 7, 8, 9):
        try:
            int(fields[idx])
        except ValueError:
            warn_skipped(lineno, f"non-integer value in column {idx + 1}: {fields[idx]!r}")
            return None
    closure = fields[5]
    if closure not in CLOSURES:
        warn_skipped(lineno, f"unknown closure {closure!r}")
        return None
    return (
        int(fields[0]),
        int(fields[1]),
        int(fields[4]),
        closure,
        int(fields[6]),
        int(fields[7]),
        int(fields[8]),
        int(fields[9]),
    )


def parse_tsv(path):
    """Return aggregates plus per-row records keyed by (window, row), or None."""
    cells = {(f, c): [0, 0] for f in FINALITIES for c in CLOSURES}
    no_prev_cell = [0, 0]
    headline_cells = {c: [0, 0] for c in CLOSURES}
    headline_len = {(c, b): [0, 0] for c in CLOSURES for b in BUCKETS}
    headline_pos = {(c, p): [0, 0] for c in CLOSURES for p in POS_BUCKETS}
    windows = set()
    rows = {}
    total_correct = 0
    total_rows = 0
    nonfinal_total = 0
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
                check_header(raw.rstrip("\r\n").split("\t"), lineno, HEADER)
                continue
            parsed = parse_row(raw.rstrip("\r\n").split("\t"), lineno)
            if parsed is None:
                continue
            window, row, patch_len, closure, is_final, pos, no_prev, correct = parsed
            key = (window, row)
            if key in rows:
                warn_skipped(lineno, f"duplicate key {key}")
                continue
            rows[key] = (is_final, no_prev, correct, closure)
            finality = "final" if is_final else "nonfinal"
            cells[(finality, closure)][0] += correct
            cells[(finality, closure)][1] += 1
            if not is_final:
                nonfinal_total += 1
            if no_prev:
                no_prev_cell[0] += correct
                no_prev_cell[1] += 1
            if is_final == 0 and no_prev == 0:
                headline_cells[closure][0] += correct
                headline_cells[closure][1] += 1
                headline_len[(closure, bucket_for(patch_len))][0] += correct
                headline_len[(closure, bucket_for(patch_len))][1] += 1
                headline_pos[(closure, pos_bucket_for(pos))][0] += correct
                headline_pos[(closure, pos_bucket_for(pos))][1] += 1
            total_correct += correct
            total_rows += 1
            windows.add(window)
    return {
        "cells": cells,
        "no_prev": no_prev_cell,
        "headline_cells": headline_cells,
        "headline_len": headline_len,
        "headline_pos": headline_pos,
        "rows": rows,
        "total_correct": total_correct,
        "total_rows": total_rows,
        "nonfinal_total": nonfinal_total,
        "windows": len(windows),
    }


def parse_baseline_row(fields, lineno):
    if len(fields) != 8:
        warn_skipped(lineno, f"baseline row: expected 8 fields, got {len(fields)}")
        return None
    for idx in (0, 1, 2, 3, 4, 6, 7):
        try:
            int(fields[idx])
        except ValueError:
            warn_skipped(lineno, f"baseline row: non-integer column {idx + 1}: {fields[idx]!r}")
            return None
    closure = fields[5]
    if closure not in CLOSURES:
        warn_skipped(lineno, f"baseline row: unknown closure {closure!r}")
        return None
    return (
        int(fields[0]),
        int(fields[1]),
        int(fields[2]),
        closure,
        int(fields[6]),
        int(fields[7]),
    )


def load_baseline(path):
    """Return {(window, row): (patch_idx, closure, is_final, correct)} plus nonfinal cells."""
    rows = {}
    nonfinal_cells = {c: [0, 0] for c in CLOSURES}
    try:
        handle = open(path, "r", encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"ERROR: cannot read baseline TSV {path}: {exc}", file=sys.stderr)
        return None
    with handle:
        seen_header = False
        for lineno, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            if not seen_header:
                seen_header = True
                check_header(raw.rstrip("\r\n").split("\t"), lineno, BASELINE_HEADER)
                continue
            parsed = parse_baseline_row(raw.rstrip("\r\n").split("\t"), lineno)
            if parsed is None:
                continue
            window, row, patch_idx, closure, is_final, correct = parsed
            rows[(window, row)] = (patch_idx, closure, is_final, correct)
            if not is_final:
                nonfinal_cells[closure][0] += correct
                nonfinal_cells[closure][1] += 1
    return {"rows": rows, "nonfinal_cells": nonfinal_cells}


def machine_expected(aggs):
    expected = {
        "total_correct": aggs["total_correct"],
        "total_rows": aggs["total_rows"],
        "windows": aggs["windows"],
    }
    for finality in FINALITIES:
        for closure in CLOSURES:
            correct, count = aggs["cells"][(finality, closure)]
            key = f"{finality}_{closure}"
            expected[key] = count
            expected[f"{key}_c"] = correct
    expected["no_prev"] = aggs["no_prev"][1]
    expected["no_prev_c"] = aggs["no_prev"][0]
    return expected


def load_machine_sum(path):
    """Return key totals summed over every windows= line, or None."""
    total = {}
    lines = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "windows=" not in line:
                    continue
                lines += 1
                for key, value in KV_RE.findall(line):
                    total[key] = total.get(key, 0) + int(value)
    except OSError as exc:
        print(f"ERROR: cannot read machine file {path}: {exc}", file=sys.stderr)
        return None
    if lines == 0:
        print(f"ERROR: no line containing 'windows=' found in {path}", file=sys.stderr)
        return None
    print(f"[VERIFY] summed {lines} machine line(s) from {path}", file=sys.stderr)
    return total


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
    print("[VERIFY] summed machine lines match TSV aggregates", file=sys.stderr)
    return True


def check_invariant(our_rows, base_rows):
    """Join on (window, row) and enforce correct-identity for no_prev rows.

    final_flips are informational: the decoder's full-causal byte
    self-attention mixes earlier non-final rows' changed cross-attn
    outputs into later final rows, so final-row logits can shift
    without their own group changing.
    """
    errors = []
    final_flips = 0
    no_prev_flips = 0
    nonfinal_changed_flips = 0
    joined = 0
    if len(our_rows) != len(base_rows):
        errors.append(f"row count mismatch: ours={len(our_rows)} baseline={len(base_rows)}")
    if len(our_rows) != EXPECTED_ROWS:
        errors.append(f"expected all {EXPECTED_ROWS} rows, ours has {len(our_rows)}")
    for key, (is_final, no_prev, correct, _closure) in our_rows.items():
        base = base_rows.get(key)
        if base is None:
            continue
        joined += 1
        base_correct = base[3]
        if correct == base_correct:
            continue
        if is_final:
            final_flips += 1
        elif no_prev:
            no_prev_flips += 1
        else:
            nonfinal_changed_flips += 1
    if joined != len(our_rows):
        errors.append(f"join incomplete: {joined}/{len(our_rows)} rows present in both files")
    if no_prev_flips:
        errors.append(
            f"no_prev_flips={no_prev_flips}: first-patch non-final rows changed correct "
            "between conventions (these keep group 0 under both rules; a flip indicates "
            "a construction error)"
        )
    return {
        "ok": not errors,
        "errors": errors,
        "joined": joined,
        "final_flips": final_flips,
        "no_prev_flips": no_prev_flips,
        "nonfinal_changed_flips": nonfinal_changed_flips,
    }


def headline_comparisons(our_rows, base):
    """Headline (a), matched baseline (b), all-baseline nonfinal (c) per closure + all."""
    base_rows = base["rows"]
    matched = {c: [0, 0] for c in CLOSURES}
    matched_all = [0, 0]
    for key, (is_final, no_prev, correct, closure) in our_rows.items():
        if is_final or no_prev:
            continue
        b = base_rows.get(key)
        if b is None:
            continue
        matched[closure][0] += b[3]
        matched[closure][1] += 1
        matched_all[0] += b[3]
        matched_all[1] += 1
    return {"matched": matched, "matched_all": matched_all}


def wilson(correct, count):
    """Return (acc, lo, hi) as fractions using the 95% Wilson score interval."""
    phat = correct / count
    denom = 1.0 + Z_95 * Z_95 / count
    center = (phat + Z_95 * Z_95 / (2.0 * count)) / denom
    variance = phat * (1.0 - phat) / count + Z_95 * Z_95 / (4.0 * count * count)
    half = Z_95 * math.sqrt(variance) / denom
    return phat, max(0.0, center - half), min(1.0, center + half)


def acc_str(correct, count):
    if count == 0:
        return "acc=—"
    return f"acc={correct / count * 100:.2f}%"


def format_cell(correct, count, min_n):
    if count == 0:
        text = "(no samples)"
    else:
        acc, lo, hi = wilson(correct, count)
        text = f"n={count} acc={acc * 100:.2f}% ({lo * 100:.2f}%–{hi * 100:.2f}%)"
    if count < min_n:
        text += " ⚠small-n"
    return text


def render_report(args, aggs, comps, inv, base):
    headline_total_c = sum(v[0] for v in aggs["headline_cells"].values())
    headline_total_n = sum(v[1] for v in aggs["headline_cells"].values())
    no_prev_c, no_prev_n = aggs["no_prev"]
    pct_all = no_prev_n / aggs["total_rows"] * 100.0 if aggs["total_rows"] else 0.0
    pct_nonfinal = no_prev_n / aggs["nonfinal_total"] * 100.0 if aggs["nonfinal_total"] else 0.0
    headline_rows_pct = headline_total_n / aggs["total_rows"] * 100.0 if aggs["total_rows"] else 0.0

    lines = [
        f"# {args.title}",
        "",
        f"Source: `{args.tsv}` — {aggs['total_rows']} rows, {aggs['windows']} windows",
        "",
        "## Setup",
        "",
        f"- Windows evaluated: {aggs['windows']}",
        f"- Total rows: {aggs['total_rows']} (expected {EXPECTED_ROWS}: "
        f"{'yes' if aggs['total_rows'] == EXPECTED_ROWS else 'NO'})",
        f"- Baseline: `{args.baseline_tsv}` ({len(base['rows'])} rows)",
        f"- Headline set: is_final==0 AND no_prev==0 → {headline_total_n} rows "
        f"({headline_rows_pct:.2f}% of all rows)",
        "- First-patch non-final bytes (no_prev==1) have no previous patch under the "
        "paper rule; they run on group 0 for the forward pass and are excluded from "
        "the headline aggregate (still counted in the machine-line nonfinal_* cells).",
        "",
        "## First-patch exclusion",
        "",
        f"- no_prev rows: n={no_prev_n}",
        f"- share of all rows: {pct_all:.2f}%",
        f"- share of non-final rows: {pct_nonfinal:.2f}% "
        f"(non-final total = {aggs['nonfinal_total']})",
        f"- no_prev accuracy under the paper rule: {no_prev_c}/{no_prev_n} "
        f"({acc_str(no_prev_c, no_prev_n)})",
        "",
        "## Non-final rows — paper rule",
        "",
        "Headline rows (is_final==0, no_prev==0) bucketed by patch length, 95% Wilson CI.",
        "",
        "| Length | nat | max | bnd |",
        "| --- | --- | --- | --- |",
    ]
    for bucket in BUCKETS:
        cells = []
        for closure in CLOSURES:
            correct, count = aggs["headline_len"][(closure, bucket)]
            cells.append(format_cell(correct, count, args.min_n))
        lines.append(f"| {bucket} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Non-final by closure")
    lines.append("")
    lines.append(
        "ours = headline accuracy under the paper rule; matched baseline = baseline "
        "accuracy on the identical (window, row) subset; all-baseline nonfinal = "
        "baseline accuracy over ALL its non-final rows of that closure (repo rule); "
        "delta = ours − matched baseline."
    )
    lines.append("")
    for closure in list(CLOSURES) + ["all"]:
        if closure == "all":
            ours_c, ours_n = headline_total_c, headline_total_n
            match_c, match_n = comps["matched_all"]
            base_cells = [
                (base["nonfinal_cells"][c][0], base["nonfinal_cells"][c][1]) for c in CLOSURES
            ]
            pub_c = sum(x[0] for x in base_cells)
            pub_n = sum(x[1] for x in base_cells)
            label = "overall"
        else:
            ours_c, ours_n = aggs["headline_cells"][closure]
            match_c, match_n = comps["matched"][closure]
            pub_c, pub_n = base["nonfinal_cells"][closure]
            label = closure
        delta = (
            f"{(ours_c / ours_n - match_c / match_n) * 100.0:+.2f}pp"
            if ours_n and match_n
            else "delta=—"
        )
        lines.append(
            f"- {label}: ours {acc_str(ours_c, ours_n)} (n={ours_n}) | "
            f"matched baseline {acc_str(match_c, match_n)} (n={match_n}) | "
            f"all-baseline nonfinal {acc_str(pub_c, pub_n)} (n={pub_n}) | "
            f"delta {delta}"
        )
    nat_pub_c, nat_pub_n = base["nonfinal_cells"]["nat"]
    nat_pub_acc = nat_pub_c / nat_pub_n * 100.0 if nat_pub_n else 0.0
    if (
        nat_pub_n == PUBLISHED_NAT_NONFINAL_ROWS
        and round(nat_pub_acc, 2) == PUBLISHED_NAT_NONFINAL_ACC
    ):
        lines.append(
            f"- note: all-baseline nonfinal nat reproduces the published "
            f"{PUBLISHED_NAT_NONFINAL_ACC}% on {PUBLISHED_NAT_NONFINAL_ROWS} rows "
            f"({nat_pub_c}/{nat_pub_n} = {nat_pub_acc:.4f}%)."
        )
    lines.append("")

    lines.append("## Position within patch")
    lines.append("")
    lines.append("Headline rows bucketed by 0-based position of the row inside its patch.")
    lines.append("")
    lines.append("| Position | nat | max | bnd |")
    lines.append("| --- | --- | --- | --- |")
    for pos_bucket in POS_BUCKETS:
        cells = []
        for closure in CLOSURES:
            correct, count = aggs["headline_pos"][(closure, pos_bucket)]
            cells.append(format_cell(correct, count, args.min_n))
        lines.append(f"| {pos_bucket} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Invariant check")
    lines.append("")
    lines.append(
        "no_prev==1 rows keep group 0 under both conventions and sit at the start "
        "of a window, where nothing before them changes; their correct flags must "
        "be identical between this dump and the baseline (hard invariant). "
        "final_flips are informational: rows with is_final==1 keep group j, but "
        "the decoder's full-causal byte self-attention mixes earlier non-final "
        "rows' changed cross-attn outputs into later rows, so final-row logits "
        "shift slightly without their own group changing."
    )
    lines.append("")
    lines.append(
        f"- joined rows: {inv['joined']} (ours {len(aggs['rows'])} / baseline "
        f"{len(base['rows'])}, expected {EXPECTED_ROWS})"
    )
    noprev_status = "OK" if inv["no_prev_flips"] == 0 else "FAIL"
    lines.append(
        f"- final_flips: {inv['final_flips']} "
        "(informational: causal self-attention mixing from earlier changed rows)"
    )
    lines.append(f"- no_prev_flips: {inv['no_prev_flips']} (must be 0) — {noprev_status}")
    lines.append(
        f"- nonfinal_changed_flips: {inv['nonfinal_changed_flips']} "
        "(informational: headline rows whose group id changed under the paper rule)"
    )
    lines.append("")

    lines.append("## Small-n caveats")
    lines.append("")
    caveats = []
    for closure in CLOSURES:
        count = aggs["headline_cells"][closure][1]
        if count < args.min_n:
            caveats.append(f"- headline/{closure}: n={count}")
        for bucket in BUCKETS:
            n = aggs["headline_len"][(closure, bucket)][1]
            if n < args.min_n:
                caveats.append(f"- headline/{closure}/len {bucket}: n={n}")
        for pos_bucket in POS_BUCKETS:
            n = aggs["headline_pos"][(closure, pos_bucket)][1]
            if n < args.min_n:
                caveats.append(f"- headline/{closure}/pos {pos_bucket}: n={n}")
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
        kv = load_machine_sum(args.machine)
        if kv is None:
            return 1
        if not verify_machine(kv, machine_expected(aggs)):
            return 1
    base = load_baseline(args.baseline_tsv)
    if base is None:
        return 1
    inv = check_invariant(aggs["rows"], base["rows"])
    comps = headline_comparisons(aggs["rows"], base)
    report = render_report(args, aggs, comps, inv, base)
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as handle:
                handle.write(report)
        except OSError as exc:
            print(f"ERROR: cannot write report {args.out}: {exc}", file=sys.stderr)
            return 1
    else:
        sys.stdout.write(report)
    if not inv["ok"]:
        for message in inv["errors"]:
            print(f"ERROR: {message}", file=sys.stderr)
        print("ERROR: invariant violated", file=sys.stderr)
        return 1
    print(
        f"[VERIFY] invariant holds: no_prev_flips={inv['no_prev_flips']} "
        f"(final_flips={inv['final_flips']} informational, "
        f"nonfinal_changed_flips={inv['nonfinal_changed_flips']})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
