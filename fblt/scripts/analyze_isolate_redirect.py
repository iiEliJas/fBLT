"""
Verify and report the isolated single-patch paper-rule redirect eval.

Reads the scored-row TSV and per-pass invariant dump written by
build/eval_isolate_redirect, joins them against the patch_final baseline and
the part-1 all-redirected dump on (window, row), hard-fails when any pre-r
row flipped relative to the baseline, and renders a Markdown report whose
headline table compares isolated-single-patch accuracy against part 1's
all-redirected accuracy and the leaky baseline on identical row sets.

The scored set is exactly the part-1 headline set: non-final rows of every
patch j >= 1 (patch 0 is never redirected — no previous patch — matching
part 1's no_prev exclusion).

Usage:
    python3 fblt/scripts/analyze_isolate_redirect.py --tsv dump.tsv \
        --inv dump.tsv.inv --machine machine.txt --out report.md
"""

import argparse
import math
import re
import sys

CLOSURES = ("nat", "max", "bnd")
BUCKETS = ("1-2", "3-4", "5-8", "9-16", "17+")
Z_95 = 1.959963984540054
KV_RE = re.compile(r"(\w+)=(\d+)")
HEADER = (
    "window",
    "row",
    "redirect_patch_idx",
    "patch_start",
    "patch_len",
    "closure",
    "pos_in_patch",
    "correct",
    "baseline_correct",
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
PART1_HEADER = (
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
MACHINE_KEYS = (
    "windows passes scored_rows scored_correct pre_rows pre_flips skipped_r0 skipped_m128"
).split()
EXPECTED_WINDOWS = 500
EXPECTED_SCORED_ROWS = 205910


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Verify and report the isolated single-patch redirect eval."
    )
    p.add_argument(
        "--tsv", required=True, help="scored-row TSV written by build/eval_isolate_redirect"
    )
    p.add_argument(
        "--inv",
        required=True,
        help="per-pass invariant dump written by build/eval_isolate_redirect",
    )
    p.add_argument("--machine", default=None, help="file containing the shard machine lines")
    p.add_argument(
        "--baseline-tsv",
        default="runs/tinystories_p7/analyses/patch_final_500w.tsv",
        help="patch_final baseline TSV with the repo-rule group ids",
    )
    p.add_argument(
        "--part1-tsv",
        default="runs/tinystories_p7/analyses/paper_rule_500w.tsv",
        help="part-1 all-redirected TSV (10-column paper-rule dump)",
    )
    p.add_argument("--out", default=None, help="write the Markdown report here instead of stdout")
    p.add_argument(
        "--title",
        default="Isolated single-patch paper-rule redirect",
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


def warn_skipped(lineno, reason):
    print(f"WARNING: skipping malformed line {lineno}: {reason}", file=sys.stderr)


def check_header(fields, lineno, expected):
    if tuple(fields) != expected:
        print(
            f"WARNING: header on line {lineno} does not match the expected TSV contract: "
            f"expected {list(expected)}, got {fields}",
            file=sys.stderr,
        )
        return False
    return True


def shown(value):
    return "—" if value is None else str(value)


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


def parse_ours(path):
    """Return scored-row records plus aggregates, or None on unreadable file."""
    grid = {(c, b): [0, 0] for c in CLOSURES for b in BUCKETS}
    closure_cells = {c: [0, 0] for c in CLOSURES}
    bucket_cells = {b: [0, 0] for b in BUCKETS}
    rows = {}
    windows = set()
    scored_correct = 0
    errors = []
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
                if not check_header(raw.rstrip("\r\n").split("\t"), lineno, HEADER):
                    errors.append(
                        f"header on line {lineno} is not the 9-column scored-row contract"
                    )
                continue
            fields = raw.rstrip("\r\n").split("\t")
            if len(fields) != 9:
                warn_skipped(lineno, f"expected 9 fields, got {len(fields)}")
                errors.append(f"line {lineno}: expected 9 fields, got {len(fields)}")
                continue
            try:
                window = int(fields[0])
                row = int(fields[1])
                patch_len = int(fields[4])
                correct = int(fields[7])
                baseline_correct = int(fields[8])
            except ValueError:
                warn_skipped(lineno, "non-integer value")
                errors.append(f"line {lineno}: non-integer value")
                continue
            closure = fields[5]
            if closure not in CLOSURES:
                warn_skipped(lineno, f"unknown closure {closure!r}")
                errors.append(f"line {lineno}: unknown closure {closure!r}")
                continue
            key = (window, row)
            if key in rows:
                warn_skipped(lineno, f"duplicate key {key}")
                errors.append(f"duplicate key {key}")
                continue
            rows[key] = (correct, baseline_correct, patch_len, closure)
            bucket = bucket_for(patch_len)
            grid[(closure, bucket)][0] += correct
            grid[(closure, bucket)][1] += 1
            closure_cells[closure][0] += correct
            closure_cells[closure][1] += 1
            bucket_cells[bucket][0] += correct
            bucket_cells[bucket][1] += 1
            scored_correct += correct
            windows.add(window)
    return {
        "rows": rows,
        "grid": grid,
        "closure_cells": closure_cells,
        "bucket_cells": bucket_cells,
        "windows": len(windows),
        "window_ids": windows,
        "scored_rows": len(rows),
        "scored_correct": scored_correct,
        "errors": errors,
    }


def parse_inv(path):
    """Return per-pass invariant lines and sums, or None on unreadable file."""
    lines = []
    pre_rows = 0
    pre_flips = 0
    errors = []
    seen = set()
    try:
        handle = open(path, "r", encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"ERROR: cannot read invariant dump {path}: {exc}", file=sys.stderr)
        return None
    with handle:
        for lineno, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            fields = raw.split()
            if len(fields) != 4:
                errors.append(f"inv line {lineno}: expected 4 fields, got {len(fields)}")
                continue
            try:
                window, redirect, rows_here, flips_here = (int(x) for x in fields)
            except ValueError:
                errors.append(f"inv line {lineno}: non-integer field")
                continue
            if redirect < 1:
                errors.append(f"inv line {lineno}: redirect_patch_idx={redirect} < 1")
            if (window, redirect) in seen:
                errors.append(f"inv line {lineno}: duplicate pass (window={window}, r={redirect})")
            seen.add((window, redirect))
            lines.append((window, redirect, rows_here, flips_here))
            pre_rows += rows_here
            pre_flips += flips_here
    return {
        "lines": lines,
        "passes": len(lines),
        "pre_rows": pre_rows,
        "pre_flips": pre_flips,
        "errors": errors,
    }


def parse_part1_headline(path):
    """Return part-1 headline rows (is_final==0 and no_prev==0), or None."""
    rows = {}
    errors = []
    try:
        handle = open(path, "r", encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"ERROR: cannot read part-1 TSV {path}: {exc}", file=sys.stderr)
        return None
    with handle:
        seen_header = False
        for lineno, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            if not seen_header:
                seen_header = True
                check_header(raw.rstrip("\r\n").split("\t"), lineno, PART1_HEADER)
                continue
            fields = raw.rstrip("\r\n").split("\t")
            if len(fields) != 10:
                warn_skipped(lineno, f"part-1 row: expected 10 fields, got {len(fields)}")
                continue
            try:
                window = int(fields[0])
                row = int(fields[1])
                patch_len = int(fields[4])
                is_final = int(fields[6])
                no_prev = int(fields[8])
                correct = int(fields[9])
            except ValueError:
                warn_skipped(lineno, "part-1 row: non-integer value")
                continue
            if is_final == 0 and no_prev == 0:
                rows[(window, row)] = (correct, patch_len)
    return {"rows": rows, "count": len(rows), "errors": errors}


def parse_baseline(path):
    """Return baseline rows keyed by (window, row) plus per-window patch counts."""
    rows = {}
    window_max_patch = {}
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
            fields = raw.rstrip("\r\n").split("\t")
            if len(fields) != 8:
                warn_skipped(lineno, f"baseline row: expected 8 fields, got {len(fields)}")
                continue
            try:
                window = int(fields[0])
                row = int(fields[1])
                patch_idx = int(fields[2])
                patch_len = int(fields[4])
                is_final = int(fields[6])
                correct = int(fields[7])
            except ValueError:
                warn_skipped(lineno, "baseline row: non-integer value")
                continue
            rows[(window, row)] = (patch_idx, patch_len, is_final, correct)
            window_max_patch[window] = max(window_max_patch.get(window, -1), patch_idx)
    return {"rows": rows, "window_max_patch": window_max_patch}


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


def run_checks(ours, inv, part1, baseline, kv):
    """Build the ordered check list; every entry is (label, ok, detail)."""
    checks = []

    header_errors = [e for e in ours["errors"] if e.startswith("header on line")]
    parse_errors = [e for e in ours["errors"] if not e.startswith("header on line")]
    checks.append(
        (
            "scored-row TSV header (9 columns)",
            not header_errors,
            "ok" if not header_errors else "; ".join(header_errors),
        )
    )
    checks.append(
        (
            "scored-row TSV parses cleanly",
            not parse_errors,
            "ok"
            if not parse_errors
            else f"{len(parse_errors)} problem(s), first: {parse_errors[0]}",
        )
    )

    checks.append(
        (
            f"windows == {EXPECTED_WINDOWS}",
            ours["windows"] == EXPECTED_WINDOWS,
            f"got {ours['windows']}",
        )
    )
    checks.append(
        (
            f"scored_rows == {EXPECTED_SCORED_ROWS}",
            ours["scored_rows"] == EXPECTED_SCORED_ROWS,
            f"got {ours['scored_rows']}",
        )
    )
    checks.append(
        (
            f"part-1 headline count == {EXPECTED_SCORED_ROWS}",
            part1["count"] == EXPECTED_SCORED_ROWS,
            f"got {part1['count']}",
        )
    )
    our_keys = set(ours["rows"])
    part1_keys = set(part1["rows"])
    missing_part1 = len(our_keys - part1_keys)
    missing_ours = len(part1_keys - our_keys)
    checks.append(
        (
            "row-set equality vs part-1 headline on (window, row)",
            our_keys == part1_keys,
            f"ours-only={missing_part1} part1-only={missing_ours}",
        )
    )

    checks.append(
        (
            "pre_flips sum across invariant lines == 0 (HARD)",
            inv["pre_flips"] == 0,
            f"pre_flips={inv['pre_flips']}, pre_rows={inv['pre_rows']}",
        )
    )
    inv_errors = inv["errors"]
    checks.append(
        (
            "invariant dump parses cleanly",
            not inv_errors,
            "ok" if not inv_errors else f"{len(inv_errors)} problem(s), first: {inv_errors[0]}",
        )
    )

    expected_passes = 0
    for window in ours["window_ids"]:
        max_patch = baseline["window_max_patch"].get(window, -1)
        expected_passes += max(max_patch, 0)
    checks.append(
        (
            "invariant line count == baseline-derived sum(M-1)",
            inv["passes"] == expected_passes,
            f"inv lines={inv['passes']} expected={expected_passes}",
        )
    )

    if kv is None:
        checks.append(("machine line verification", True, "no --machine provided (skipped)"))
    else:
        recomputed = {
            "windows": ours["windows"],
            "passes": inv["passes"],
            "scored_rows": ours["scored_rows"],
            "scored_correct": ours["scored_correct"],
            "pre_rows": inv["pre_rows"],
            "pre_flips": inv["pre_flips"],
        }
        problems = []
        for key in MACHINE_KEYS:
            if key not in kv:
                problems.append(f"{key} missing")
                continue
            if key in recomputed and kv[key] != recomputed[key]:
                problems.append(f"{key}: machine={kv[key]} recomputed={recomputed[key]}")
        checks.append(
            (
                "machine sums match TSV/invariant recomputation",
                not problems,
                "ok" if not problems else "; ".join(problems),
            )
        )

    join_errors = []
    for key, (_correct, baseline_correct, _patch_len, _closure) in ours["rows"].items():
        base = baseline["rows"].get(key)
        if base is None:
            join_errors.append(f"missing baseline key {key}")
            continue
        if baseline_correct != base[3]:
            join_errors.append(
                f"baseline_correct mismatch at {key}: dump={baseline_correct} file={base[3]}"
            )
        if len(join_errors) > 5:
            break
    checks.append(
        (
            "baseline_correct column matches --baseline-tsv on every row",
            not join_errors,
            "ok" if not join_errors else f"{len(join_errors)}+ problem(s), first: {join_errors[0]}",
        )
    )
    return checks


def render_report(args, ours, inv, part1, baseline, kv, checks):
    all_ok = all(ok for _label, ok, _detail in checks)
    total_correct = ours["scored_correct"]
    total_rows = ours["scored_rows"]

    skipped_r0 = kv.get("skipped_r0") if kv else None
    skipped_m128 = kv.get("skipped_m128") if kv else None
    post_rows = kv.get("post_rows") if kv else None
    post_flips = kv.get("post_flips") if kv else None

    lines = [
        f"# {args.title}",
        "",
        f"Source: `{args.tsv}` — {total_rows} scored rows, {ours['windows']} windows",
        "",
        "## Setup",
        "",
        f"- Windows evaluated: {ours['windows']} (expected {EXPECTED_WINDOWS}: "
        f"{'yes' if ours['windows'] == EXPECTED_WINDOWS else 'NO'})",
        f"- Scored rows: {total_rows} (expected {EXPECTED_SCORED_ROWS}: "
        f"{'yes' if total_rows == EXPECTED_SCORED_ROWS else 'NO'})",
        f"- Scored correct: {total_correct} ({acc_str(total_correct, total_rows)})",
        f"- Passes (invariant lines): {inv['passes']}",
        f"- Part-1 headline rows (is_final==0, no_prev==0): {part1['count']}; row-set equality: "
        f"{'yes' if set(ours['rows']) == set(part1['rows']) else 'NO'}",
        f"- Baseline: `{args.baseline_tsv}` ({len(baseline['rows'])} rows)",
        f"- Part-1 dump: `{args.part1_tsv}` ({part1['count']} headline rows)",
        f"- skipped_r0: {shown(skipped_r0)} — patch 0 is never redirected: it has no "
        "previous patch, matching part 1's exclusion of first-patch non-final (no_prev) "
        "rows from the headline aggregate.",
        f"- skipped_m128: {shown(skipped_m128)} "
        "(windows with M>=128 skip identically to part 1, so the window set matches)",
        "- Machine verification: "
        + ("not run (no --machine)" if kv is None else "summed shard lines, see check results"),
        "",
        "### Checks",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for label, ok, detail in checks:
        lines.append(f"| {label} | {'PASS' if ok else 'FAIL'} | {detail} |")
    lines.append("")

    noprev_status = "OK" if inv["pre_flips"] == 0 else "FAIL"
    lines += [
        "## Invariant",
        "",
        "Every row before the redirected patch r keeps its group id under the isolated "
        "redirect (only patch r's non-final bytes change group), so its correct flag must "
        "match the patch_final baseline exactly — a flip indicates a construction or "
        "plumbing error (hard invariant, the part-2 analog of part 1's no_prev_flips). "
        "post_flips are informational: causal byte self-attention can mix the redirected "
        "patch's changed rows into later rows without those rows' own group changing.",
        "",
        f"- pre_rows checked: {inv['pre_rows']} row comparisons against the baseline",
        f"- pre_flips: {inv['pre_flips']} (HARD: must be 0) — {noprev_status}",
    ]
    if post_rows is not None:
        lines.append(
            f"- post_rows/post_flips: {post_rows}/{post_flips} (informational, from machine line)"
        )
    lines.append("")

    lines += [
        "## Isolated redirect accuracy by patch length × closure",
        "",
        "Scored rows only (non-final bytes of the redirected patch), 95% Wilson CI.",
        "",
        "| Length | nat | max | bnd |",
        "| --- | --- | --- | --- |",
    ]
    for bucket in BUCKETS:
        cells = []
        for closure in CLOSURES:
            correct, count = ours["grid"][(closure, bucket)]
            cells.append(format_cell(correct, count, args.min_n))
        lines.append(f"| {bucket} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += [
        "## Comparison: isolated vs all-redirected vs leaky baseline",
        "",
        "All three columns cover the identical (window, row) set — this run's scored rows, "
        "joined against the part-1 all-redirected dump and the patch_final leaky baseline. "
        "95% Wilson CI.",
        "",
        "| Patch length | Isolated single-patch (this run) | "
        "All-redirected (part 1) | Leaky baseline |",
        "| --- | --- | --- | --- |",
    ]

    def column_for(keys, which):
        correct = 0
        count = 0
        for key in keys:
            if which == "ours":
                correct += ours["rows"][key][0]
                count += 1
            elif which == "part1":
                correct += part1["rows"][key][0]
                count += 1
            else:
                base = baseline["rows"].get(key)
                if base is not None:
                    correct += base[3]
                    count += 1
        return correct, count

    def row_keys(bucket=None, closure=None):
        keys = []
        for key, (_c, _bc, patch_len, row_closure) in ours["rows"].items():
            if bucket is not None and bucket_for(patch_len) != bucket:
                continue
            if closure is not None and row_closure != closure:
                continue
            keys.append(key)
        return keys

    comparison = {}
    for bucket in BUCKETS:
        keys = row_keys(bucket=bucket)
        comparison[bucket] = {which: column_for(keys, which) for which in ("ours", "part1", "base")}
        cells = [
            format_cell(*comparison[bucket][which], args.min_n)
            for which in ("ours", "part1", "base")
        ]
        lines.append(f"| {bucket} | " + " | ".join(cells) + " |")
    overall_keys = row_keys()
    comparison["overall"] = {
        which: column_for(overall_keys, which) for which in ("ours", "part1", "base")
    }
    cells = [
        format_cell(*comparison["overall"][which], args.min_n)
        for which in ("ours", "part1", "base")
    ]
    lines.append("| overall | " + " | ".join(cells) + " |")
    lines.append("")

    lines += [
        "## Non-final by closure (matched row sets)",
        "",
        "isolated = this run on the redirected patch's non-final bytes; part1 = the same "
        "(window, row) keys under the all-redirected rule; baseline = the same keys under "
        "the leaky repo rule; deltas are isolated − part1 and isolated − baseline.",
        "",
    ]
    for closure in [*CLOSURES, "all"]:
        keys = row_keys(closure=None if closure == "all" else closure)
        iso_c, iso_n = column_for(keys, "ours")
        p1_c, p1_n = column_for(keys, "part1")
        base_c, base_n = column_for(keys, "base")
        label = "overall" if closure == "all" else closure
        if iso_n and p1_n:
            d_part1 = f"{(iso_c / iso_n - p1_c / p1_n) * 100.0:+.2f}pp"
        else:
            d_part1 = "—"
        if iso_n and base_n:
            d_base = f"{(iso_c / iso_n - base_c / base_n) * 100.0:+.2f}pp"
        else:
            d_base = "—"
        lines.append(
            f"- {label}: isolated {acc_str(iso_c, iso_n)} (n={iso_n}) | "
            f"part1 {acc_str(p1_c, p1_n)} (n={p1_n}) | "
            f"baseline {acc_str(base_c, base_n)} (n={base_n}) | "
            f"delta iso−part1 {d_part1} | delta iso−base {d_base}"
        )
    lines.append("")

    lines += ["## Small-n caveats", ""]
    caveats = []
    for closure in CLOSURES:
        count = ours["closure_cells"][closure][1]
        if count < args.min_n:
            caveats.append(f"- all/{closure}: n={count}")
        for bucket in BUCKETS:
            n = ours["grid"][(closure, bucket)][1]
            if n < args.min_n:
                caveats.append(f"- {closure}/len {bucket}: n={n}")
    for bucket in BUCKETS:
        count = ours["bucket_cells"][bucket][1]
        if count < args.min_n:
            caveats.append(f"- all/len {bucket}: n={count}")
    if caveats:
        lines.append(f"Cells below n={args.min_n} are unstable and flagged ⚠small-n above:")
        lines.append("")
        lines.extend(caveats)
    else:
        lines.append(f"- none (all cells n>={args.min_n})")
    lines.append("")

    status = "ALL CHECKS PASS" if all_ok else "CHECK FAILURES PRESENT"
    lines.append(f"Overall: {status}.")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    ours = parse_ours(args.tsv)
    if ours is None:
        return 1
    inv = parse_inv(args.inv)
    if inv is None:
        return 1
    baseline = parse_baseline(args.baseline_tsv)
    if baseline is None:
        return 1
    part1 = parse_part1_headline(args.part1_tsv)
    if part1 is None:
        return 1
    kv = None
    if args.machine:
        kv = load_machine_sum(args.machine)
        if kv is None:
            kv_missing = True
        else:
            kv_missing = False
    else:
        kv_missing = False

    checks = run_checks(ours, inv, part1, baseline, None if kv_missing else kv)
    report = render_report(args, ours, inv, part1, baseline, None if kv_missing else kv, checks)
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as handle:
                handle.write(report)
        except OSError as exc:
            print(f"ERROR: cannot write report {args.out}: {exc}", file=sys.stderr)
            return 1
    else:
        sys.stdout.write(report)

    failures = [(label, detail) for label, ok, detail in checks if not ok]
    if kv_missing:
        print("ERROR: machine file was provided but could not be read", file=sys.stderr)
        return 1
    if failures:
        for label, detail in failures:
            print(f"ERROR: {label}: {detail}", file=sys.stderr)
        print("ERROR: checks failed", file=sys.stderr)
        return 1
    print(
        f"[VERIFY] all checks pass: windows={ours['windows']} scored_rows={ours['scored_rows']} "
        f"passes={inv['passes']} pre_rows={inv['pre_rows']} pre_flips={inv['pre_flips']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
