#!/usr/bin/env python3
"""Validate regenerated chunk outputs before merging/training.

This is deliberately conservative: it flags suspicious records for review or
regeneration instead of trying to repair prose automatically.
"""
import argparse
import json
import os
import re
import sys

from path_profiles import add_path_profile_arg, mapped_entry


PROMPT_MARKERS = ("### Instruction", "### Input", "### Output")
BAD_FINAL_WORDS = {
    "and", "or", "of", "to", "for", "with", "from", "by", "as", "in", "on",
    "the", "a", "an", "including", "while", "PL", "PLN", "EUR", "USD",
}
SUSPICIOUS_M_RE = re.compile(r"\b(?:PLN|EUR|USD)\s+\d{1,3}[,\s]\d{3,}(?:\.\d+)?M\b|\b\d{1,3}[,\s]\d{3,}(?:\.\d+)?M\s+(?:PLN|EUR|USD)\b")
THOUSANDS_AMOUNT_RE = re.compile(
    r"(?P<num>\d{1,3}(?:[ \u00a0]\d{3})+|\d{4,})\s*"
    r"(?P<unit>tys\.?|tysi(?:ą|a)c(?:ach|e|y)?|thousand(?:s)?)\s*"
    r"(?P<cur>zł|PLN|EUR|USD)?",
    re.IGNORECASE,
)


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _amount_to_int(num):
    return int(re.sub(r"[^\d]", "", num))


def _contains_wrong_magnitude(output, thousands_value):
    """Catch common off-by-10 / off-by-100 / not-divided-by-1000 mistakes."""
    bad_values = {
        thousands_value / 100,   # 56 887k -> 568.9M
        thousands_value / 10,    # 56 887k -> 5,688.7M
        float(thousands_value),  # 56 887k -> 56,887M
    }
    for val in bad_values:
        variants = {
            f"{val:,.1f}",
            f"{val:.1f}",
            f"{val:,.0f}",
            f"{val:.0f}",
        }
        for v in variants:
            if re.search(rf"\b{re.escape(v)}\s*M\b|\bM\s*{re.escape(v)}\b", output):
                return True
    return False


def _ends_incomplete(output):
    stripped = output.strip()
    if not stripped or stripped == "INSUFFICIENT_DATA":
        return False
    if stripped[-1] in ".!?)%]\"'":
        return False
    last = re.sub(r"[^A-Za-z]", "", stripped.split()[-1]).lower()
    return last in BAD_FINAL_WORDS or len(last) <= 2


def validate_record(entry):
    issues = []
    inp = _read(entry["input_path"])
    out_path = entry["output_path"]
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return ["missing_or_empty_output"]
    out = _read(out_path).strip()

    for marker in PROMPT_MARKERS:
        if marker in out:
            issues.append("prompt_marker_leak")
            break

    budget = entry.get("out_char_budget")
    if budget and len(out) > int(budget):
        issues.append(f"over_budget:{len(out)}>{budget}")

    if _ends_incomplete(out):
        issues.append("possibly_truncated_ending")

    if SUSPICIOUS_M_RE.search(out):
        issues.append("suspicious_million_unit_format")

    for match in THOUSANDS_AMOUNT_RE.finditer(inp):
        value = _amount_to_int(match.group("num"))
        if value < 1000:
            continue
        if _contains_wrong_magnitude(out, value):
            issues.append(f"possible_thousands_to_millions_error:{match.group('num')}")
            break

    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--fail-on-issues", action="store_true")
    add_path_profile_arg(ap)
    args = ap.parse_args()

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    flagged = []
    for raw in manifest:
        entry = mapped_entry(raw, args.path_profile)
        issues = validate_record(entry)
        if issues:
            flagged.append((entry, issues))

    if not flagged:
        print("OUTPUT_QA: CLEAN")
        return

    print(f"OUTPUT_QA: FLAGGED {len(flagged)} / {len(manifest)}")
    for entry, issues in flagged[:200 if args.verbose else 30]:
        print(f"{entry['key']} file={entry['file']} id={entry['id']} issues={','.join(issues)} output={entry['output_path']}")
    if len(flagged) > (200 if args.verbose else 30):
        print(f"... {len(flagged) - (200 if args.verbose else 30)} more")
    if args.fail_on_issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
