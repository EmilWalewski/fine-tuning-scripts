#!/usr/bin/env python3
"""
leak_qa.py — cross-chunk contamination detector.

The failure mode this guards against: a summary that contains a specific
figure which does NOT appear in its own input but DOES appear in some OTHER
chunk's input. That is the signature of context bleed — the model carried a
number across chunk boundaries instead of processing each chunk in isolation.

Method (deliberately conservative, to minimise false alarms):
  1. Strip dates before extracting numbers. Dates are the #1 source of false
     positives because the same date legitimately appears in many chunks.
     We strip both numeric dates (DD.MM.YYYY, YYYY-MM-DD, ...) AND English
     "Month D, YYYY" / "D Month YYYY" / "Month YYYY" forms — the latter was a
     real false-positive source (e.g. "October 13, 2025" -> spurious 132025).
  2. Extract distinctive numbers: digit runs of length >= MIN_DIGITS after
     removing thousands separators. Short numbers (years, counts, percentages)
     are too common to be evidence of a leak, so they are ignored.
  3. For each record, flag output numbers that are absent from its own input
     but present in at least one other record's input.

A leak is strong evidence, not proof — review flagged records before acting.
Zero leaks across the corpus is the success condition for a regeneration pass.
"""
import argparse, json, os, re
from collections import defaultdict

from path_profiles import add_path_profile_arg, mapped_entry

MIN_DIGITS = 6

MONTHS = ("January|February|March|April|May|June|July|August|September|"
          "October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
DATE_PATTERNS = [
    re.compile(rf"\b(?:{MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b", re.I),  # Month D, YYYY
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTHS})\.?\s+\d{{4}}\b", re.I),     # D Month YYYY
    re.compile(rf"\b(?:{MONTHS})\.?\s+\d{{4}}\b", re.I),                                  # Month YYYY
    re.compile(r"\b\d{1,2}\s*[.\-/]\s*\d{1,2}\s*[.\-/]\s*\d{2,4}\b"),                     # DD.MM.YYYY (tolerates stray spaces)
    re.compile(r"\b\d{4}\s*[.\-/]\s*\d{1,2}\s*[.\-/]\s*\d{1,2}\b"),                       # YYYY-MM-DD (tolerates stray spaces)
]
NUM_RUN = re.compile(r"\d[\d.,   ]*\d|\d")

def strip_dates(text):
    for p in DATE_PATTERNS:
        text = p.sub(" ", text)
    return text

def numbers(text, min_digits=MIN_DIGITS):
    """Distinctive numbers in a piece of text (dates removed, separators stripped)."""
    text = strip_dates(text)
    out = set()
    for m in NUM_RUN.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if len(digits) >= min_digits:
            out.add(digits)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--verbose", action="store_true", help="List each leaked number and where it actually belongs")
    ap.add_argument("--min-digits", type=int, default=MIN_DIGITS)
    add_path_profile_arg(ap)
    args = ap.parse_args()
    md = args.min_digits

    manifest = json.load(open(args.manifest, encoding="utf-8"))

    in_nums, out_nums = {}, {}
    for m in manifest:
        m = mapped_entry(m, args.path_profile)
        k = m["key"]
        in_nums[k] = numbers(open(m["input_path"], encoding="utf-8").read(), md) if os.path.exists(m["input_path"]) else set()
        op = m["output_path"]
        out_nums[k] = numbers(open(op, encoding="utf-8").read(), md) if os.path.exists(op) and os.path.getsize(op) > 0 else set()

    # Reverse index: number -> set of record keys whose INPUT contains it
    num_to_inputs = defaultdict(set)
    for k, nums in in_nums.items():
        for n in nums:
            num_to_inputs[n].add(k)

    flagged, total_leaks = [], 0
    by_key = {m["key"]: mapped_entry(m, args.path_profile) for m in manifest}
    for m in manifest:
        m = mapped_entry(m, args.path_profile)
        k = m["key"]
        leaks = []
        for n in out_nums[k]:
            if n in in_nums[k]:
                continue  # legitimately from own input
            others = num_to_inputs.get(n, set()) - {k}
            if others:
                leaks.append((n, sorted(by_key[o]["output_path"].rsplit("/", 1)[-1] for o in others)[:3]))
        if leaks:
            flagged.append((m, leaks))
            total_leaks += len(leaks)

    print(f"Records checked      : {len(manifest)}")
    print(f"Records with leaks   : {len(flagged)}")
    print(f"Total leaked numbers : {total_leaks}")
    if not flagged:
        print("\nCLEAN — no cross-chunk numeric contamination detected.")
        return
    print("\nFlagged records (number absent from own input, present in another chunk):")
    for m, leaks in sorted(flagged, key=lambda x: -len(x[1])):
        name = m["output_path"].rsplit("/", 1)[-1]
        print(f"  [{len(leaks):>2}] {name}")
        if args.verbose:
            for n, where in leaks:
                print(f"        {n}  ~ also in: {', '.join(where)}")

if __name__ == "__main__":
    main()
