#!/usr/bin/env python3
"""
remaining.py — the idempotent batch driver for stateless regeneration.

Prints the records that still need a summary: those whose output file is
missing or empty. This is what makes the whole workflow resumable — after a
crash, a session limit, or a partial batch, just run this again and it tells
you exactly what is left. Nothing is recomputed that already succeeded.

Default output is `INPUT || OUTPUT` lines (handy to read and to feed the next
round of subagents). Use --json to also dump the full remaining list to a file.
"""
import argparse, json, os

def remaining(manifest):
    rem = []
    for m in manifest:
        op = m["output_path"]
        if (not os.path.exists(op)) or os.path.getsize(op) == 0:
            rem.append(m)
    return rem

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=15, help="How many to print (0 = all)")
    ap.add_argument("--json", default=None, help="Optional path to dump the full remaining list as JSON")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    rem = remaining(manifest)
    done = len(manifest) - len(rem)
    print(f"# done {done}/{len(manifest)} | remaining {len(rem)}")
    if args.json:
        json.dump(rem, open(args.json, "w", encoding="utf-8"), ensure_ascii=False)
    n = len(rem) if args.limit == 0 else args.limit
    for m in rem[:n]:
        line = f"{m['input_path']} || {m['output_path']}"
        if "out_char_budget" in m:
            line += f" || BUDGET={m['out_char_budget']}"
        print(line)

if __name__ == "__main__":
    main()
