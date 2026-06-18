#!/usr/bin/env python3
"""
measure_tokens.py — length QA: does each record fit the context window?

Builds the exact training text (### Instruction / ### Input / ### Output + EOS)
for every record and tokenises it, then reports the distribution and — given a
--window — exactly which records overflow and by how much. Run it:
  * before training, to choose a window, and
  * after a budget-constrained regeneration, to confirm everything now fits.

Tokeniser: tiktoken cl100k_base when available (a conservative proxy that
slightly over-counts vs Llama 3, so "fits here" implies "fits there"). Falls
back to a chars/token heuristic if tiktoken isn't installed — less precise, so
prefer installing tiktoken (`pip install tiktoken`) for this check.

Needs the original source folder (for `instruction`) via --src, since the
manifest only stores the input dump and output path.
"""
import argparse, glob, json, os, sys

from path_profiles import add_path_profile_arg, mapped_entry

CPT_FALLBACK = 2.8

def get_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return (lambda s: len(enc.encode(s or ""))), True
    except Exception:
        return (lambda s: int(len(s or "") / CPT_FALLBACK)), False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--src", required=True, help="Original folder of .jsonl files (for instruction)")
    ap.add_argument("--window", type=int, default=None, help="Report records exceeding this token window")
    ap.add_argument("--id-key", default="id")
    add_path_profile_arg(ap)
    args = ap.parse_args()

    count, exact = get_counter()
    manifest = json.load(open(args.manifest, encoding="utf-8"))

    orig = {}
    for f in glob.glob(os.path.join(args.src, "*.jsonl")):
        b = os.path.basename(f)
        for ln, line in enumerate(open(f, encoding="utf-8"), 1):
            line = line.strip()
            if line:
                r = json.loads(line)
                orig[(b, str(r.get(args.id_key, ln)))] = r

    rows = []
    for m in manifest:
        m = mapped_entry(m, args.path_profile)
        instr = orig.get((m["file"], str(m["id"])), {}).get("instruction", "")
        inp = open(m["input_path"], encoding="utf-8").read() if os.path.exists(m["input_path"]) else ""
        op = m["output_path"]
        out = open(op, encoding="utf-8").read() if os.path.exists(op) and os.path.getsize(op) > 0 else ""
        text = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Output:\n{out}"
        rows.append((count(text) + 1, op.rsplit("/", 1)[-1]))

    tot = sorted(r[0] for r in rows)
    n = len(tot)
    pct = lambda p: tot[min(n - 1, int(p * n))]
    print(f"Tokeniser: {'cl100k_base' if exact else 'chars/token heuristic'} | records: {n}")
    print(f"  median {pct(.5)}  mean {sum(tot)//n}  p90 {pct(.9)}  p95 {pct(.95)}  p99 {pct(.99)}  MAX {tot[-1]}")
    if args.window:
        over = sorted([r for r in rows if r[0] > args.window], reverse=True)
        print(f"\n  > {args.window}: {len(over)} record(s)")
        for total, name in over:
            print(f"    {total:>6}  (+{total-args.window})  {name}")
        if not over:
            print("  All records fit the window.")

if __name__ == "__main__":
    main()
