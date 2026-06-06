#!/usr/bin/env python3
"""
build_manifest.py — Stage 1 of stateless chunk regeneration.

Scans a folder of JSONL files (one record per line, Alpaca-style keys
id / instruction / input / output) and prepares everything the isolated
subagents need:

  <work>/manifest.json     one entry per record (key, file, id, stem,
                           input_path, input_chars, output_path, and — when a
                           context window is set — out_char_budget)
  <work>/inputs/<key>.txt  the raw `input` text of each record, one file
                           per chunk — this is the ONLY thing a subagent reads
  <work>/RULES.md          the analyst contract + isolation + output rules

The per-record input dump is the heart of the isolation guarantee: each
subagent is handed exactly one of these files and forbidden from touching
anything else, so it physically cannot bleed figures from a neighbouring chunk.

CONTEXT-WINDOW / LENGTH BUDGET
------------------------------
Fine-tuning packs `instruction + input + output` into one fixed context window
(e.g. 8192 for Llama 3). The input is fixed, so the only lever to keep a record
inside the window is the OUTPUT length. With --window set, this script computes
a per-record output budget (in characters) = window - input_tokens - reserve,
and stores it as `out_char_budget`. The driver then tells each subagent "keep
your summary under N characters", so generated outputs fit by construction —
no post-hoc truncation of summaries, which is what destroys training signal.

Token counting uses tiktoken (cl100k_base, a conservative proxy that slightly
over-counts vs Llama 3, so budgets err on the safe side) when available; if
tiktoken isn't installed it falls back to a chars/token heuristic.

Re-running is safe: it overwrites manifest/inputs/RULES but never touches the
output directory, so regeneration progress is preserved.
"""
import argparse, hashlib, json, os, sys, shutil

# Conservative chars/token constants for the heuristic fallback. Polish source
# text tokenises densely (low chars/token) so we under-estimate chars/token for
# INPUT (=> higher token estimate => smaller, safer output budget); English
# summaries are looser, so OUTPUT uses a higher chars/token to convert a token
# budget into a character cap.
CPT_INPUT_FALLBACK = 2.8
CPT_OUTPUT = 3.6

def get_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s or "")), True
    except Exception:
        return lambda s: int(len(s or "") / CPT_INPUT_FALLBACK), False

def key_for(file_base, rec_id):
    return hashlib.md5(f"{file_base}__{rec_id}".encode()).hexdigest()[:12]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="A single .jsonl file OR a folder of .jsonl chunk files")
    ap.add_argument("--work", default="/tmp/chunk-regen", help="Working dir for manifest/inputs/RULES")
    ap.add_argument("--out", default=None, help="Output dir for regenerated summaries (default: <src>/_outputs_v2)")
    ap.add_argument("--rules", default=None, help="Path to a RULES template (default: skill's assets/RULES_TEMPLATE.md)")
    ap.add_argument("--window", type=int, default=8100,
                    help="Context window in tokens for instruction+input+output (default 8100). "
                         "Sets a per-record output char budget so records fit without truncation.")
    ap.add_argument("--reserve", type=int, default=120,
                    help="Tokens reserved for instruction + format markers + EOS (default 120)")
    ap.add_argument("--input-key", default="input")
    ap.add_argument("--id-key", default="id")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    work = os.path.abspath(args.work)
    # --src may be a single .jsonl file OR a folder of them.
    if os.path.isfile(src):
        base_dir = os.path.dirname(src)
        jsonl_files = [os.path.basename(src)]
    else:
        base_dir = src
        jsonl_files = sorted(f for f in os.listdir(src) if f.endswith(".jsonl"))
    out = os.path.abspath(args.out) if args.out else os.path.join(base_dir, "_outputs_v2")
    inputs_dir = os.path.join(work, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(out, exist_ok=True)

    rules_src = args.rules or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "RULES_TEMPLATE.md")
    shutil.copyfile(rules_src, os.path.join(work, "RULES.md"))

    count, exact = get_counter()
    if not jsonl_files:
        sys.exit(f"No .jsonl files found in {src}")

    manifest, seen, dups, tight = [], set(), 0, 0
    for fname in jsonl_files:
        stem = os.path.splitext(fname)[0]
        with open(os.path.join(base_dir, fname), encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    sys.exit(f"{fname}:{ln} invalid JSON: {e}")
                rid = rec.get(args.id_key, ln)
                text = rec.get(args.input_key, "")
                if not isinstance(text, str):
                    text = str(text)
                k = key_for(fname, rid)
                if k in seen:
                    dups += 1
                    k = key_for(fname, f"{rid}#{ln}")
                seen.add(k)
                ip = os.path.join(inputs_dir, f"{k}.txt")
                with open(ip, "w", encoding="utf-8") as w:
                    w.write(text)
                entry = {
                    "key": k, "file": fname, "id": rid, "stem": stem,
                    "input_path": ip, "input_chars": len(text),
                    "output_path": os.path.join(out, f"{stem}__{rid}.txt"),
                }
                if args.window:
                    budget_tok = args.window - count(text) - args.reserve
                    entry["out_char_budget"] = max(300, int(budget_tok * CPT_OUTPUT))
                    if budget_tok < 600:
                        tight += 1
                manifest.append(entry)

    with open(os.path.join(work, "manifest.json"), "w", encoding="utf-8") as w:
        json.dump(manifest, w, ensure_ascii=False, indent=1)

    print(f"Source files : {len(jsonl_files)}")
    print(f"Records       : {len(manifest)}")
    if dups:
        print(f"Disambiguated duplicate keys: {dups}")
    print(f"Token counter : {'tiktoken cl100k_base (exact-ish)' if exact else 'chars/token heuristic (tiktoken not installed)'}")
    if args.window:
        print(f"Window        : {args.window} tok  -> per-record out_char_budget written to manifest")
        if tight:
            print(f"  WARNING: {tight} record(s) have a very small output budget (<600 tok) — their inputs nearly fill the window.")
    print(f"Work dir      : {work}")
    print(f"Output dir    : {out}")

if __name__ == "__main__":
    main()
