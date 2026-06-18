#!/usr/bin/env python3
"""
merge_to_jsonl.py — Final stage: fold regenerated summaries back into a dataset.

Reads the manifest, re-reads each ORIGINAL source record (to preserve its
`instruction` and `input` verbatim), and pairs it with the freshly regenerated
`output` from the output directory. Writes a new JSONL file.

This is non-destructive by design: it writes a brand-new file and never edits
the source folder. Point your training script at the new file once you've
confirmed the QA looks good.

By default records whose output is missing or empty are skipped (with a
warning); use --keep-missing to emit them with an empty output instead.
"""
import argparse, json, os

from path_profiles import add_path_profile_arg, mapped_entry

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--src", required=True, help="Original .jsonl file OR folder of them (for instruction/input)")
    ap.add_argument("--out", required=True, help="Path to write the merged .jsonl")
    ap.add_argument("--id-key", default="id")
    ap.add_argument("--keep-missing", action="store_true", help="Emit records with empty output instead of skipping")
    add_path_profile_arg(ap)
    args = ap.parse_args()

    manifest = json.load(open(args.manifest, encoding="utf-8"))

    # Index original records by (file, str(id)). --src may be a file or folder.
    src = os.path.abspath(args.src)
    if os.path.isfile(src):
        base_dir, files = os.path.dirname(src), [os.path.basename(src)]
    else:
        base_dir = src
        files = sorted(f for f in os.listdir(src) if f.endswith(".jsonl"))
    orig = {}
    for fname in files:
        with open(os.path.join(base_dir, fname), encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rid = rec.get(args.id_key, ln)
                orig[(fname, str(rid))] = rec

    written, skipped, missing = 0, 0, []
    with open(args.out, "w", encoding="utf-8") as w:
        for m in manifest:
            m = mapped_entry(m, args.path_profile)
            op = m["output_path"]
            has = os.path.exists(op) and os.path.getsize(op) > 0
            if not has:
                missing.append(op.rsplit("/", 1)[-1])
                if not args.keep_missing:
                    skipped += 1
                    continue
            out_text = open(op, encoding="utf-8").read().strip() if has else ""
            base = orig.get((m["file"], str(m["id"])), {})
            rec = {
                args.id_key: m["id"],
                "instruction": base.get("instruction", ""),
                "input": base.get("input", ""),
                "output": out_text,
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"Written : {written} records -> {args.out}")
    if skipped:
        print(f"Skipped : {skipped} (missing/empty output)")
    if missing:
        print(f"Missing outputs ({len(missing)}): {', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else ""))

if __name__ == "__main__":
    main()
