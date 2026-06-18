# Stateless Chunk Regen - AI Handoff

This folder contains a local Codex skill for rebuilding and reviewing
per-chunk outputs in an Alpaca-style JSONL dataset:

```json
{"id": 1, "instruction": "...", "input": "...", "output": "..."}
```

Use it when outputs must be generated or audited independently for each chunk,
with no cross-chunk context bleed.

## First Step

Read `SKILL.md` before doing any work. It is the source of truth for the
stateless regeneration workflow and isolation rules.

If the task is only to review existing outputs, do not regenerate anything
unless explicitly asked.

## Directory Layout

Expected files in the parent project directory:

- `dataset.jsonl` - final merged dataset with `instruction`, `input`, `output`
- `_regen_work/manifest.json` - chunk manifest
- `_regen_work/inputs/` - one input text file per chunk
- `_regen_work/RULES.md` - per-chunk analyst contract used for generation
- `_ready_outputs/` - one generated output text file per chunk

Skill files:

- `SKILL.md` - full workflow
- `assets/RULES_TEMPLATE.md` - default output-generation prompt template
- `scripts/build_manifest.py` - creates manifest and per-chunk input files
- `scripts/remaining.py` - lists missing or empty output files
- `scripts/measure_tokens.py` - checks whether records fit a token window
- `scripts/leak_qa.py` - detects likely numeric leakage between chunks
- `scripts/merge_to_jsonl.py` - merges generated outputs back into JSONL
- `scripts/path_profiles.py` - maps paths between Linux and macOS

## Path Profiles

The scripts support portable manifest paths with:

```bash
--path-profile auto
--path-profile linux
--path-profile macos
--path-profile none
```

Profiles map project-root paths between:

- Linux: `/home/atlas/python/traning/fine-tuning-scripts`
- macOS: `/Users/ewalewski/python/fine-tuning-scripts`

Use `--path-profile auto` by default. It maps manifest paths to the current OS.
Use `none` only when you want to trust paths exactly as written in the manifest.

## QA Existing Outputs

Run from the parent project directory, for example:

```bash
cd /home/atlas/python/traning/fine-tuning-scripts
```

Check whether every chunk has a non-empty output:

```bash
python3 ./stateless-chunk-regen/scripts/remaining.py \
  --manifest ./_regen_work/manifest.json \
  --path-profile auto
```

Expected final state:

```text
# done N/N | remaining 0
```

Check context-window fit, for example 5120 tokens:

```bash
python3 ./stateless-chunk-regen/scripts/measure_tokens.py \
  --manifest ./_regen_work/manifest.json \
  --src . \
  --window 5120 \
  --path-profile auto
```

Expected final state:

```text
All records fit the window.
```

Check for likely cross-chunk numeric contamination:

```bash
python3 ./stateless-chunk-regen/scripts/leak_qa.py \
  --manifest ./_regen_work/manifest.json \
  --path-profile auto
```

Expected final state:

```text
CLEAN - no cross-chunk numeric contamination detected.
```

If leaks are reported, review them manually. A flagged number is strong
evidence, not absolute proof.

## Manual Review Checklist

Sample several records from `dataset.jsonl` and compare each `output` with its
own `input`.

Check that:

- the output only uses facts present in that input
- financial figures, dates, company names, and table values match the input
- the output does not summarize a neighboring chunk
- the output is not empty unless `INSUFFICIENT_DATA` is justified
- the output is concise enough for the configured token window
- broken or low-value inputs are handled conservatively

For table-heavy chunks, verify that the summary does not invent relationships
between columns when the table itself is malformed or ambiguous.

## Regenerating Missing Outputs

Only regenerate when the user explicitly asks.

List missing chunks:

```bash
python3 ./stateless-chunk-regen/scripts/remaining.py \
  --manifest ./_regen_work/manifest.json \
  --limit 15 \
  --path-profile auto
```

For each printed line:

```text
INPUT || OUTPUT || BUDGET=N
```

start one isolated subtask/subagent. It may read only:

- `_regen_work/RULES.md`
- that single `INPUT` file

It must write only the requested `OUTPUT` file. It must not read other chunks.
If `BUDGET=N` is present, the output must not exceed `N` characters.

## Merging Outputs

Only merge after QA passes:

```bash
python3 ./stateless-chunk-regen/scripts/merge_to_jsonl.py \
  --manifest ./_regen_work/manifest.json \
  --src ./dataset.jsonl \
  --out ./dataset-ready.jsonl \
  --path-profile auto
```

This preserves original `instruction` and `input`, and replaces only `output`.

## Important Rules

- Do not delete `_ready_outputs` manually.
- Do not regenerate completed outputs unless the user asks.
- `INSUFFICIENT_DATA` is a valid output for chunks without useful factual
  content.
- Treat `remaining.py` as the source of truth for completion.
- Keep chunk processing stateless: one chunk, one clean context, no neighboring
  files.
