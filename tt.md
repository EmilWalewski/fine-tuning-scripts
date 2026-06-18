Oto pełna lista zmian — co i gdzie.

## Zmienione / nadpisane

| Plik | Zmiana |
|---|---|
| [_regen_work/manifest.json](_regen_work/manifest.json) | **(a)** przepisany prefiks ścieżek `/home/atlas/python/traning/fine-tuning-scripts` → `/Users/ewalewski/python/fine-tuning-scripts` (930 wystąpień; teraz 0 odwołań do `/home/atlas`). **(b)** przeliczone `out_char_budget` dla wszystkich 465 rekordów pod okno 5120 (rundy: 5120/margines150 → cel 4900 → cel 4700 dla niedobitków). |
| [_ready_outputs/](_ready_outputs/)`dataset__1.txt … dataset__465.txt` | wszystkie 465 obecne, 0 pustych, wszystkie ≤ 5120 tok. W tej sesji: dogenerowane brakujące 245–465, potem regenerowane przekraczające limit (210 → 84 → 6). Wg gita **332 z 465** różni się od ostatniego commita. |

## Utworzone (nowe pliki pomocnicze/backupy w `_regen_work/`)
- `manifest.json.home-bak` — oryginał z linuksowymi ścieżkami (przed naprawą)
- `manifest.json.win8100-bak` — po naprawie ścieżek, budżety pod 8100 (przed re-fitem 5120)
- `manifest.json.win5120-safe150-bak` — po pierwszym przeliczeniu pod 5120
- `leak_report.txt`, `leak_report2.txt` — raporty kontroli przecieków (verbose)
- `refit2.json` — lista robocza 84 rekordów do re-fitu

## Pamięć (poza projektem, `~/.claude/projects/-Users-ewalewski-python-fine-tuning-scripts/memory/`)
- `stateless-regen-run.md` — stan zadania + dwie pułapki (ścieżka `/home/atlas`→`/Users`; przekraczanie limitu znaków przez subagentów i obejście promptem)
- `MEMORY.md` — wpis indeksowy

## Nietknięte (zweryfikowane)
- [dataset.jsonl](dataset.jsonl) — **źródło, bez zmian** (3 MiB, mtime 18 cze)
- [_regen_work/inputs/](_regen_work/inputs/) — 930 plików wejściowych, bez zmian
- [_regen_work/RULES.md](_regen_work/RULES.md) — bez zmian
- `stateless-chunk-regen/scripts/*` — skrypty bez zmian
- **Merge (Stage 4) nieuruchomiony** → `dataset.regenerated.jsonl` jeszcze nie istnieje

Jedyne nieodwracalne operacje to nadpisania w `_ready_outputs/` (regeneracje) i edycje `manifest.json` — ale każdy stan manifestu mam w backupach `.bak`, a `_ready_outputs` budowane było nie-destrukcyjnie (usuwane i odtwarzane tylko pliki przekraczające limit). Źródłowy `dataset.jsonl` ani katalog `inputs/` nie zostały dotknięte.

Nietknięte (zweryfikowane)
dataset.jsonl — źródło, bez zmian (3 MiB, mtime 18 cze)
_regen_work/inputs/ — 930 plików wejściowych, bez zmian
_regen_work/RULES.md — bez zmian
stateless-chunk-regen/scripts/* — skrypty bez zmian
Merge (Stage 4) nieuruchomiony → dataset.regenerated.jsonl jeszcze nie istnieje
Jedyne nieodwracalne operacje to nadpisania w _ready_outputs/ (regeneracje) i edycje manifest.json — ale każdy stan manifestu mam w backupach .bak, a _ready_outputs budowane było nie-destrukcyjnie (usuwane i odtwarzane tylko pliki przekraczające limit). Źródłowy dataset.jsonl ani katalog inputs/ nie zostały dotknięte.