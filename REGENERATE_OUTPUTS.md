# Regeneracja outputów na innym komputerze (Codex CLI)

Ten folder zawiera wszystko, czego potrzeba, żeby **wygenerować outputy (streszczenia)**
dla chunków z `dataset.jsonl` — bezstanowo, jeden izolowany chunk = jedno streszczenie,
bez przecieków między chunkami.

Skill leży w `./stateless-chunk-regen/` i **podróżuje razem z tym folderem** — nic nie
trzeba „instalować" globalnie.

---

## 0. Jak „zainstalować" skilla na Codex CLI

Codex CLI **nie ma** systemu skili jak Claude Code (SKILL.md / pluginy). I nie musi —
ten skill to po prostu **zwykłe skrypty Pythona + plik instrukcji**. „Instalacja" =
mieć ten folder w projekcie i wskazać go agentowi. Konkretnie:

- Trzymaj `stateless-chunk-regen/` w repo (już jest).
- W prompt'cie każ Codexowi przeczytać `stateless-chunk-regen/SKILL.md` i uruchamiać
  skrypty z `stateless-chunk-regen/scripts/` przez `python3`.

(Opcjonalnie: jeśli Twoja wersja Codexa czyta `AGENTS.md`, możesz dopisać tam jedno
zdanie „Do regeneracji outputów użyj stateless-chunk-regen/SKILL.md" — ale prompt
poniżej i tak jest samowystarczalny.)

---

## 1. Wymagania (jednorazowo, na nowym komputerze)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install tiktoken                       # do dokładnych budżetów okna (opcjonalne, ale zalecane)
# do (ewentualnej) ekstrakcji z PDF:
pip install pymupdf4llm pymupdf langchain-text-splitters
```

Skrypty skilla są stdlib-only **poza** `tiktoken` (a i bez niego działają — wtedy
budżet liczony heurystyką znaki/token).

---

## 2. Skąd `dataset.jsonl`

`dataset.jsonl` = chunki ze **wszystkich** raportów **zmergowane w jeden plik**
(pola `id` / `instruction` / `input` / `output`, gdzie `output` jest pusty — to go
generujemy). Jeśli go nie masz, zrób go z `extract6.py`:

```bash
# 1) ekstrakcja wszystkich PDF z materials/ -> per-dokument *-clean.jsonl
python3 -c "import extract6; extract6.process_pdf_to_clean_dataset(pdf_dir='materials', out_dir='_extract_out', max_tokens=3700, min_characters=800, max_garbled_risk=0.05)"
# 2) merge w jeden plik
cat _extract_out/*.jsonl > dataset.jsonl
```

`extract6.py` ma już **dwustopniową ekstrakcję** (Tier-1 + pozycyjny rescue zlanych
tabel + guard) — czyli inputy są czyste, bez „rozsypanych" tabel.

---

## 3. Prompt dla Codexa (skopiuj w całości)

> Uruchom Codex CLI w katalogu `fine-tuning-scripts` i wklej poniższy prompt.
> Zakłada `dataset.jsonl` w tym katalogu i okno kontekstu **5120**.

````
Masz wygenerować outputy (streszczenia) dla chunków z ./dataset.jsonl, bezstanowo i z
pełną izolacją (każdy chunk niezależnie, bez wpływu innych chunków). Logika jest w
skilu ./stateless-chunk-regen/ — najpierw przeczytaj ./stateless-chunk-regen/SKILL.md.

KROK 1 — zbuduj manifest i pliki wejściowe (jednorazowo):
  python3 ./stateless-chunk-regen/scripts/build_manifest.py \
    --src ./dataset.jsonl \
    --work ./_regen_work \
    --out ./_ready_outputs \
    --window 5120
  To tworzy ./_regen_work/manifest.json, ./_regen_work/inputs/<klucz>.txt (po jednym
  pliku na chunk) oraz ./_regen_work/RULES.md (kontrakt analityka — pełne reguły).

KROK 2 — pętla generowania (powtarzaj aż zostanie 0):
  a) Zapytaj, co zostało (drukuje linie INPUT || OUTPUT || BUDGET=N tylko dla chunków,
     których plik wyjścia nie istnieje/jest pusty):
       python3 ./stateless-chunk-regen/scripts/remaining.py --manifest ./_regen_work/manifest.json --limit 15
  b) Dla KAŻDEJ zwróconej linii wygeneruj streszczenie jako NIEZALEŻNE zadanie:
       - przeczytaj instrukcje z ./_regen_work/RULES.md ORAZ TYLKO ten jeden plik INPUT,
       - napisz streszczenie do pliku OUTPUT (samą prozę; albo token INSUFFICIENT_DATA),
       - twardy limit: streszczenie nie może przekroczyć N znaków (BUDGET=N),
       - NIE czytaj żadnego innego pliku/chunku; nie pozwól, by treść innych chunków
         wpłynęła na ten — każdy chunk to czysta karta.
     IZOLACJA jest kluczowa (inaczej liczby z jednego raportu przeciekną do drugiego).
     Jeśli Twój runtime potrafi odpalać równoległe pod-zadania/sub-agentów — odpal ~10-15
     na rundę, po jednym na chunk (najlepsza izolacja). Jeśli nie — rób po jednym chunku
     na raz, za każdym razem zaczynając od zera.
  c) Wróć do (a). Powtarzaj aż remaining.py pokaże "remaining 0".

KROK 3 — scal w finalny zbiór:
  python3 ./stateless-chunk-regen/scripts/merge_to_jsonl.py \
    --manifest ./_regen_work/manifest.json \
    --src ./dataset.jsonl \
    --out ./dataset-ready.jsonl
  (zachowa oryginalne instruction+input, wstawi wygenerowane output)

ZASADY:
- Idempotentnie: remaining.py liczy braki z dysku — po przerwaniu/limicie po prostu
  wznawiasz pętlę, nic gotowego się nie powtórzy.
- Nigdy nie czyść ./_ready_outputs ręcznie.
- INSUFFICIENT_DATA to poprawny output (chunk bez treści), nie błąd.
````

---

## 4. Co dostajesz na końcu

`dataset-ready.jsonl` — komplet rekordów z czystymi, izolowanymi streszczeniami,
każdy ≤ 5120 tokenów (input+output). Gotowy do treningu (`fifth-unsloth_training.py`
czyta `dataset.jsonl`/`dataset2.jsonl` — podmień ścieżkę albo zmień nazwę).

## 5. (Opcjonalnie) QA przed treningiem
```bash
# kontaminacja cross-chunk (liczby z cudzego inputu):
python3 ./stateless-chunk-regen/scripts/leak_qa.py --manifest ./_regen_work/manifest.json
# długości (czy wszystko mieści się w oknie):
python3 ./stateless-chunk-regen/scripts/measure_tokens.py --manifest ./_regen_work/manifest.json --src ./dataset.jsonl --window 5120
```

---

### Uwaga o izolacji (dlaczego to ważne)
Pierwotny problem datasetu to były **przecieki między chunkami** (output zawierał liczby
z cudzego inputu), bo generował go jeden stanowy agent z handoffami. Skill rozwiązuje to
przez izolację: jeden chunk = jeden świeży kontekst, widzi tylko swój input. Na Codexie:
jeśli masz pod-agentów — używaj ich (gwarantowana izolacja). Jeśli generujesz sekwencyjnie
w jednym kontekście, **rygorystycznie** trzymaj się reguły „czysta karta na każdy chunk",
bo inaczej przeciek wróci. Dla 100% pewności izolacji najlepsze są oddzielne wywołania
(pod-agent lub osobne wywołanie modelu na chunk).
```
