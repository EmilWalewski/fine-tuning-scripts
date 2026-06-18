import os
import json
import re
import tiktoken
import pymupdf4llm
import fitz  # PyMuPDF — for Tier-2 positional table rescue
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

INSTRUCTION_TEMPLATE = (
    "You are a Senior Strategic Corporate Analyst. Summarize the report excerpt into a high-density English executive summary.\n"
    "STRICT COMMANDS:\n"
    "1. SOURCE FIDELITY: Use ONLY the provided text. Never use external knowledge or general history.\n"
    "2. FINANCIAL ACCURACY: Extract all material figures (revenue, costs, profit). Normalize units: convert 'tys.' (thousands) to M (millions). Example: 4,156,476k PLN -> 4,156.5M PLN. Always state the currency.\n"
    "3. STRATEGIC EVENTS: Explicitly include milestones like acquisitions (zakup), dividends, or board changes.\n"
    "4. TELEGRAPHIC STYLE: Write a continuous, professional narrative. Skip introductions, transitions, and bullet points. Use dense noun phrases.\n"
    "5. NO HALLUCINATION: If a specific data point (e.g. effective tax rate) is not in the text, omit it. Do not invent boilerplate or advisory names."
)

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# ---------------------------------------------------------------------------
# FIX 1 – Page-header / footer removal
#
# pymupdf4llm appends repeating page headers/footers as bold Markdown lines
# that interrupt running sentences.  The pattern is:
#
#   ...end of sentence fragment  \n       ← trailing two-space soft-break
#   **DOCUMENT TITLE LINE 1**  \n
#   **DOCUMENT TITLE LINE 2**  \n         ← optional 2nd bold line
#   <page number>  \n                      ← standalone 1-4 digit integer
#   rest of sentence continues...
#
# Both patterns (soft-break trailing and bare newline) are matched and the
# whole block is collapsed to a single space so surrounding text stays joined.
# ---------------------------------------------------------------------------

# Pattern 1: with soft-break (two trailing spaces before \n)
_PAGE_HEADER_SOFT = re.compile(
    r"  \n"                              # pymupdf4llm soft-break before block
    r"(?:\*\*[^\n]+\*\*  \n){1,4}"      # 1-4 bold header lines (soft-break)
    r"\d{1,4}  \n",                      # standalone page number line
    re.MULTILINE,
)

# Pattern 2: bare newlines (fallback for other PDF variants)
_PAGE_HEADER_BARE = re.compile(
    r"\n(?:\*\*[^\n]+\*\*\s*\n){1,4}\d{1,4}\s*\n",
    re.MULTILINE,
)

# Pattern 3: page-number + URL footer interrupting text, e.g.  "43\nxtb.com\n"
# (both orders). The number+domain combo is an unambiguous running footer.
_PAGE_FOOTER_NUM_URL = re.compile(
    r"\n\s*\d{1,4}\s*\n\s*(?:www\.)?[a-z0-9-]+\.(?:com|pl|eu|net|org)\s*\n",
    re.MULTILINE | re.IGNORECASE,
)
_PAGE_FOOTER_URL_NUM = re.compile(
    r"\n\s*(?:www\.)?[a-z0-9-]+\.(?:com|pl|eu|net|org)\s*\n\s*\d{1,4}\s*\n",
    re.MULTILINE | re.IGNORECASE,
)
# standalone domain-only line (running footer like "xtb.com" on its own line)
_BARE_URL_LINE = re.compile(
    r"\n\s*(?:www\.)?[a-z0-9-]{2,}\.(?:com|pl|eu|net|org)\s*\n",
    re.MULTILINE | re.IGNORECASE,
)


def remove_page_headers(text: str) -> str:
    """Strip repeating page headers / footers that interrupt running text."""
    cleaned = _PAGE_HEADER_SOFT.sub(" ", text)
    cleaned = _PAGE_HEADER_BARE.sub("\n", cleaned)
    cleaned = _PAGE_FOOTER_NUM_URL.sub(" ", cleaned)   # FIX 1b (issue 3 / page artifacts)
    cleaned = _PAGE_FOOTER_URL_NUM.sub(" ", cleaned)
    cleaned = _BARE_URL_LINE.sub("\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# FIX 2 – <br> tag handling in Markdown tables
#
# pymupdf4llm produces multi-line cell content as   text<br>more text   inside
# pipe-table cells.  Three cases are handled:
#
#  A) Bold/italic header cells:  **Title**<br>**Subtitle**  → **Title Subtitle**
#  B) Date stacks:  30.06.2025<br>31.12.2024  →  30.06.2025 / 31.12.2024
#  C) Bullet-list cells (▪<br>text): whole row converted to a plain paragraph
#  D) Default: replace <br> with a single space
# ---------------------------------------------------------------------------

# A <br>-stacked cell of NUMBERS/DATES (multi-period financial column collapsed
# into one Markdown cell). Joining these with a plain space flattens the periods
# ("296 209 1 306 985 ...") so the LLM can't tell where one figure ends — because
# spaces are also thousands separators. We join them with an explicit " ; " so
# each period's value stays an unambiguous, separate token.
def _seg_is_numeric(s: str) -> bool:
    s2 = s.strip().strip("_*").strip()
    if not any(c.isdigit() for c in s2):
        return False
    return bool(re.fullmatch(r"\(?-?[\d\s .,%]+\)?", s2))


def _is_value_stack(cell: str) -> bool:
    segs = [s.strip() for s in cell.split("<br>") if s.strip()]
    if len(segs) < 2:
        return False
    numeric = sum(1 for s in segs if _seg_is_numeric(s))
    return numeric >= max(2, int(len(segs) * 0.6))


# --- Expand a numeric/date <br>-stack cell into SEPARATE Markdown columns ---
# Better than joining with " ; " inside one cell: each period's value lands in
# its own | column, so the table is valid Markdown AND value↔header alignment
# is explicit. The <br> already marks the boundaries, so no PDF geometry needed.
# Runs as its own pass (expand_value_columns) BEFORE fix_br_in_tables; the block
# is then width-normalised in sanitize_tables.
def _expand_row_value_columns(line: str) -> str:
    if not _is_table_row(line):
        return line
    cells = line.strip().strip("|").split("|")
    out = []
    for c in cells:
        c2 = c.strip()
        if "<br>" in c2 and _is_value_stack(c2):
            out.extend(p.strip() for p in c2.split("<br>"))
        else:
            out.append(c2)
    return "| " + " | ".join(out) + " |"


def expand_value_columns(text: str) -> str:
    return "\n".join(
        _expand_row_value_columns(l) if ("<br>" in l and _is_table_row(l)) else l
        for l in text.split("\n")
    )


def _replace_br_in_cell(cell: str) -> str:
    if not cell:
        return cell
    # Bullet cell – handled at row level
    if re.match(r"[▪•]\s*<br>", cell):
        return cell
    # Fallback: any numeric/date stack that survived expansion → explicit ' ; '
    # (never a plain space — that flattens figures).
    if "<br>" in cell and _is_value_stack(cell):
        return re.sub(r"\s*<br>\s*", " ; ", cell)
    # Bold multi-line header: **A**<br>**B**  →  **A B**
    if cell.startswith("**") and "<br>" in cell:
        return re.sub(r"\*\*\s*<br>\s*\*\*", " ", cell).replace("<br>", " ")
    # Italic header
    if cell.startswith("_") and "<br>" in cell:
        return cell.replace("<br>", " ")
    # Date stacks (single-line start) – fallback if not caught as value stack
    if re.match(r"\d{2}\.\d{2}\.\d{4}", cell):
        return cell.replace("<br>", " ; ")
    # Default
    return cell.replace("<br>", " ")


def _row_has_bullet_br(row: str) -> bool:
    return bool(re.search(r"[▪•]\s*<br>", row))


def _convert_bullet_row_to_prose(row: str) -> str:
    """Turn a table row with ▪<br> cells into a readable bullet paragraph."""
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    bullets = []
    for cell in cells:
        if not cell:
            continue
        parts = [p.strip().lstrip("▪•").strip() for p in cell.split("<br>")]
        text = " ".join(p for p in parts if p)
        if text:
            bullets.append("• " + text)
    return "\n".join(bullets) if bullets else ""


def fix_br_in_tables(text: str) -> str:
    """Clean up <br> tags inside Markdown table cells."""
    lines = text.split("\n")
    result = []
    for line in lines:
        if "|" in line and "<br>" in line:
            if _row_has_bullet_br(line):
                prose = _convert_bullet_row_to_prose(line)
                if prose:
                    result.append(prose)
            else:
                leading  = "|" if line.lstrip().startswith("|") else ""
                trailing = "|" if line.rstrip().endswith("|") else ""
                inner = line.strip().strip("|")
                cells = inner.split("|")
                result.append(leading + "|".join(_replace_br_in_cell(c) for c in cells) + trailing)
        else:
            result.append(line)
    return "\n".join(result)


# ---------------------------------------------------------------------------
# FIX 2b – Table-block sanitizer  (tt.md issues 2 & 3)
#
# pymupdf4llm, breaking tables across page boundaries, produces two defects:
#   (2) injected separator lines mid-table, sometimes with a different column
#       count than the real header (e.g. 4 → 8) — destroys the table grid.
#   (3) a logical row whose label word-wraps in the PDF is emitted as SEVERAL
#       Markdown rows; the wrap fragments are "label-only" rows (first cell has
#       text, all other cells empty: "|w formie akcji|||||"). The figure ends up
#       on one physical row while its full label is scattered across neighbours.
#
# This pass works per contiguous table block:
#   • keeps only the first separator, drops duplicates and column-count mismatches
#   • merges label-only fragments into the adjacent data row:
#       – fragment starting UPPERCASE = start of a wrapped entry → prepend to the
#         NEXT data row
#       – fragment starting lowercase  = continuation            → append to the
#         PREVIOUS data row
# A cell containing only "-" is a real value (nil), NOT empty.
# ---------------------------------------------------------------------------

def _is_sep_line(line: str) -> bool:
    s = line.replace(" ", "")
    return bool(re.fullmatch(r"\|(?::?-+:?\|)+", s))


def _is_table_row(line: str) -> bool:
    st = line.strip()
    return st.startswith("|") and "|" in st[1:]


def _split_cells(line: str):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _join_cells(cells) -> str:
    return "| " + " | ".join(cells) + " |"


def _process_table_block(block):
    # Drop ALL separators up front; a single correct one is re-inserted after
    # width-normalisation (this also removes injected mid-table separators).
    rows = [l for l in block if not _is_sep_line(l)]

    # ── merge word-wrapped label-only fragments into data rows ────────────────
    def _append_backward(txt):
        pc = _split_cells(result[last_data])
        pc[0] = f"{pc[0]} {txt}".strip()
        result[last_data] = _join_cells(pc)

    result, pending_fwd, last_data = [], None, None
    for l in rows:
        cells = _split_cells(l)
        multi_empty = len(cells) > 1 and all(c == "" for c in cells[1:])
        # single-cell row starting lowercase = unambiguous wrap continuation.
        # (uppercase single-cell rows may be sub-headers, e.g. "AKTYWA" — leave them.)
        single_cont = len(cells) == 1 and cells[0][:1].islower()

        if multi_empty:
            txt = cells[0]
            if not txt:
                continue                          # fully empty row → drop
            if txt[:1].isupper() or last_data is None:
                pending_fwd = f"{pending_fwd} {txt}".strip() if pending_fwd else txt
            else:
                _append_backward(txt)
            continue
        if single_cont and last_data is not None and not pending_fwd:
            _append_backward(cells[0]); continue

        if pending_fwd:
            cells[0] = f"{pending_fwd} {cells[0]}".strip()
            pending_fwd = None
        result.append(_join_cells(cells))
        last_data = len(result) - 1
    if pending_fwd:
        if last_data is not None:
            _append_backward(pending_fwd)
        else:
            result.append(_join_cells([pending_fwd]))

    # ── normalise column widths + re-insert ONE valid separator ───────────────
    if not result:
        return result
    counts = [len(_split_cells(r)) for r in result]
    # target width = the DATA width (most common count), not the max — otherwise a
    # single messy multi-fragment header row inflates the whole table with empties.
    width = max(set(counts), key=counts.count)
    grid = []
    for r in result:
        c = _split_cells(r)
        if len(c) > width:          # over-wide row (usually a split header) → fold overflow into last cell
            c = c[:width - 1] + [" ".join(x for x in c[width - 1:] if x).strip()]
        elif len(c) < width:
            c = c + [""] * (width - len(c))
        grid.append(c)
    norm = [_join_cells(c) for c in grid]
    if width >= 2:                                # only real tables get a separator
        sep = "| " + " | ".join(["---"] * width) + " |"
        norm = [norm[0], sep] + norm[1:]
    return norm


def clean_loose_semicolons(text: str) -> str:
    """Tidy ' ; ' artefacts on NON-table lines (chart residue / stray punctuation).

    The ' ; ' separator is only meaningful inside table value cells. On a plain
    line it is leftover noise: a pure-number line is bar-chart residue (drop it);
    prose with a stray '; ' just gets the marker collapsed to a space.
    """
    out = []
    for ln in text.split("\n"):
        if " ; " in ln and not ln.strip().startswith("|"):
            toks = ln.split()
            numish = sum(1 for t in toks if re.fullmatch(r"\(?-?[\d.,%)]+;?", t))
            if toks and numish / len(toks) >= 0.6:
                continue                       # chart-residue number line → drop
            ln = ln.replace(" ; ", " ")        # prose stray ';' → space
        out.append(ln)
    return "\n".join(out)


def sanitize_tables(text: str) -> str:
    """Repair injected separators and word-wrapped label rows (tt.md 2 & 3)."""
    lines = text.split("\n")
    out, i, n = [], 0, len(lines)
    while i < n:
        if not _is_table_row(lines[i]):
            out.append(lines[i]); i += 1; continue
        block = []
        while i < n and _is_table_row(lines[i]):
            block.append(lines[i]); i += 1
        out.extend(_process_table_block(block))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# FIX 3 – Garbled-text quality flag
#
# Some PDFs use non-standard font encodings that cause pymupdf4llm to drop
# random vowels, producing unreadable OCR noise (e.g. "fiaswymi" instead of
# "finansowymi").  Reliable per-word detection is impossible without the
# original PDF because garbled and legitimate Polish words have overlapping
# consonant-ratio distributions.
#
# Strategy: compute a "garbled_risk" score (0–1) per chunk and embed it in
# the JSONL metadata.  Downstream pipelines can use this flag to:
#   • Skip the chunk during fine-tuning
#   • Route it to human review
#   • Lower its sample weight
#
# Score definition:
#   fraction of prose tokens (≥5 chars, non-markdown) whose consonant ratio
#   is ≥ 0.85 (well above the good-Polish noise floor of ~0.75).
# ---------------------------------------------------------------------------

_POLISH_VOWELS = set("aeiouąęóyAEIOUĄĘÓY")
_IGNORE_GARBLE = re.compile(
    r"^[-|_=*#>\[\](){}/\\]+$"   # markdown / separator
    r"|^\d"                        # starts with digit
    r"|^https?://"                 # URL
    r"|^[A-Z]{2,6}$"              # all-caps abbreviation (MSSF, ESG, PLN…)
)
# Known company abbreviations that look garbled but aren't
_KNOWN_ABBREVS = re.compile(r"Dvlpmt|XTB|GPW|MSSF|MSR|IFRS|ESG|PLN|EUR|USD|CEO|CFO")


def _token_consonant_ratio(token: str) -> float:
    """Fraction of alphabetic chars that are consonants (Polish-aware)."""
    # strip HTML and punctuation
    clean = re.sub(r"<br>", "", token)
    clean = re.sub(r"^[*_.,;:!?'\"()\[\]|<>]+|[*_.,;:!?'\"()\[\]|<>]+$", "", clean)
    letters = [c for c in clean if c.isalpha()]
    if len(letters) < 5:
        return 0.0
    consonants = sum(1 for c in letters if c not in _POLISH_VOWELS)
    return consonants / len(letters)


def compute_garbled_score(text: str) -> float:
    """
    Return 0–1 float.  Values ≥ 0.05 indicate likely OCR corruption.
    Only prose tokens (non-table lines) are analysed.
    """
    lines = text.split("\n")
    prose = "\n".join(l for l in lines if not l.strip().startswith("|"))
    tokens = prose.split()
    qualifying = [
        t for t in tokens
        if not _IGNORE_GARBLE.match(t)
        and not _KNOWN_ABBREVS.search(t)
        and len(t) >= 5
        and "<br>" not in t
    ]
    if len(qualifying) < 8:
        return 0.0
    high_consonant = sum(1 for t in qualifying if _token_consonant_ratio(t) >= 0.85)
    return high_consonant / len(qualifying)


# ---------------------------------------------------------------------------
# Existing helpers (unchanged)
# ---------------------------------------------------------------------------

def is_toc_chunk(text: str) -> bool:
    lower = text.lower()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return False

    toc_marker = "spis treści" in lower or "contents" in lower
    leader_lines = [
        l for l in lines
        if re.search(r"\.{6,}\s*\d{1,4}(?:\s*\*\*)?\s*\|?\s*$", l)
    ]

    if toc_marker and len(leader_lines) >= 3:
        return True
    return len(leader_lines) >= 8 and (len(leader_lines) / len(lines)) >= 0.25


def strip_toc_blocks(text: str) -> str:
    """Remove table-of-contents blocks while preserving any real report text.

    TOCs in PDFs usually appear as long dotted-leader ranges ending with page
    numbers. Some entries wrap across lines without dots, so when a chunk has a
    dense leader range we remove the whole range, not only the dotted lines.
    """
    lines = text.split("\n")
    leader_idxs = [
        i for i, l in enumerate(lines)
        if re.search(r"\.{6,}\s*\d{1,4}(?:\s*\*\*)?\s*\|?\s*$", l.strip())
    ]
    if len(leader_idxs) < 3:
        return text

    start, end = leader_idxs[0], leader_idxs[-1]
    while start > 0:
        prev = lines[start - 1].strip().lower()
        if "spis treści" in prev or "contents" in prev or re.fullmatch(r"\|?\s*-+\s*(?:\|\s*-+\s*)*\|?", prev):
            start -= 1
            continue
        break

    cleaned = lines[:start] + lines[end + 1:]
    cleaned_text = "\n".join(cleaned)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()
    return cleaned_text


def extract_table_header(text: str):
    lines = text.split("\n")
    for i in range(len(lines) - 1):
        line_clean = lines[i + 1].replace(" ", "")
        if "|" in lines[i] and ("|---" in line_clean or "|:---" in line_clean):
            return f"{lines[i]}\n{lines[i + 1]}"
    return None


def clean_markdown_artifacts(text: str) -> str:
    """Remove picture-omission markers and picture-text wrappers."""
    cleaned = re.sub(
        r"\*\*==> picture \[\d+ x \d+\] intentionally omitted <==\*\*", "", text
    )
    cleaned = re.sub(r"\*\*----- End of picture text ----- \*\*", "", cleaned)
    cleaned = re.sub(r"\*\*----- Start of picture text -----\*\*\s*<br>?\s*", "", cleaned)
    cleaned = re.sub(r"\*\*----- End of picture text -----\*\*\s*<br>?\s*", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def save_chunk_to_jsonl(parts, contexts, out_file_handle, chunk_id, max_garbled_risk=None):
    """Write a single clean JSONL entry with quality metadata.

    Returns True if written, False if skipped (garbled_risk above threshold).
    With max_garbled_risk set, chunks whose OCR-corruption score exceeds it are
    dropped (issue 4: dropped vowels) — excluding poison beats keeping it.
    """
    combined_text = "\n\n".join(parts)
    combined_context = " | ".join(sorted(list(contexts)))
    combined_text = strip_toc_blocks(combined_text)
    final_input = f"DOKUMENT SEKCJA: {combined_context}\n\n{combined_text}"

    if len(combined_text.strip()) < 100 or is_toc_chunk(combined_text):
        print(f"    [skip] chunk {chunk_id}: table of contents")
        return False

    garbled = compute_garbled_score(combined_text)
    if max_garbled_risk is not None and garbled > max_garbled_risk:
        print(f"    [skip] chunk {chunk_id}: garbled_risk={garbled:.4f} > {max_garbled_risk}")
        return False

    jsonl_row = {
        "id": chunk_id,
        "instruction": INSTRUCTION_TEMPLATE,
        "input": final_input,
        "output": "",
        "garbled_risk": round(garbled, 4),   # FIX 3: quality flag for downstream use
    }
    out_file_handle.write(json.dumps(jsonl_row, ensure_ascii=False) + "\n")
    return True


# ---------------------------------------------------------------------------
# FIX 5 – Two-tier table extraction with positional rescue (issue: jammed tables)
#
# pymupdf4llm sometimes renders dense, borderless financial tables catastrophically:
# row labels jammed into one cell and the numbers jammed into a SEPARATE row, so
# the figures cannot be mapped to their line items (e.g. Cognor balance sheet).
# The Tier-1 cleaning passes above don't help (there is no <br> to expand).
#
#   Tier 1 (cheap):  pymupdf4llm + cleaning passes — used for everything.
#   Detector:        _has_value_jam() on the cleaned text — fires only on real
#                    value-jams (>=4 financial figures crammed into one cell).
#   Tier 2 (costly): _positional_rows() — reconstruct the table from word x/y
#                    coordinates (PyMuPDF), which recovers label+aligned-columns.
#   Guard:           _validate_rescue() — accept the rescue ONLY if it looks like
#                    a real table (labelled rows, well-formed balanced numbers,
#                    consistent column count). On complex layouts (multi-level
#                    headers, KRUK/XTB-annual) positional reading scrambles the
#                    numbers; the guard rejects those so we NEVER emit garbage —
#                    the jammed table is simply dropped instead.
# ---------------------------------------------------------------------------

_FINVAL = re.compile(r"\(\s*\d[\d  ]*\d\s*\)|\d{1,3}(?:[  ]\d{3})+|\d{5,}")

def _has_value_jam(text: str, thr: int = 4) -> bool:
    """True if any table cell crams >= thr financial values (years excluded)."""
    for line in text.split("\n"):
        if not _is_table_row(line):
            continue
        for cell in line.split("|"):
            vals = [v for v in _FINVAL.findall(cell)
                    if not re.fullmatch(r"(?:19|20)\d{2}", v.strip())]
            if len(vals) >= thr:
                return True
    return False


def _positional_rows(page, x_gap_merge: int = 6):
    """Tier 2: cluster a page's words into rows by Y, merging thousand-groups of a
    single number by small X-gap. Returns list[list[str]] (cells per row)."""
    rows = {}
    for x0, y0, x1, y1, wd, *_ in page.get_text("words"):
        yc = round((y0 + y1) / 2)
        k = next((k for k in rows if abs(k - yc) <= 3), None)
        rows.setdefault(k if k is not None else yc, []).append((x0, x1, wd))
    out = []
    for y in sorted(rows):
        ws = sorted(rows[y]); toks = []
        for x0, x1, wd in ws:
            if toks:
                px0, px1, pw = toks[-1]
                if x0 - px1 < x_gap_merge and re.search(r"[\d)]$", pw) and re.match(r"[\d(]", wd):
                    toks[-1] = (px0, x1, pw + " " + wd); continue
            toks.append((x0, x1, wd))
        out.append([t[2] for t in toks])
    return out


_NUMCELL = re.compile(r"^[\d  .,()%-]+$")
_WELL = re.compile(r"^\(?-?\d{1,3}(?:[  ]\d{3})*\)?$|^\(?-?\d+\)?$|^-$|^\d+,\d+$")

def _is_numcell(c: str) -> bool:
    return bool(re.search(r"\d", c)) and bool(_NUMCELL.match(c))

def _well_formed(c: str) -> bool:
    return bool(_WELL.match(c)) and c.count("(") == c.count(")")

def _row_has_label(cells) -> bool:
    return any(re.search(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{4,}", c) and not _is_numcell(c) for c in cells)


def _validate_rescue(rows) -> bool:
    """Accept positional rescue only if it looks like a real table: >=3 labelled
    rows with >=2 well-formed numbers, low ill-formed ratio, consistent widths."""
    good, widths, ill, numtot = 0, [], 0, 0
    for cells in rows:
        nums = [c for c in cells if _is_numcell(c)]
        numtot += len(nums)
        ill += sum(1 for n in nums if not _well_formed(n))
        wf = [n for n in nums if _well_formed(n) and len(re.sub(r"\D", "", n)) >= 3]
        if len(wf) >= 2 and _row_has_label(cells):
            good += 1; widths.append(len(wf))
    if good < 3:
        return False
    if numtot and ill / numtot > 0.12:        # scrambled numbers (KRUK/XTB-annual)
        return False
    from statistics import mode
    m = mode(widths)
    cons = sum(1 for w in widths if abs(w - m) <= 1) / len(widths)
    return cons >= 0.6


def _rows_to_markdown(rows) -> str:
    """Render rescued rows as a clean Markdown table. Leading non-numeric tokens
    become the label cell; numeric tokens become columns. Pure-prose rows (no
    numbers) are skipped — they remain in the Tier-1 prose."""
    md_rows = []
    for cells in rows:
        if not any(_is_numcell(c) for c in cells):
            continue
        label, vals = [], []
        for c in cells:
            (vals if (_is_numcell(c) or c == "-") else label).append(c)
        row = [" ".join(label).strip()] + vals
        md_rows.append(row)
    if not md_rows:
        return ""
    width = max(len(r) for r in md_rows)
    out = []
    for i, r in enumerate(md_rows):
        r = r + [""] * (width - len(r))
        out.append("| " + " | ".join(r) + " |")
        if i == 0 and width >= 2:
            out.append("| " + " | ".join(["---"] * width) + " |")
    return "\n".join(out)


def _strip_table_blocks(md: str, only_jammed: bool) -> str:
    """Remove table-row blocks. If only_jammed, drop only blocks containing a
    value-jam (keep good tables); otherwise drop all table blocks."""
    lines = md.split("\n"); out = []; i = 0; n = len(lines)
    while i < n:
        if not _is_table_row(lines[i]):
            out.append(lines[i]); i += 1; continue
        blk = []
        while i < n and _is_table_row(lines[i]):
            blk.append(lines[i]); i += 1
        if only_jammed and not _has_value_jam("\n".join(blk)):
            out.extend(blk)        # keep non-jammed tables
        # else drop
    return "\n".join(out)


def _clean_page(md: str) -> str:
    md = clean_markdown_artifacts(md)
    md = remove_page_headers(md)
    md = expand_value_columns(md)
    md = fix_br_in_tables(md)
    md = sanitize_tables(md)
    md = clean_loose_semicolons(md)
    return md


def extract_clean_markdown(pdf_path: str) -> str:
    """Page-aware extraction with two-tier table rescue. Returns the full cleaned
    Markdown for the document (consumed by the chunker)."""
    pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, show_progress=False)
    doc = fitz.open(pdf_path)
    out_pages = []
    rescued_n = dropped_n = 0
    for i, pg in enumerate(pages):
        md = _clean_page(pg["text"])
        if _has_value_jam(md):
            rows = _positional_rows(doc[i])
            if _validate_rescue(rows):
                prose = _strip_table_blocks(md, only_jammed=False)   # drop jammed garbage tables
                rescued = _rows_to_markdown(rows)
                md = (prose.rstrip() + "\n\n" + rescued).strip()
                rescued_n += 1
            else:
                md = _strip_table_blocks(md, only_jammed=True)       # drop only jammed; keep rest
                dropped_n += 1
        out_pages.append(md)
    if rescued_n or dropped_n:
        print(f"    [tier-2] strony z jamem: rescue={rescued_n}, pominięte_tabele={dropped_n}")
    return "\n\n".join(out_pages)


# ---------------------------------------------------------------------------
# Shared extraction adapter — used BOTH for dataset creation and by
# report_analyzer.py for a single uploaded file (guarantees train↔inference parity).
# ---------------------------------------------------------------------------

def extract_inputs_from_pdf(
    pdf_path: str,
    max_tokens: int = 3700,
    min_characters: int = 800,
    max_garbled_risk: float = 0.05,
):
    """Extract a PDF into the list of final 'input' strings (the exact text fed to
    the model under '### Input:'). Applies Tier-1+Tier-2 extraction, chunking and
    the garbled-risk filter. Returns list[str]."""
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "Naglowek_Glowny"), ("##", "Sekcja"), ("###", "Podsekcja")])
    markdown_content = extract_clean_markdown(pdf_path)
    sections = md_splitter.split_text(markdown_content)

    inputs = []
    buffer_parts, buffer_tokens, buffer_contexts = [], 0, set()

    def _emit(parts, contexts):
        combined = strip_toc_blocks("\n\n".join(parts))
        if len(combined.strip()) < min_characters or is_toc_chunk(combined):
            return
        if max_garbled_risk is not None and compute_garbled_score(combined) > max_garbled_risk:
            return
        ctx = " | ".join(sorted(list(contexts)))
        inputs.append(f"DOKUMENT SEKCJA: {ctx}\n\n{combined}")

    for doc in sections:
        context_hierarchy = [doc.metadata[h] for h in ("Naglowek_Glowny", "Sekcja", "Podsekcja")
                             if h in doc.metadata]
        context_string = " > ".join(context_hierarchy) if context_hierarchy else "Główny Dokument"
        reconstructed_text = ""
        if "Naglowek_Glowny" in doc.metadata: reconstructed_text += f"# {doc.metadata['Naglowek_Glowny']}\n"
        if "Sekcja" in doc.metadata:          reconstructed_text += f"## {doc.metadata['Sekcja']}\n"
        if "Podsekcja" in doc.metadata:       reconstructed_text += f"### {doc.metadata['Podsekcja']}\n"
        reconstructed_text += doc.page_content
        if len(reconstructed_text.strip()) < 100:
            continue
        doc_tokens = count_tokens(reconstructed_text)

        if doc_tokens > max_tokens:
            if buffer_parts:
                _emit(buffer_parts, buffer_contexts)
                buffer_parts, buffer_tokens, buffer_contexts = [], 0, set()
            token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base", chunk_size=max_tokens, chunk_overlap=400,
                separators=["\n\n", "\n|", "\n", ". ", "? ", "! ", " "])
            current_table_header = None
            for sub_text in token_splitter.split_text(reconstructed_text):
                sub_text = strip_toc_blocks(sub_text)
                detected_header = extract_table_header(sub_text)
                if detected_header:
                    current_table_header = detected_header
                elif "|" in sub_text and "|---" not in sub_text.replace(" ", "") and current_table_header:
                    sub_text = current_table_header + "\n" + sub_text
                if len(sub_text) >= min_characters and not is_toc_chunk(sub_text):
                    _emit([sub_text], {context_string})
            continue

        if buffer_tokens + doc_tokens > max_tokens:
            if buffer_parts and buffer_parts[0].strip():
                _emit(buffer_parts, buffer_contexts)
            buffer_parts, buffer_tokens, buffer_contexts = [], 0, set()
        buffer_parts.append(reconstructed_text)
        buffer_tokens += doc_tokens
        buffer_contexts.add(context_string)

    if buffer_parts:
        _emit(buffer_parts, buffer_contexts)
    return inputs


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def process_pdf_to_clean_dataset(
    pdf_dir: str,
    out_dir: str,
    max_tokens: int = 3700,
    min_characters: int = 800,
    max_garbled_risk: float = None,
):
    os.makedirs(out_dir, exist_ok=True)
    src = os.path.abspath(pdf_dir)
    if os.path.isfile(src):
        base_dir, pdf_files = os.path.dirname(src), [os.path.basename(src)]
    else:
        base_dir, pdf_files = src, sorted(f for f in os.listdir(src) if f.endswith(".pdf"))

    for pdf_file in pdf_files:
        if not pdf_file.endswith(".pdf"):
            continue
        pdf_path = os.path.join(base_dir, pdf_file)
        name_only = os.path.splitext(pdf_file)[0]
        output_file = os.path.join(out_dir, f"{name_only}-clean.jsonl")
        print(f"\nProcesowanie: {pdf_file} → {output_file}")

        inputs = extract_inputs_from_pdf(pdf_path, max_tokens, min_characters, max_garbled_risk)

        with open(output_file, "w", encoding="utf-8") as out_f:
            for cid, final_input in enumerate(inputs, 1):
                row = {
                    "id": cid,
                    "instruction": INSTRUCTION_TEMPLATE,
                    "input": final_input,
                    "output": "",
                    "garbled_risk": round(compute_garbled_score(final_input), 4),
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  → Zapisano {len(inputs)} chunków")
    return


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- ścieżki produkcyjne (kontener) — zakomentowane na czas lokalnych testów ---
    process_pdf_to_clean_dataset(
        pdf_dir="/app/fine-tuning-scripts/materials",
        out_dir="/app/fine-tuning-scripts/_extract_out",
        max_tokens=3700,
        min_characters=800,
        max_garbled_risk=0.05,   # issue 4: odrzuć chunki z silną korupcją OCR (samogłoski)
    )

    # --- ścieżki lokalne (pełny zestaw źródeł z materials/) ---
    # process_pdf_to_clean_dataset(
    #     pdf_dir="/Users/ewalewski/python/fine-tuning-scripts/materials",
    #     out_dir="/Users/ewalewski/python/fine-tuning-scripts/_extract_out",
    #     max_tokens=3700,
    #     min_characters=800,
    #     max_garbled_risk=0.05,   # issue 4: odrzuć chunki z silną korupcją OCR (samogłoski)
    # )
