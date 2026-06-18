import os
import json
import re
import argparse
import urllib.error
import urllib.request
import tiktoken
import pymupdf4llm
import fitz  # PyMuPDF — for Tier-2 positional table rescue
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

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


_DATE_TOKEN = re.compile(r"\*{0,2}\d{2}\.\d{2}\.\d{4}\*{0,2}")


def _cell_is_date_stack(cell: str) -> bool:
    """True when one Markdown table cell is only several period dates."""
    plain = re.sub(r"[*_\s]+", "", cell)
    dates = _DATE_TOKEN.findall(cell)
    if len(dates) < 2:
        return False
    joined = "".join(re.sub(r"[*_\s]+", "", d) for d in dates)
    return plain == joined


def _expand_embedded_date_columns(line: str) -> str:
    if not _is_table_row(line):
        return line
    cells = _split_cells(line)
    out = []
    changed = False
    for cell in cells:
        if _cell_is_date_stack(cell):
            out.extend(d.strip() for d in _DATE_TOKEN.findall(cell))
            changed = True
        else:
            out.append(cell)
    return _join_cells(out) if changed else line


def expand_embedded_date_columns(text: str) -> str:
    return "\n".join(
        _expand_embedded_date_columns(l) if _is_table_row(l) else l
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


_FAIR_VALUE_HEADERS = ("POZIOM 1", "POZIOM 2", "POZIOM 3", "ŁĄCZNIE")
_FAIR_VALUE_TRAILING_NUM = re.compile(
    r"\s+(\*{0,2}(?:-?\(?\d{1,3}(?:\s\d{3})*(?:,\d+)?\)?|-)\*{0,2})\s*$"
)
_FAIR_VALUE_ROW = re.compile(
    r"^(.*?)\s+"
    r"(\*{0,2}(?:-|\d{1,3}\s\d{3})\*{0,2})\s+"
    r"(\*{0,2}(?:-|\d{1,3}\s\d{3})\*{0,2})\s+"
    r"(\*{0,2}(?:-|\d{1,3}\s\d{3})\*{0,2})\s+"
    r"(\*{0,2}(?:-|\d{1,3}(?:\s\d{3}){1,2})\*{0,2})$"
)


def _plain_cell(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[*_`]+", "", text)).strip()


def _strip_trailing_fin_values(text: str, n: int) -> str:
    result = text
    for _ in range(n):
        result = _FAIR_VALUE_TRAILING_NUM.sub("", result).strip()
    return _plain_cell(result)


def _take_trailing_fin_values(text: str, n: int):
    fair_match = _FAIR_VALUE_ROW.match(text.strip())
    if n == 4 and fair_match:
        return _plain_cell(fair_match.group(1)), [_plain_cell(v) for v in fair_match.groups()[1:]]

    result = text.strip()
    values = []
    for _ in range(n):
        match = _FAIR_VALUE_TRAILING_NUM.search(result)
        if not match:
            return "", []
        values.append(_plain_cell(match.group(1)))
        result = result[:match.start()].strip()
    values.reverse()
    return _plain_cell(result), values


def _repair_fair_value_hierarchy_block(block):
    text = "\n".join(block)
    if not all(h in text for h in _FAIR_VALUE_HEADERS):
        return None
    if any(len(_split_cells(line)) > 1 for line in block if not _is_sep_line(line)):
        return None

    out = [
        _join_cells(["Okres", "Pozycja", "POZIOM 1", "POZIOM 2", "POZIOM 3", "ŁĄCZNIE"]),
        _join_cells(["---", "---", "---", "---", "---", "---"]),
    ]
    period = ""
    section = ""
    data_rows = 0

    for line in block:
        if _is_sep_line(line):
            continue
        cell = _split_cells(line)[0] if _split_cells(line) else ""
        plain = _plain_cell(cell)
        if not plain:
            continue

        date_match = re.search(r"\b\d{2}\.\d{2}\.\d{4}\b", plain)
        if "(w tys. PLN)" in plain and date_match:
            period = date_match.group(0)
            section = ""
            continue
        if all(h in plain for h in _FAIR_VALUE_HEADERS):
            continue
        if plain in ("Aktywa finansowe", "Zobowiązania finansowe"):
            section = plain
            out.append(_join_cells([period, section, "", "", "", ""]))
            continue

        label, values = _take_trailing_fin_values(cell, 4)
        if len(values) < 4:
            continue
        if not label or label in ("Aktywa finansowe", "Zobowiązania finansowe"):
            continue
        if section and not label.startswith(section) and label.lower().startswith(("razem", "aktywa razem", "zobowiązania razem")):
            label = f"{section} - {label}"
        out.append(_join_cells([period, label] + values))
        data_rows += 1

    return out if data_rows >= 2 else None


_RESERVE_HEADER_START = re.compile(r"\b(?:Rezerwy z|Rezerwa na|Pozostałe rezerwy)\b", re.IGNORECASE)
_RESERVE_ROW_LABEL = re.compile(r"\bWartość na dzień\b", re.IGNORECASE)


def _split_reserve_headers(text: str):
    plain = _plain_cell(text)
    parts = [p.strip() for p in re.split(r"(?=\b(?:Rezerwy z|Rezerwa na|Pozostałe rezerwy)\b)", plain) if p.strip()]
    return parts


def _repair_reserve_matrix_block(block):
    """Split KRUK-style reserve matrices whose column headers are glued to row 1.

    PDF extraction can emit a row like:
      | **Rezerwy z** ... **Pozostałe** **rezerwy** Wartość na dzień... | 15 945 | ...
    That makes the first cell both a header blob and the first row label.  The
    values are still aligned, so convert it into a real Markdown matrix.
    """
    start = None
    headers = []
    first_row_label = ""
    for idx, line in enumerate(block):
        if _is_sep_line(line):
            continue
        cells = _split_cells(line)
        if len(cells) < 3:
            continue
        first = cells[0]
        if not (_RESERVE_ROW_LABEL.search(_plain_cell(first)) and len(_RESERVE_HEADER_START.findall(_plain_cell(first))) >= 2):
            continue
        before, after = re.split(_RESERVE_ROW_LABEL, first, maxsplit=1)
        headers = _split_reserve_headers(before)
        if len(headers) < 2:
            continue
        first_row_label = "Wartość na dzień " + _plain_cell(after)
        start = idx
        break

    if start is None:
        return None

    out = []
    if start > 0:
        out.extend(_process_table_block(block[:start]))
        out.append("")

    width = len(headers) + 1
    out.append(_join_cells(["Pozycja"] + headers))
    out.append(_join_cells(["---"] * width))

    for idx, line in enumerate(block[start:]):
        if _is_sep_line(line):
            continue
        cells = _split_cells(line)
        if idx == 0:
            cells[0] = first_row_label
        if len(cells) > width:
            cells = cells[:width - 1] + [" ".join(x for x in cells[width - 1:] if x).strip()]
        elif len(cells) < width:
            cells = cells + [""] * (width - len(cells))
        out.append(_join_cells(cells))

    return out


def _process_table_block(block):
    fair_value = _repair_fair_value_hierarchy_block(block)
    if fair_value:
        return fair_value
    reserve_matrix = _repair_reserve_matrix_block(block)
    if reserve_matrix:
        return reserve_matrix

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
    # A common PDF failure is a compact financial table whose header has dates
    # split into separate period columns while data rows have missing values.
    # Prefer the header width for small tables; otherwise the header dates are
    # folded back into one cell and the table loses its period mapping.
    first_width = counts[0]
    if 2 <= width < first_width <= 8:
        width = first_width
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


_NONPIPE_FINANCIAL_TABLE_HINT = re.compile(
    r"\b(?:kapitał|aktywa|zobowiązania|przychod|koszt|zysk|strat|"
    r"należności|zapasy|środki pieniężne|przepływ|razem)\b",
    re.IGNORECASE,
)
_GROUPED_AMOUNT = r"\(?\d{1,3}(?:[ \u00a0]\d{3})+(?:[,.]\d+)?\)?"
_ADJACENT_GROUPED_AMOUNTS_RE = re.compile(
    rf"{_GROUPED_AMOUNT}\s+{_GROUPED_AMOUNT}"
)
_DIGIT_GROUP_SOUP_RE = re.compile(r"(?:\b\d{3}\b[\s\u00a0]+){4,}\b\d{3}\b")


def _is_nonpipe_financial_table_blob(line: str) -> bool:
    """Detect flattened financial tables emitted as one prose line with <br>.

    These blocks have lost their column grid completely. Keeping them is worse
    than dropping them: the LLM tends to assign clear-looking figures to the
    wrong financial statement line item.
    """
    s = line.strip()
    if "|" in s or "<br>" not in s:
        return False
    br_count = len(re.findall(r"<br>", s, flags=re.IGNORECASE))
    if br_count < 4:
        return False
    values = _count_financial_values(s)
    if values < 8:
        return False
    words = re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{4,}", s)
    if len(words) < 6:
        return False
    return bool(_NONPIPE_FINANCIAL_TABLE_HINT.search(s))


def strip_nonpipe_financial_table_blobs(text: str) -> str:
    """Drop unrecoverable <br>-flattened financial tables without Markdown pipes."""
    out, dropped = [], 0
    for line in text.split("\n"):
        if _is_nonpipe_financial_table_blob(line):
            dropped += 1
            continue
        out.append(line)
    if dropped:
        print(f"    [cleanup] usunięte spłaszczone tabele bez siatki: {dropped}")
    return "\n".join(out)


def _pipe_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_collapsed_pipe_numeric_row(line: str) -> bool:
    """Detect Markdown-ish rows whose columns collapsed into one numeric cell."""
    s = line.strip()
    if "|" not in s or re.fullmatch(r"\|?\s*:?-{3,}:?(?:\s*\|\s*:?-{3,}:?)+\s*\|?", s):
        return False

    if _ADJACENT_GROUPED_AMOUNTS_RE.search(s) or _DIGIT_GROUP_SOUP_RE.search(s):
        return True

    values = _count_financial_values(s)
    if values < 4:
        return False

    cells = _pipe_cells(s)
    nonempty = [cell for cell in cells if cell]
    if not nonempty:
        return False

    # Page-number-prefixed OCR/table debris, e.g. "60 |**01.01.2024 ..."
    if re.match(r"^\d{1,3}\s*\|", s) and values >= 4:
        return True

    # Label + one cell containing several period values without separators.
    if len(nonempty) <= 2:
        words = re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{3,}", s)
        return bool(_NONPIPE_FINANCIAL_TABLE_HINT.search(s)) or len(words) <= 2

    # A single cell still contains a long run of values, so the grid is broken.
    for cell in nonempty:
        if _ADJACENT_GROUPED_AMOUNTS_RE.search(cell) or _DIGIT_GROUP_SOUP_RE.search(cell):
            return True
        if _count_financial_values(cell) >= 5 and len(re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{3,}", cell)) <= 2:
            return True

    return False


def strip_collapsed_pipe_numeric_rows(text: str) -> str:
    """Drop table rows that retained pipes but lost their true column grid."""
    out, dropped = [], 0
    for line in text.split("\n"):
        if _is_collapsed_pipe_numeric_row(line):
            dropped += 1
            continue
        out.append(line)
    if dropped:
        print(f"    [cleanup] usunięte zlane wiersze tabel: {dropped}")
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
_KNOWN_ABBREVS = re.compile(r"XTB|GPW|MSSF|MSR|IFRS|ESG|PLN|EUR|USD|CEO|CFO")
_PRIVATE_GLYPH_RE = re.compile(r"[\ue000-\uf8ff]")
_VOWEL_DROP_HINTS = re.compile(
    r"\b(?:"
    r"rku|diu|grudia|listpada|zstał[ay]?|"
    r"krsach|okrsach|dwuastu|misiecy|zaknczych|zakoncz[oy]nych|"
    r"spol?ka|społka|strą|przdstaw|trasakcji|pdmitami|pwiązaymi|piżj|tablach|"
    r"rachuku|zyskow|zapaswy|przyzaych|mdżrskich|menedżrskich|pcji|"
    r"całkwita|ralizacji|ilść|przychdyz|umwych|przumin|"
    r"wartści|przychdy|kosztysprz|zobwiąza"
    r")\b",
    re.IGNORECASE,
)


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


def _token_vowel_ratio(token: str) -> float:
    clean = re.sub(r"<br>", "", token)
    clean = re.sub(r"^[*_.,;:!?'\"()\[\]|<>]+|[*_.,;:!?'\"()\[\]|<>]+$", "", clean)
    letters = [c for c in clean if c.isalpha()]
    if len(letters) < 5:
        return 1.0
    return sum(1 for c in letters if c in _POLISH_VOWELS) / len(letters)


def _max_consonant_cluster(token: str) -> int:
    max_run = run = 0
    for c in token:
        if c.isalpha() and c not in _POLISH_VOWELS:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


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
    high_consonant = sum(1 for t in qualifying if _token_consonant_ratio(t) >= 0.85) / len(qualifying)
    low_vowel = sum(1 for t in qualifying if _token_vowel_ratio(t) < 0.25) / len(qualifying)
    consonant_clusters = sum(1 for t in qualifying if _max_consonant_cluster(t) >= 5) / len(qualifying)
    return max(high_consonant, low_vowel, consonant_clusters)


def compute_font_damage_score(text: str) -> float:
    """Detect broken PDF font mappings and dropped-vowel text.

    This catches the DDSA-style failure mode where a text parser maps vowels to
    private-use glyphs (//) or drops them during normalization. It is separate
    from visual OCR noise: the text may look sentence-like, but Polish words are
    unreadable and should be recovered by OCR or skipped.
    """
    if not text:
        return 0.0
    prose = "\n".join(l for l in text.split("\n") if not l.strip().startswith("|"))
    private_glyphs = len(_PRIVATE_GLYPH_RE.findall(text))
    alpha_chars = max(1, sum(1 for c in text if c.isalpha()))
    private_score = min(1.0, private_glyphs / alpha_chars * 10)

    tokens = re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż\ue000-\uf8ff]{5,}", prose)
    qualifying = [
        t for t in tokens
        if not _KNOWN_ABBREVS.search(t)
        and not re.fullmatch(r"[A-ZĄĆĘŁŃÓŚŹŻ]{2,}", t)
    ]
    if not qualifying:
        hint_score = 1.0 if _VOWEL_DROP_HINTS.search(text) else 0.0
        return max(private_score, hint_score)

    low_vowel = sum(
        1
        for t in qualifying
        if _token_vowel_ratio(t) <= 0.20 and _max_consonant_cluster(t) >= 4
    ) / len(qualifying)
    hints = len(_VOWEL_DROP_HINTS.findall(text))
    hint_score = min(1.0, hints / max(3, len(qualifying) * 0.04))
    return max(private_score, low_vowel, hint_score)


def has_font_encoding_damage(text: str, threshold: float = 0.06) -> bool:
    if not text:
        return False
    if len(_PRIVATE_GLYPH_RE.findall(text)) >= 3:
        return True
    return compute_font_damage_score(text) > threshold


def _line_word_quality(line: str) -> float:
    words = re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{3,}", line)
    if not words:
        return 0.0
    good = 0
    for word in words:
        if _token_vowel_ratio(word) >= 0.25 and _max_consonant_cluster(word) < 5:
            good += 1
    return good / len(words)


def _is_visual_noise_line(line: str) -> bool:
    """Detect OCR residue from charts/graphics, not ordinary prose or tables."""
    s = line.strip()
    if not s or _is_table_row(s) or _is_heading_line(s):
        return False
    if re.fullmatch(r"\d{1,4}", s):
        return False

    tokens = s.split()
    if len(s) <= 2:
        return True

    alpha = sum(1 for c in s if c.isalpha())
    digits = sum(1 for c in s if c.isdigit())
    symbols = sum(1 for c in s if not c.isalnum() and not c.isspace())
    nonspace = max(1, sum(1 for c in s if not c.isspace()))
    single = sum(1 for t in tokens if len(t.strip(".,;:()[]{}|/\\-_")) <= 1)
    quality = _line_word_quality(s)

    if len(tokens) >= 4 and single / len(tokens) >= 0.55:
        return True
    if len(tokens) >= 3 and quality < 0.25 and (symbols + digits) / nonspace >= 0.35:
        return True
    if alpha >= 4 and quality < 0.2 and len(tokens) >= 3:
        return True
    if symbols / nonspace >= 0.35 and alpha < 12:
        return True
    return False


def _is_spaced_caps_noise_line(line: str) -> bool:
    s = re.sub(r"<br>", " ", line.strip())
    if _is_table_row(s) or len(s) < 8:
        return False
    tokens = [t.strip(".,;:()[]{}") for t in s.split() if t.strip(".,;:()[]{}")]
    if not tokens:
        return False
    single_alnum = sum(1 for t in tokens if len(t) == 1 and t.isalnum())
    letters = [c for c in s if c.isalpha()]
    if re.fullmatch(r"\d{1,3}", s):
        return True
    if re.fullmatch(r"(?:\d\s+){1,4}\d", s):
        return True
    if re.fullmatch(r"(?:[A-ZĄĆĘŁŃÓŚŹŻ]\s+){1,6}(?:\d\s+){3,}\d", s):
        return True
    return (
        len(tokens) >= 8
        and single_alnum / len(tokens) >= 0.65
        and len(letters) >= 8
        and sum(c.isupper() for c in letters) / len(letters) >= 0.80
    )


_MIRRORED_TABLE_TOKEN_RE = re.compile(
    r"\b(?:ywrezer|ywrezeR|awytka|awytkA|alartnec|alartneC|ynsałw|mełógo|"
    r"iksyz|enamyzrtaz|ewowrezer|hcycąjulortnokein)\b",
    re.IGNORECASE,
)


def _count_financial_values(text: str) -> int:
    return len(re.findall(r"\(?-?\d{1,3}(?:[ \u00a0]\d{3})*(?:[,.]\d+)?%?\)?", text))


def _is_gross_number_blob_cell(cell: str) -> bool:
    plain = _plain_cell(cell)
    if len(plain) < 35:
        return False
    values = _count_financial_values(plain)
    words = re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{4,}", plain)
    return values >= 8 and len(words) <= 2


def _has_bad_table_artifact(text: str) -> bool:
    """True for table artefacts that are worse than losing the table.

    These include mirrored table cells (e.g. "ywrezeR"/"awytkA") and cells that
    are only long unlabelled financial-number blobs. They usually come from a
    failed table extraction where the original column mapping is no longer
    recoverable.
    """
    if any(_is_numbered_table_artifact_line(line) for line in text.split("\n")):
        return True
    for block in _markdown_atomic_blocks(text):
        lines = block.split("\n")
        if not lines or not _is_table_row(lines[0]):
            continue
        rows = [r for r in lines if _is_table_row(r) and not _is_sep_line(r)]
        if not rows:
            continue
        block_text = "\n".join(rows)
        if _MIRRORED_TABLE_TOKEN_RE.search(block_text):
            return True
        for row in rows:
            cells = _split_cells(row)
            if any(_is_gross_number_blob_cell(c) for c in cells):
                return True
    return False


def _is_numbered_table_artifact_line(line: str) -> bool:
    """Detect page-number-prefixed table garbage, e.g. '46 |...|(...numbers...)|'."""
    s = line.strip()
    if not re.match(r"^\d{1,4}\s*\|", s):
        return False
    if _MIRRORED_TABLE_TOKEN_RE.search(s):
        return True
    cells = [c.strip() for c in s.split("|")[1:]]
    if any(_is_gross_number_blob_cell(c) for c in cells):
        return True
    values = _count_financial_values(s)
    words = re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{4,}", _plain_cell(s))
    return values >= 10 and len(words) <= 8


def strip_visual_noise_blocks(text: str) -> str:
    """Drop dense OCR/graphics residue blocks while preserving prose and tables."""
    def _table_metrics(block):
        rows = [row for row in block if not _is_sep_line(row)]
        if not rows:
            return {
                "rows": [], "width": 0, "word_ratio": 0.0, "numeric_ratio": 0.0,
                "numish_ratio": 0.0, "long_ratio": 0.0, "url_count": 0,
            }
        widths = [len(_split_cells(row)) for row in rows]
        width = max(set(widths), key=widths.count)
        all_cells = [c for row in rows for c in _split_cells(row)]
        nonempty = [_plain_cell(c) for c in all_cells if _plain_cell(c)]
        if not nonempty:
            return {
                "rows": rows, "width": width, "word_ratio": 0.0, "numeric_ratio": 0.0,
                "numish_ratio": 0.0, "long_ratio": 0.0, "url_count": 0,
            }
        return {
            "rows": rows,
            "width": width,
            "word_ratio": sum(bool(re.search(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{4,}", c)) for c in nonempty) / len(nonempty),
            "numeric_ratio": sum(bool(re.search(r"\d", c)) for c in nonempty) / len(nonempty),
            "numish_ratio": sum(bool(re.fullmatch(r"\(?-?[\d\s.,%]+\)?", c)) for c in nonempty) / len(nonempty),
            "long_ratio": sum(len(c) > 55 for c in nonempty) / len(nonempty),
            "url_count": sum(bool(re.search(r"https?://|www\.|\.(?:com|pl|eu|net|org)\b", c, re.IGNORECASE)) for c in nonempty),
        }

    def _is_prose_layout_table_block(block):
        metrics = _table_metrics(block)
        rows = metrics["rows"]
        if len(rows) < 2:
            return False
        plain_block = _plain_cell("\n".join(block)).lower()
        width = metrics["width"]

        if (
            width >= 8
            and (
                "informacja dodatkowa" in plain_block
                or (metrics["url_count"] >= 1 and any(term in plain_block for term in ("wydarzenie", "zarząd", "rada nadzorcza", "konferencja")))
                or ("konferencja wynikowa" in plain_block and "wydarzenie" in plain_block)
            )
        ):
            return True

        if (
            width >= 12
            and metrics["word_ratio"] >= 0.60
            and metrics["long_ratio"] >= 0.35
            and metrics["numish_ratio"] < 0.25
        ):
            return True

        if (
            width >= 16
            and len(rows) <= 6
            and metrics["word_ratio"] >= 0.25
            and metrics["numish_ratio"] < 0.75
        ):
            return True

        # Some one/two-column artefacts are just paragraphs wrapped in pipes.
        return (
            width <= 2
            and len(rows) >= 3
            and metrics["word_ratio"] >= 0.70
            and metrics["long_ratio"] >= 0.60
            and metrics["numish_ratio"] < 0.20
        )

    def _table_block_to_prose(block):
        prose = []
        seen = set()
        def emit(line):
            line = re.sub(r"\s+", " ", line).strip()
            if not line or line in seen:
                return
            prose.append(line)
            seen.add(line)

        for row in block:
            if _is_sep_line(row):
                continue
            parts = [_plain_cell(c) for c in _split_cells(row) if _plain_cell(c)]
            if not parts:
                continue
            if len(parts) > 8:
                for start in range(0, len(parts), 6):
                    emit(" ".join(parts[start:start + 6]))
            else:
                emit(" ".join(parts))
        return prose

    def is_noise_table_block(block):
        block_text = "\n".join(block)
        def _cashflow_norm(s: str) -> str:
            s = re.sub(r"[*_`]+", "", s.lower())
            s = s.replace("środkówpieniężnych", "środków pieniężnych")
            s = s.replace("działalno- ści", "działalności")
            s = s.replace("działalno-ść", "działalność")
            return re.sub(r"[\s\-]+", "", s)

        if _MIRRORED_TABLE_TOKEN_RE.search(block_text):
            return True
        if any(_is_gross_number_blob_cell(cell) for row in block for cell in _split_cells(row)):
            return True

        if (
            "Liczba pracowników" in block_text
            and "OGÓŁEM" in block_text
            and re.search(r"\|\s*\d+(?:\s+\d+){8,}", block_text)
        ):
            return True
        if any(row.count("Wypływy netto ze sprzedaży akcji/udziałów") >= 2 for row in block):
            return True
        if any(
            (
                "przepływ" in row.lower()
                and row.lower().count("działalno") >= 2
            )
            or (
                "środkipieniężnenetto" in _cashflow_norm(row)
                and "przepływ" in _cashflow_norm(row)
            )
            or (
                _cashflow_norm(row).count("przepływ") >= 2
                and "działalności" in row.lower()
            )
            for row in block
        ):
            return True

        cells = [c for row in block for c in _split_cells(row) if not _is_sep_line(row)]
        rows = [row for row in block if not _is_sep_line(row)]
        if rows:
            widths = [len(_split_cells(row)) for row in rows]
            width = max(set(widths), key=widths.count)
            plain_block = _plain_cell(block_text).lower()
            url_count = len(re.findall(r"https?://|www\.|\.com|\.pl", block_text, re.IGNORECASE))
            # XTB-style layout tables: descriptive investor-relations calendars
            # extracted as 20+ Markdown columns. They contain useful prose, but
            # the pipe grid is artificial and repeatedly damages chunk quality.
            if (
                width >= 8
                and (
                    "informacja dodatkowa" in plain_block
                    or (url_count >= 2 and "wydarzenie" in plain_block)
                    or ("konferencja wynikowa" in plain_block and "wydarzenie" in plain_block)
                )
            ):
                return True
            nonempty_cells = [_plain_cell(c) for c in cells if _plain_cell(c)]
            if nonempty_cells:
                numeric_ratio = sum(bool(re.search(r"\d", c)) for c in nonempty_cells) / len(nonempty_cells)
                word_ratio = sum(bool(re.search(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{4,}", c)) for c in nonempty_cells) / len(nonempty_cells)
                # PDF layout tables sometimes turn paragraphs into 10-30 Markdown
                # columns. These are not data tables; preserving the pipe grid
                # teaches the model a broken structure.
                if width >= 8 and len(rows) >= 2 and word_ratio > 0.65 and numeric_ratio < 0.35:
                    return True

            for row in rows:
                row_cells = _split_cells(row)
                if (
                    len(row_cells) >= 3
                    and len(_plain_cell(row_cells[0])) > 180
                    and sum(bool(re.search(r"\d", c)) for c in row_cells[1:]) >= 2
                ):
                    return True

            glued_rows = 0
            for row in rows:
                for cell in _split_cells(row):
                    plain = _plain_cell(cell)
                    nums = re.findall(r"\(?-?\d{1,3}(?:\s\d{3})*(?:,\d+)?\)?", plain)
                    if len(nums) >= 4 and len(plain) > 35:
                        glued_rows += 1
                        break
            if cells:
                empty_ratio = (len(cells) - len([c for c in cells if c.strip()])) / len(cells)
                if width >= 4 and empty_ratio > 0.35 and glued_rows >= 2:
                    return True

        if len(cells) < 30:
            return False
        nonempty = [c for c in cells if c.strip()]
        if not nonempty:
            return True
        single_alpha = sum(1 for c in nonempty if re.fullmatch(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]", c.strip()))
        empty_ratio = (len(cells) - len(nonempty)) / len(cells)
        numeric = sum(1 for c in nonempty if re.search(r"\d", c))
        wordish = sum(1 for c in nonempty if re.search(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{4,}", c))
        return (
            single_alpha / len(nonempty) >= 0.25
            and empty_ratio >= 0.25
            and numeric / len(nonempty) < 0.35
            and wordish / len(nonempty) < 0.25
        )

    lines = text.split("\n")
    out, buf, dropped = [], [], 0

    def flush():
        nonlocal buf, dropped
        if not buf:
            return
        content = [l for l in buf if l.strip()]
        noisy = sum(1 for l in content if _is_visual_noise_line(l))
        words = sum(len(re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{3,}", l)) for l in content)
        avg_len = sum(len(l.strip()) for l in content) / max(1, len(content))
        sentence_lines = sum(1 for l in content if re.search(r"[.!?]\s*$", l.strip()) and len(l.strip()) > 35)
        drop = (
            len(content) >= 5
            and noisy / len(content) >= 0.6
            and words < len(content) * 2.5
        ) or (
            len(content) >= 12
            and noisy / len(content) >= 0.45
            and sum(_line_word_quality(l) for l in content) / len(content) < 0.35
        ) or (
            len(content) >= 18
            and noisy / len(content) >= 0.30
            and avg_len < 24
            and sentence_lines == 0
        )
        if drop:
            dropped += len(content)
        else:
            out.extend(buf)
        buf = []

    for line in lines:
        if _is_numbered_table_artifact_line(line):
            flush()
            dropped += 1
            continue
        if _is_table_row(line):
            flush()
            block = [line]
            continue_marker = None
            # The outer loop is easier to keep simple by using a marker: table
            # blocks are rare and short relative to page text.
            out.append(("__TABLE_BLOCK__", block))
            continue
        if _is_heading_line(line) or not line.strip():
            flush()
            out.append(line)
            continue
        buf.append(line)
    flush()

    flattened = []
    i = 0
    while i < len(out):
        item = out[i]
        if not (isinstance(item, tuple) and item[0] == "__TABLE_BLOCK__"):
            flattened.append(item)
            i += 1
            continue
        block = []
        while i < len(out) and isinstance(out[i], tuple) and out[i][0] == "__TABLE_BLOCK__":
            block.extend(out[i][1])
            i += 1
        kept_block = []
        for row in block:
            row_text = " ".join(_split_cells(row))
            if _is_spaced_caps_noise_line(row_text):
                dropped += 1
                continue
            kept_block.append(row)
        block = kept_block
        if not block:
            continue
        if _is_prose_layout_table_block(block):
            flattened.extend(_table_block_to_prose(block))
        elif is_noise_table_block(block):
            dropped += len(block)
        else:
            flattened.extend(block)

    if dropped:
        print(f"    [cleanup] usunięte bloki OCR/grafiki: {dropped}")
    cleaned = "\n".join(flattened)
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def visual_noise_score(text: str) -> float:
    lines = [l for l in text.split("\n") if l.strip() and not _is_table_row(l) and not _is_heading_line(l)]
    if len(lines) < 5:
        return 0.0
    return sum(1 for l in lines if _is_visual_noise_line(l)) / len(lines)


def _numeric_tokens(text: str):
    return re.findall(r"(?<![A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż])\(?-?\d{1,3}(?:[ \u00a0]\d{3})*(?:[,.]\d+)?%?\)?", text)


def _norm_num_token(token: str) -> str:
    return re.sub(r"\s+", "", token.replace("\u00a0", " ")).replace(",", ".")


def _has_layout_table_issue(text: str) -> bool:
    for block in _markdown_atomic_blocks(text):
        block_lines = block.split("\n")
        if not block_lines or not _is_table_row(block_lines[0]):
            continue
        rows = [r for r in block_lines if _is_table_row(r) and not _is_sep_line(r)]
        if len(rows) < 2:
            continue
        widths = [len(_split_cells(r)) for r in rows]
        width = max(set(widths), key=widths.count)
        all_cells = [_plain_cell(c) for r in rows for c in _split_cells(r)]
        nonempty = [c for c in all_cells if c]
        if not nonempty:
            continue
        word_ratio = sum(bool(re.search(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{4,}", c)) for c in nonempty) / len(nonempty)
        numeric_ratio = sum(bool(re.search(r"\d", c)) for c in nonempty) / len(nonempty)
        numish_ratio = sum(bool(re.fullmatch(r"\(?-?[\d\s.,%]+\)?", c)) for c in nonempty) / len(nonempty)
        long_ratio = sum(len(c) > 55 for c in nonempty) / len(nonempty)
        if width >= 8 and word_ratio > 0.65 and numeric_ratio < 0.35:
            return True
        if width >= 12 and word_ratio >= 0.60 and long_ratio >= 0.35 and numish_ratio < 0.25:
            return True
        if width <= 2 and len(rows) >= 3 and word_ratio >= 0.70 and long_ratio >= 0.60 and numish_ratio < 0.20:
            return True
        if any(
            len(_split_cells(r)) >= 3
            and len(_plain_cell(_split_cells(r)[0])) > 180
            and sum(bool(re.search(r"\d", c)) for c in _split_cells(r)[1:]) >= 2
            for r in rows
        ):
            return True
    return False


def should_llama_repair(text: str) -> bool:
    """Return True only for chunks where deterministic cleanup still looks risky."""
    if has_font_encoding_damage(text):
        return True
    if is_mirrored_text(text) or contains_mirrored_token(text):
        return True
    if visual_noise_score(text) > 0.30:
        return True
    if re.search(r"(?:[A-ZĄĆĘŁŃÓŚŹŻ0-9]\s+){12,}", text):
        return True
    if _has_layout_table_issue(text):
        return True
    if _has_bad_table_artifact(text):
        return True
    return False


def _validate_llama_repair(original: str, repaired: str) -> bool:
    if not repaired:
        return False
    repaired = repaired.strip()
    if repaired == "UNREPAIRABLE" or repaired.startswith("```") or "<source" in repaired:
        return False
    if len(repaired) < max(120, int(len(original) * 0.35)):
        return False

    orig_nums = [_norm_num_token(n) for n in _numeric_tokens(original)]
    new_nums = [_norm_num_token(n) for n in _numeric_tokens(repaired)]
    if orig_nums:
        orig_set = set(orig_nums)
        new_set = set(new_nums)
        invented = new_set - orig_set
        if invented:
            return False
        retained = len(orig_set & new_set) / max(1, len(orig_set))
        min_retained = 0.75 if should_llama_repair(original) else 0.90
        if retained < min_retained:
            return False

    if is_mirrored_text(repaired) or contains_mirrored_token(repaired):
        return False
    if has_font_encoding_damage(repaired, threshold=0.04):
        return False
    if _has_layout_table_issue(repaired):
        return False
    if _has_bad_table_artifact(repaired):
        return False
    if visual_noise_score(repaired) > max(0.35, visual_noise_score(original)):
        return False
    return True


def llama_repair_text(
    text: str,
    model: str = "chunk-repair:latest",
    url: str = None,
    timeout: int = 120,
) -> str:
    """Use Ollama as a conservative repair fallback. Returns original on failure."""
    endpoint = (url or os.environ.get("OLLAMA_REPAIR_URL") or "http://127.0.0.1:11434").rstrip("/")
    prompt = (
        "You repair noisy PDF extraction chunks for a financial-report training dataset.\n"
        "Rules:\n"
        "- Preserve all factual content and every financial number exactly as written.\n"
        "- Do not summarize, translate, infer, add explanations, or add new numbers.\n"
        "- The source language is usually Polish. Keep the original language.\n"
        "- You may restore obvious broken Polish letters/vowels caused by PDF font encoding "
        "(for example krsach -> okresach, misiecy -> miesięcy), but only when the word is unambiguous.\n"
        "- Remove PDF layout garbage, repeated headers/footers, mirrored text, private-use glyphs, "
        "and broken table-layout artifacts.\n"
        "- If a table is too damaged to preserve safely, convert only its readable content to plain prose.\n"
        "- Return only the repaired chunk text. If you cannot repair safely, return exactly: UNREPAIRABLE.\n\n"
        "CHUNK:\n"
        f"{text}"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.9,
            "num_ctx": 4096,
        },
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{endpoint}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"    [llama] repair unavailable: {exc}")
        return text

    repaired = (body.get("response") or "").strip()
    if _validate_llama_repair(text, repaired):
        return repaired
    print("    [llama] repair rejected by validator")
    return text


def trim_orphan_chunk_edges(text: str) -> str:
    """Remove obvious boundary punctuation left by PDF/chunk splits."""
    cleaned = re.sub(r"^\s*[.,;:]\s+", "", text.strip())
    return cleaned


_BAD_FINAL_WORDS = {
    "oraz", "i", "a", "w", "z", "na", "do", "dla", "przez", "które", "który",
    "która", "których", "m.in", "np", "tj", "w tym",
}


def has_incomplete_chunk_boundary(text: str) -> bool:
    tail = re.sub(r"\s+", " ", text.rstrip()[-220:]).strip(" *_`")
    if not tail:
        return True
    tail_lower = tail.lower().rstrip(".,;:")
    if any(tail_lower.endswith(f" {word}") or tail_lower == word for word in _BAD_FINAL_WORDS):
        return True
    return False


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
    out, i, n = [], 0, len(lines)
    removed_heading_toc = False
    while i < n:
        line = lines[i]
        if _is_heading_line(line) and "spis treści" in line.lower():
            i += 1
            while i < n:
                nxt = lines[i]
                if _is_heading_line(nxt) and "spis treści" not in nxt.lower():
                    break
                i += 1
            removed_heading_toc = True
            continue
        out.append(line)
        i += 1
    if removed_heading_toc:
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
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


def _clean_contexts(contexts):
    cleaned = []
    for ctx in contexts:
        parts = [p.strip() for p in re.split(r"\s+\|\s+", ctx) if p.strip()]
        parts = [p for p in parts if "spis treści" not in p.lower() and "contents" not in p.lower()]
        if parts:
            cleaned.append(" | ".join(parts))
    return cleaned


def demote_spurious_headings(text: str) -> str:
    """Convert sentence fragments accidentally parsed as Markdown headings to prose."""
    out = []
    for line in text.split("\n"):
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not m:
            out.append(line)
            continue
        raw = m.group(2).strip()
        plain = re.sub(r"[*_`]+", "", raw).strip()
        words = re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9%]+", plain)
        sentence_like = (
            bool(plain)
            and (
                plain[:1].islower()
                or (len(words) >= 5 and plain.rstrip().endswith((".", ",", ";", ":")))
            )
        )
        if sentence_like:
            out.append(plain)
        else:
            out.append(line)
    return "\n".join(out)


_MIRROR_TERMS = (
    "kapitał", "własny", "ogółem", "udziały", "niekontrolujące",
    "właścicieli", "jednostki", "dominującej", "zyski", "zatrzymane",
    "przychody", "odsetkowe", "koszty", "wartość", "godziwej",
    "wierzytelności", "pożyczek", "zabezpieczających", "instrumentów",
    "sprawozdawczy", "czerwca", "grudnia", "stycznia", "niebadane",
    "tysiącach", "złotych", "razem", "pozostałe", "operacyjne",
    "zabezpieczenie", "przepływów", "przepływy", "pieniężnych",
    "pieniężne", "inwestycji", "jednostkę", "zależną", "ryzyko",
    "walutowe", "procentowej", "przyszłych",
)

_MIRROR_TOKEN_HINTS = (
    "łatipak", "ynsałw", "mełógo", "yłaizdu", "ecąjulortnokein",
    "ileicicśałw", "iktsondej", "jecąjunimod", "iksyz", "enamyzrtaz",
    "ełatsozop", "yłatipak", "ewowrezer", "zewosruk", "ainezcilezrp",
    "hcycąjałaizd", "zywowrezer", "ynecyw", "wómargorp",
    "hcycąjazceipzebaz", "zewoktesdoydohcyzrp", "ydohcyzrp",
    "zydohcyzrp", "gułsu", "icśonletyzreiw", "hcytołzhcacąisytw",
    "enadabein", "mezar", "wówyłpezrp", "hcynżęineip",
    "ęktsondejwijcytsewni", "ąnżelaz", "ewotulaw", "okyzyr",
)


def _alpha_tokens(text: str):
    return re.findall(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{5,}", text)


def _is_mirrored_token(token: str) -> bool:
    low = token.lower()
    rev = low[::-1]
    return any(hint in low for hint in _MIRROR_TOKEN_HINTS) or any(term in rev for term in _MIRROR_TERMS)


def mirrored_text_score(text: str) -> float:
    tokens = [
        t for t in _alpha_tokens(text)
        if not re.fullmatch(r"[IVXLCDM]+", t)
    ]
    if len(tokens) < 4:
        return 0.0
    return sum(1 for t in tokens if _is_mirrored_token(t)) / len(tokens)


def contains_mirrored_token(text: str) -> bool:
    return any(_is_mirrored_token(t) for t in _alpha_tokens(text))


def is_mirrored_text(text: str) -> bool:
    tokens = _alpha_tokens(text)
    if len(tokens) < 4:
        return contains_mirrored_token(text)
    score = mirrored_text_score(text)
    hits = sum(1 for t in tokens if _is_mirrored_token(t))
    return score >= 0.25 or hits >= 2 or (score >= 0.15 and len(tokens) >= 12)


def strip_mirrored_blocks(text: str) -> str:
    """Drop PDF extraction artefacts whose text direction is reversed.

    These blocks usually appear as duplicated, unusable table copies next to a
    valid table extracted by another fallback. Keeping them is worse than losing
    the block because the model would learn right-to-left gibberish.
    """
    def _filter_table_block(block):
        keep = []
        mirrored_flags = [is_mirrored_text(row) for row in block]
        for idx, row in enumerate(block):
            if mirrored_flags[idx] or contains_mirrored_token(row):
                continue
            if _is_sep_line(row):
                prev_kept = any(not mirrored_flags[j] and not _is_sep_line(block[j]) for j in range(0, idx))
                next_kept = any(not mirrored_flags[j] and not _is_sep_line(block[j]) for j in range(idx + 1, len(block)))
                if not (prev_kept and next_kept):
                    continue
            keep.append(row)
        return keep, sum(1 for flag in mirrored_flags if flag)

    lines = text.split("\n")
    out, i, n, dropped = [], 0, len(lines), 0
    while i < n:
        if _is_table_row(lines[i]):
            block = []
            while i < n and _is_table_row(lines[i]):
                block.append(lines[i])
                i += 1
            filtered, dropped_rows = _filter_table_block(block)
            dropped += dropped_rows
            out.extend(filtered)
            continue

        if is_mirrored_text(lines[i]):
            dropped += 1
            i += 1
            continue

        out.append(lines[i])
        i += 1

    if dropped:
        print(f"    [cleanup] usunięte odwrócone bloki: {dropped}")
    return "\n".join(out)


def _is_heading_line(line: str) -> bool:
    return bool(re.match(r"^(?:\d{1,4}\s*<br>\s*)?#{1,6}\s+", line.strip()))


def _markdown_atomic_blocks(text: str):
    """Split Markdown into atomic blocks: headings, full tables and paragraphs."""
    lines = text.split("\n")
    blocks, i, n = [], 0, len(lines)
    while i < n:
        if not lines[i].strip():
            i += 1
            continue
        if _is_heading_line(lines[i]):
            blocks.append(lines[i].rstrip())
            i += 1
            continue
        if _is_table_row(lines[i]):
            block = []
            while i < n and _is_table_row(lines[i]):
                block.append(lines[i].rstrip())
                i += 1
            blocks.append("\n".join(block))
            continue
        block = []
        while i < n and lines[i].strip() and not _is_heading_line(lines[i]) and not _is_table_row(lines[i]):
            block.append(lines[i].rstrip())
            i += 1
        blocks.append("\n".join(block))
    return [b for b in blocks if b.strip()]


_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=(?:[A-ZĄĆĘŁŃÓŚŹŻ0-9_*#]|[\"„]))"
)
_PROTECTED_ABBREVIATIONS = {
    "m.in.": "__ABBR_MIN__",
    "m. in.": "__ABBR_MIN2__",
    "np.": "__ABBR_NP__",
    "tj.": "__ABBR_TJ__",
    "tzn.": "__ABBR_TZN__",
    "itp.": "__ABBR_ITP__",
    "itd.": "__ABBR_ITD__",
    "r.": "__ABBR_R__",
    "art.": "__ABBR_ART__",
    "pkt.": "__ABBR_PKT__",
    "ust.": "__ABBR_UST__",
    "lit.": "__ABBR_LIT__",
    "mln.": "__ABBR_MLN__",
    "tys.": "__ABBR_TYS__",
    "zł.": "__ABBR_ZL__",
}


def _protect_abbreviations(text: str) -> str:
    protected = text
    for abbr, marker in _PROTECTED_ABBREVIATIONS.items():
        protected = re.sub(re.escape(abbr), marker, protected, flags=re.IGNORECASE)
    return protected


def _restore_abbreviations(text: str) -> str:
    restored = text
    for abbr, marker in _PROTECTED_ABBREVIATIONS.items():
        restored = restored.replace(marker, abbr)
    return restored


def _split_paragraph_to_sentences(paragraph: str):
    protected = _protect_abbreviations(paragraph.strip())
    parts = [_restore_abbreviations(p).strip() for p in _SENTENCE_BOUNDARY.split(protected) if p.strip()]
    return parts or [paragraph.strip()]


def _split_oversize_sentence(sentence: str, max_tokens: int):
    """Last-resort split for pathological sentences longer than the chunk budget."""
    words = sentence.split()
    chunks, current = [], []
    for word in words:
        candidate = current + [word]
        if current and count_tokens(" ".join(candidate)) > max_tokens:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current = candidate
    if current:
        chunks.append(" ".join(current))
    return chunks


def _split_large_text_block(block: str, max_tokens: int):
    """Split prose on paragraph/sentence boundaries; never cut at comma/space."""
    units = []
    for paragraph in re.split(r"\n{2,}", block):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if count_tokens(paragraph) <= max_tokens:
            units.append(paragraph)
            continue
        units.extend(_split_paragraph_to_sentences(paragraph))

    chunks, current = [], []
    for unit in units:
        if count_tokens(unit) > max_tokens:
            if current:
                chunks.append("\n\n".join(current).strip())
                current = []
            chunks.extend(_split_oversize_sentence(unit, max_tokens))
            continue
        candidate = current + [unit]
        if current and count_tokens("\n\n".join(candidate)) > max_tokens:
            chunks.append("\n\n".join(current).strip())
            current = [unit]
        else:
            current = candidate
    if current:
        chunks.append("\n\n".join(current).strip())
    return [p for p in chunks if p.strip()]


def _contains_table_row(text: str) -> bool:
    return any(_is_table_row(line) for line in text.split("\n"))


def _split_large_table_block(block: str, max_tokens: int):
    rows = [r for r in block.split("\n") if r.strip()]
    if len(rows) <= 2:
        return _split_large_text_block(block, max_tokens)
    header = rows[:2] if len(rows) > 1 and _is_sep_line(rows[1]) else rows[:1]
    data_rows = rows[len(header):]
    chunks, current = [], header[:]
    for row in data_rows:
        candidate = current + [row]
        if len(current) > len(header) and count_tokens("\n".join(candidate)) > max_tokens:
            chunks.append("\n".join(current))
            current = header[:] + [row]
        else:
            current = candidate
    if current:
        chunks.append("\n".join(current))
    return chunks


def split_markdown_structural(text: str, max_tokens: int):
    """Split without cutting table rows or paragraph blocks when possible."""
    blocks = _markdown_atomic_blocks(text)
    chunks, current = [], []

    def render(parts):
        return "\n\n".join(p for p in parts if p.strip()).strip()

    for block in blocks:
        block_tokens = count_tokens(block)
        if block_tokens > max_tokens:
            if current:
                chunks.append(render(current))
                current = []
            large_parts = (
                _split_large_table_block(block, max_tokens)
                if _is_table_row(block.split("\n", 1)[0])
                else _split_large_text_block(block, max_tokens)
            )
            chunks.extend(large_parts)
            continue

        candidate = current + [block]
        if current and count_tokens(render(candidate)) > max_tokens:
            chunks.append(render(current))
            current = [block]
        else:
            current = candidate

    if current:
        chunks.append(render(current))
    return [c for c in chunks if c.strip()]


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
    cleaned = re.sub(r"(?<!\n)\b\d{1,4}\s*<br>\s*(#{1,6}\s+)", r"\n\1", cleaned)
    cleaned = re.sub(
        r"(?:[A-ZĄĆĘŁŃÓŚŹŻ0-9]\s+){10,}[A-ZĄĆĘŁŃÓŚŹŻ0-9]",
        " ",
        cleaned,
    )
    cleaned = re.sub(
        r"(?:[A-ZĄĆĘŁŃÓŚŹŻ]\s+){1,6}(?:\d\s+){3,}\d",
        " ",
        cleaned,
    )
    cleaned = "\n".join(
        line for line in cleaned.split("\n")
        if not _is_spaced_caps_noise_line(line)
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def save_chunk_to_jsonl(
    parts,
    contexts,
    out_file_handle,
    chunk_id,
    max_garbled_risk=None,
    use_llama_repair: bool = False,
    llama_model: str = "chunk-repair:latest",
    llama_url: str = None,
    llama_timeout: int = 120,
):
    """Write a single clean JSONL entry with quality metadata.

    Returns True if written, False if skipped (garbled_risk above threshold).
    With max_garbled_risk set, chunks whose OCR-corruption score exceeds it are
    dropped (issue 4: dropped vowels) — excluding poison beats keeping it.
    """
    combined_text = "\n\n".join(parts)
    combined_context = " | ".join(sorted(_clean_contexts(contexts))) or "Główny Dokument"
    combined_text = strip_toc_blocks(combined_text)
    combined_text = strip_mirrored_blocks(combined_text)
    combined_text = strip_visual_noise_blocks(combined_text)
    combined_text = strip_nonpipe_financial_table_blobs(combined_text)
    combined_text = strip_collapsed_pipe_numeric_rows(combined_text)
    combined_text = sanitize_tables(combined_text)
    combined_text = strip_visual_noise_blocks(combined_text)
    combined_text = strip_nonpipe_financial_table_blobs(combined_text)
    combined_text = strip_collapsed_pipe_numeric_rows(combined_text)
    combined_text = trim_orphan_chunk_edges(combined_text)
    if use_llama_repair and should_llama_repair(combined_text):
        repaired = llama_repair_text(combined_text, model=llama_model, url=llama_url, timeout=llama_timeout)
        if repaired != combined_text:
            print(f"    [llama] chunk {chunk_id}: repaired")
        combined_text = trim_orphan_chunk_edges(repaired)
    final_input = f"DOKUMENT SEKCJA: {combined_context}\n\n{combined_text}"

    if len(combined_text.strip()) < 100 or is_toc_chunk(combined_text) or is_mirrored_text(combined_text):
        print(f"    [skip] chunk {chunk_id}: table of contents or mirrored text")
        return False
    if has_incomplete_chunk_boundary(combined_text):
        print(f"    [skip] chunk {chunk_id}: incomplete boundary")
        return False
    noise = visual_noise_score(combined_text)
    if noise > 0.35:
        print(f"    [skip] chunk {chunk_id}: visual_noise={noise:.4f}")
        return False
    if _has_bad_table_artifact(combined_text):
        print(f"    [skip] chunk {chunk_id}: bad_table_artifact")
        return False
    font_damage = compute_font_damage_score(combined_text)
    if has_font_encoding_damage(combined_text):
        print(f"    [skip] chunk {chunk_id}: font_damage={font_damage:.4f}")
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
        "font_damage": round(font_damage, 4),
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


def _extract_fin_values(cell: str):
    """Extract separate financial values from one jammed Markdown table cell."""
    values = []
    for m in _FINVAL.finditer(cell):
        v = re.sub(r"\s+", " ", m.group(0)).strip()
        if re.fullmatch(r"(?:19|20)\d{2}", v):
            continue
        values.append(v)
    return values


def _cell_without_values(cell: str) -> str:
    stripped = _FINVAL.sub(" ", cell)
    stripped = re.sub(r"[*_`]+", "", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip(" |:-")
    return stripped


def _repair_jammed_table_block(block) -> str:
    """Repair a jammed Markdown table block without using page geometry.

    This handles the common pymupdf4llm failure mode where several period
    values are crammed into one cell, or where the label is emitted on one row
    and all values on the next row. It is intentionally conservative: if it
    cannot produce multiple labelled rows with numeric values, it returns "" so
    the caller can fall back to the existing drop behavior.
    """
    rows, pending_label = [], None
    for line in block:
        if _is_sep_line(line):
            continue
        cells = _split_cells(line)
        vals, label_bits = [], []
        for cell in cells:
            cell_vals = _extract_fin_values(cell)
            vals.extend(cell_vals)
            label = _cell_without_values(cell)
            if label and not re.fullmatch(r"[-.,%() ]+", label):
                label_bits.append(label)

        label_text = " ".join(label_bits).strip()
        if vals and label_text:
            rows.append([label_text] + vals)
            pending_label = None
        elif vals and pending_label:
            rows.append([pending_label] + vals)
            pending_label = None
        elif label_text:
            pending_label = f"{pending_label} {label_text}".strip() if pending_label else label_text

    labelled = [r for r in rows if len(r) >= 3 and re.search(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{4,}", r[0])]
    if len(labelled) < 2:
        return ""
    width_counts = [len(r) for r in labelled]
    target = max(set(width_counts), key=width_counts.count)
    consistent = sum(1 for w in width_counts if abs(w - target) <= 1) / len(width_counts)
    if consistent < 0.5:
        return ""

    width = max(len(r) for r in labelled)
    out = []
    for i, r in enumerate(labelled):
        r = r + [""] * (width - len(r))
        out.append("| " + " | ".join(r) + " |")
        if i == 0 and width >= 2:
            out.append("| " + " | ".join(["---"] * width) + " |")
    return "\n".join(out)


def _repair_jammed_table_blocks(md: str):
    """Replace jammed table blocks with conservative Markdown-level repairs."""
    lines = md.split("\n")
    out, i, n, repaired = [], 0, len(lines), 0
    while i < n:
        if not _is_table_row(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        block = []
        while i < n and _is_table_row(lines[i]):
            block.append(lines[i])
            i += 1
        block_text = "\n".join(block)
        if _has_value_jam(block_text):
            fixed = _repair_jammed_table_block(block)
            if fixed:
                out.append(fixed)
                repaired += 1
            else:
                out.extend(block)
        else:
            out.extend(block)
    return "\n".join(out), repaired


def _table_matrix_to_markdown(table) -> str:
    rows = []
    for raw in table or []:
        row = [re.sub(r"\s+", " ", (c or "").replace("\n", " ")).strip() for c in raw]
        if any(row):
            rows.append(row)
    if len(rows) < 2:
        return ""
    width = max(len(r) for r in rows)
    norm = []
    for r in rows:
        r = r + [""] * (width - len(r))
        norm.append(r)
    out = []
    for i, r in enumerate(norm):
        out.append("| " + " | ".join(r) + " |")
        if i == 0 and width >= 2:
            out.append("| " + " | ".join(["---"] * width) + " |")
    return "\n".join(out)


def _validate_external_table_md(md: str) -> bool:
    if not md or _has_value_jam(md):
        return False
    rows = [_split_cells(l) for l in md.split("\n") if _is_table_row(l) and not _is_sep_line(l)]
    labelled = 0
    numeric_rows = 0
    for cells in rows:
        if _row_has_label(cells):
            labelled += 1
        nums = [c for c in cells if _is_numcell(c)]
        if len(nums) >= 2:
            numeric_rows += 1
    return labelled >= 2 and numeric_rows >= 2


def _pdfplumber_page_tables(pdf_path: str, page_index: int) -> str:
    if pdfplumber is None:
        return ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_index]
            tables = page.extract_tables({
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "intersection_tolerance": 5,
            })
            if not tables:
                tables = page.extract_tables({
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                })
    except Exception:
        return ""

    md_tables = []
    for table in tables or []:
        md = _table_matrix_to_markdown(table)
        if _validate_external_table_md(md):
            md_tables.append(md)
    return "\n\n".join(md_tables)


def _ocr_page_text(page, require_financial: bool = True) -> str:
    if pytesseract is None:
        return ""
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        img = pix.pil_image()
        text = pytesseract.image_to_string(img, lang="pol+eng", config="--psm 6")
    except Exception:
        return ""
    lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    numeric = sum(1 for l in lines if re.search(r"\d", l))
    labelled = sum(1 for l in lines if re.search(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]{4,}", l))
    if require_financial and (numeric < 3 or labelled < 3):
        return ""
    if not require_financial and labelled < 3 and len(" ".join(lines)) < 250:
        return ""
    return "\n".join(lines)


def _external_table_fallback(pdf_path: str, page, page_index: int):
    """Final fallback for failed jam pages: structured PDF tables, then OCR text."""
    md = _pdfplumber_page_tables(pdf_path, page_index)
    if md:
        return md, "pdfplumber"
    ocr = _ocr_page_text(page)
    if ocr:
        return ocr, "ocr"
    return "", ""


def _fallback_quality_score(text: str) -> float:
    if not text or len(text.strip()) < 80:
        return 1.0
    return max(
        compute_font_damage_score(text),
        compute_garbled_score(text),
        visual_noise_score(text),
    )


def _clean_ocr_page_candidate(text: str) -> str:
    if not text:
        return ""
    cleaned = clean_markdown_artifacts(text)
    cleaned = remove_page_headers(cleaned)
    cleaned = strip_mirrored_blocks(cleaned)
    cleaned = strip_visual_noise_blocks(cleaned)
    cleaned = clean_loose_semicolons(cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _recover_font_damaged_page(page, current_md: str) -> str:
    """Use OCR only when a page has broken font encoding and OCR is safer."""
    if not has_font_encoding_damage(current_md):
        return current_md
    ocr = _clean_ocr_page_candidate(_ocr_page_text(page, require_financial=False))
    if not ocr:
        return current_md
    current_score = _fallback_quality_score(current_md)
    ocr_score = _fallback_quality_score(ocr)
    min_len = max(180, int(len(current_md.strip()) * 0.35))
    if len(ocr.strip()) < min_len:
        return current_md
    if ocr_score <= 0.08 and ocr_score <= max(0.02, current_score * 0.70):
        return ocr
    return current_md


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
    md = expand_embedded_date_columns(md)
    md = fix_br_in_tables(md)
    md = strip_nonpipe_financial_table_blobs(md)
    md = strip_collapsed_pipe_numeric_rows(md)
    md = sanitize_tables(md)
    md = strip_mirrored_blocks(md)
    md = strip_visual_noise_blocks(md)
    md = strip_nonpipe_financial_table_blobs(md)
    md = strip_collapsed_pipe_numeric_rows(md)
    md = sanitize_tables(md)
    md = strip_visual_noise_blocks(md)
    md = strip_nonpipe_financial_table_blobs(md)
    md = strip_collapsed_pipe_numeric_rows(md)
    md = clean_loose_semicolons(md)
    return md


def extract_clean_markdown(pdf_path: str, use_global_ocr: bool = False) -> str:
    """Page-aware extraction with two-tier table rescue. Returns the full cleaned
    Markdown for the document (consumed by the chunker).

    Global OCR is disabled by default. It is slow on long born-digital reports
    and is not needed for normal text extraction; OCR remains available below as
    the final per-page fallback for unresolved jammed tables.
    """
    pages = pymupdf4llm.to_markdown(
        pdf_path,
        page_chunks=True,
        show_progress=False,
        use_ocr=use_global_ocr,
        force_ocr=False,
        ocr_language="pol+eng",
    )
    doc = fitz.open(pdf_path)
    out_pages = []
    rescued_n = fallback_n = external_n = pdfplumber_n = ocr_n = dropped_n = font_ocr_n = 0
    for i, pg in enumerate(pages):
        md = _clean_page(pg["text"])
        recovered_md = _recover_font_damaged_page(doc[i], md)
        if recovered_md != md:
            md = recovered_md
            font_ocr_n += 1
        if _has_value_jam(md):
            rows = _positional_rows(doc[i])
            if _validate_rescue(rows):
                prose = _strip_table_blocks(md, only_jammed=False)   # drop jammed garbage tables
                rescued = _rows_to_markdown(rows)
                md = (prose.rstrip() + "\n\n" + rescued).strip()
                rescued_n += 1
            else:
                repaired, fixed_blocks = _repair_jammed_table_blocks(md)
                if fixed_blocks and not _has_value_jam(repaired):
                    md = repaired
                    fallback_n += 1
                else:
                    external, external_kind = _external_table_fallback(pdf_path, doc[i], i)
                    if external and not _has_value_jam(external):
                        prose = _strip_table_blocks(md, only_jammed=True)
                        md = (prose.rstrip() + "\n\n" + external).strip()
                        external_n += 1
                        if external_kind == "pdfplumber":
                            pdfplumber_n += 1
                        elif external_kind == "ocr":
                            ocr_n += 1
                    else:
                        md = _strip_table_blocks(md, only_jammed=True)   # drop only jammed; keep rest
                        dropped_n += 1
        out_pages.append(md)
    if rescued_n or fallback_n or external_n or dropped_n or font_ocr_n:
        print(
            f"    [tier-2] strony z jamem: rescue={rescued_n}, "
            f"fallback={fallback_n}, external={external_n}, "
            f"pdfplumber={pdfplumber_n}, ocr={ocr_n}, "
            f"font_ocr={font_ocr_n}, pominięte_tabele={dropped_n}"
        )
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
    use_global_ocr: bool = False,
    use_llama_repair: bool = False,
    llama_model: str = "chunk-repair:latest",
    llama_url: str = None,
    llama_timeout: int = 120,
):
    """Extract a PDF into the list of final 'input' strings (the exact text fed to
    the model under '### Input:'). Applies Tier-1+Tier-2 extraction, chunking and
    the garbled-risk filter. Returns list[str]."""
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "Naglowek_Glowny"), ("##", "Sekcja"), ("###", "Podsekcja")])
    markdown_content = extract_clean_markdown(pdf_path, use_global_ocr=use_global_ocr)
    markdown_content = demote_spurious_headings(markdown_content)
    sections = md_splitter.split_text(markdown_content)

    inputs = []
    buffer_parts, buffer_tokens, buffer_contexts = [], 0, set()

    def _emit(parts, contexts):
        combined = strip_toc_blocks("\n\n".join(parts))
        combined = strip_mirrored_blocks(combined)
        combined = strip_visual_noise_blocks(combined)
        combined = strip_nonpipe_financial_table_blobs(combined)
        combined = strip_collapsed_pipe_numeric_rows(combined)
        combined = sanitize_tables(combined)
        combined = strip_visual_noise_blocks(combined)
        combined = strip_nonpipe_financial_table_blobs(combined)
        combined = strip_collapsed_pipe_numeric_rows(combined)
        combined = trim_orphan_chunk_edges(combined)
        if use_llama_repair and should_llama_repair(combined):
            repaired = llama_repair_text(combined, model=llama_model, url=llama_url, timeout=llama_timeout)
            if repaired != combined:
                print("    [llama] input chunk: repaired")
            combined = trim_orphan_chunk_edges(repaired)
        if len(combined.strip()) < min_characters or is_toc_chunk(combined) or is_mirrored_text(combined):
            return
        if has_incomplete_chunk_boundary(combined):
            return
        if visual_noise_score(combined) > 0.35:
            return
        if _has_bad_table_artifact(combined):
            return
        if has_font_encoding_damage(combined):
            return
        if max_garbled_risk is not None and compute_garbled_score(combined) > max_garbled_risk:
            return
        ctx = " | ".join(sorted(_clean_contexts(contexts))) or "Główny Dokument"
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
            current_table_header = None
            for sub_text in split_markdown_structural(reconstructed_text, max_tokens):
                sub_text = strip_toc_blocks(sub_text)
                detected_header = extract_table_header(sub_text)
                if detected_header:
                    current_table_header = detected_header
                elif _contains_table_row(sub_text) and "|---" not in sub_text.replace(" ", "") and current_table_header:
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
    use_global_ocr: bool = False,
    use_llama_repair: bool = False,
    llama_model: str = "chunk-repair:latest",
    llama_url: str = None,
    llama_timeout: int = 120,
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

        inputs = extract_inputs_from_pdf(
            pdf_path,
            max_tokens,
            min_characters,
            max_garbled_risk,
            use_global_ocr=use_global_ocr,
            use_llama_repair=use_llama_repair,
            llama_model=llama_model,
            llama_url=llama_url,
            llama_timeout=llama_timeout,
        )

        with open(output_file, "w", encoding="utf-8") as out_f:
            for cid, final_input in enumerate(inputs, 1):
                row = {
                    "id": cid,
                    "instruction": INSTRUCTION_TEMPLATE,
                    "input": final_input,
                    "output": "",
                    "garbled_risk": round(compute_garbled_score(final_input), 4),
                    "font_damage": round(compute_font_damage_score(final_input), 4),
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  → Zapisano {len(inputs)} chunków")
    return


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract clean JSONL chunks from PDF reports.")
    parser.add_argument("--pdf-dir", default="/app/fine-tuning-scripts/materials")
    parser.add_argument("--out-dir", default="/app/fine-tuning-scripts/_extract_out")
    parser.add_argument("--max-tokens", type=int, default=3700)
    parser.add_argument("--min-characters", type=int, default=800)
    parser.add_argument("--max-garbled-risk", type=float, default=0.05)
    parser.add_argument("--global-ocr", action="store_true", help="Use global OCR in pymupdf4llm; normally keep disabled.")
    parser.add_argument(
        "--enable-llama-repair",
        action="store_true",
        help="Enable optional Ollama/Llama repair pass for suspicious chunks only.",
    )
    parser.add_argument("--llama-model", default="chunk-repair:latest")
    parser.add_argument("--llama-url", default=os.environ.get("OLLAMA_REPAIR_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--llama-timeout", type=int, default=120)
    args = parser.parse_args()

    process_pdf_to_clean_dataset(
        pdf_dir=args.pdf_dir,
        out_dir=args.out_dir,
        max_tokens=args.max_tokens,
        min_characters=args.min_characters,
        max_garbled_risk=args.max_garbled_risk,
        use_global_ocr=args.global_ocr,
        use_llama_repair=args.enable_llama_repair,
        llama_model=args.llama_model,
        llama_url=args.llama_url,
        llama_timeout=args.llama_timeout,
    )
