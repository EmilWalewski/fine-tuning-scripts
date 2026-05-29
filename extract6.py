import os
import json
import re
import tiktoken
import pymupdf4llm
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


def remove_page_headers(text: str) -> str:
    """Strip repeating page headers / footers that interrupt running text."""
    cleaned = _PAGE_HEADER_SOFT.sub(" ", text)
    cleaned = _PAGE_HEADER_BARE.sub("\n", cleaned)
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


def _replace_br_in_cell(cell: str) -> str:
    if not cell:
        return cell
    # Bullet cell – handled at row level
    if re.match(r"[▪•]\s*<br>", cell):
        return cell
    # Numeric/date stack (multi-period financial column) → explicit separator,
    # NEVER a plain space (that is what was flattening the figures).
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
    # ── 1) separator hygiene: dominant data-column count, keep one valid sep ──
    data_cols = [len(_split_cells(l)) for l in block if not _is_sep_line(l)]
    dom = max(set(data_cols), key=data_cols.count) if data_cols else None
    cleaned, seen_sep = [], False
    for l in block:
        if _is_sep_line(l):
            if seen_sep:
                continue                          # drop duplicate separators
            if dom and len(_split_cells(l)) != dom:
                continue                          # drop column-count mismatch
            seen_sep = True
        cleaned.append(l)

    # ── 2) merge word-wrapped label-only fragments into data rows ─────────────
    def _append_backward(txt):
        pc = _split_cells(result[last_data])
        pc[0] = f"{pc[0]} {txt}".strip()
        result[last_data] = _join_cells(pc)

    def _flush_pending():
        nonlocal pending_fwd
        if pending_fwd:
            # no data row followed → attach to previous data row if any, else keep
            if last_data is not None:
                _append_backward(pending_fwd)
            else:
                result.append(_join_cells([pending_fwd]))
            pending_fwd = None

    result, pending_fwd, last_data = [], None, None
    for l in cleaned:
        if _is_sep_line(l):
            _flush_pending(); result.append(l); continue
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

        # normal data / header row
        if pending_fwd:
            cells[0] = f"{pending_fwd} {cells[0]}".strip()
            pending_fwd = None
        result.append(_join_cells(cells))
        last_data = len(result) - 1
    _flush_pending()
    return result


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
    return "......." in lower or "spis treści" in lower or "contents" in lower


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
    final_input = f"DOKUMENT SEKCJA: {combined_context}\n\n{combined_text}"

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
# Main processing pipeline
# ---------------------------------------------------------------------------

def process_pdf_to_clean_dataset(
    pdf_dir: str,
    out_dir: str,
    max_tokens: int = 3700,
    min_characters: int = 800,
    max_garbled_risk: float = None,   # issue 4: drop chunks with OCR-corruption above this (e.g. 0.05)
):
    os.makedirs(out_dir, exist_ok=True)

    headers_to_split_on = [
        ("#", "Naglowek_Glowny"),
        ("##", "Sekcja"),
        ("###", "Podsekcja"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    for pdf_file in sorted(os.listdir(pdf_dir)):
        if not pdf_file.endswith(".pdf"):
            continue

        pdf_path = os.path.join(pdf_dir, pdf_file)
        name_only = os.path.splitext(pdf_file)[0]
        output_file = os.path.join(out_dir, f"{name_only}-clean.jsonl")
        print(f"\nProcesowanie: {pdf_file} → {output_file}")

        # ── Step 1: extract raw Markdown ────────────────────────────────────
        raw_markdown = pymupdf4llm.to_markdown(pdf_path)

        # ── Step 2: all cleaning passes in order ────────────────────────────
        markdown_content = clean_markdown_artifacts(raw_markdown)  # picture noise
        markdown_content = remove_page_headers(markdown_content)   # FIX 1
        markdown_content = fix_br_in_tables(markdown_content)      # FIX 2
        markdown_content = sanitize_tables(markdown_content)       # FIX 2b (issues 2 & 3)

        # ── Step 3: split by Markdown headings ──────────────────────────────
        sections = md_splitter.split_text(markdown_content)

        buffer_parts: list = []
        buffer_tokens: int = 0
        buffer_contexts: set = set()
        file_saved_count = 1

        with open(output_file, "w", encoding="utf-8") as out_f:
            for doc in sections:
                # Build context breadcrumb
                context_hierarchy = [
                    doc.metadata[h]
                    for h in ("Naglowek_Glowny", "Sekcja", "Podsekcja")
                    if h in doc.metadata
                ]
                context_string = (
                    " > ".join(context_hierarchy) if context_hierarchy else "Główny Dokument"
                )

                # Reconstruct heading prefixes for readability
                reconstructed_text = ""
                if "Naglowek_Glowny" in doc.metadata:
                    reconstructed_text += f"# {doc.metadata['Naglowek_Glowny']}\n"
                if "Sekcja" in doc.metadata:
                    reconstructed_text += f"## {doc.metadata['Sekcja']}\n"
                if "Podsekcja" in doc.metadata:
                    reconstructed_text += f"### {doc.metadata['Podsekcja']}\n"
                reconstructed_text += doc.page_content

                if len(reconstructed_text.strip()) < 100:
                    continue

                doc_tokens = count_tokens(reconstructed_text)

                # ── Case A: single section exceeds budget → sub-split ────────
                if doc_tokens > max_tokens:
                    if buffer_parts:
                        if save_chunk_to_jsonl(
                            buffer_parts, buffer_contexts, out_f, file_saved_count, max_garbled_risk
                        ):
                            file_saved_count += 1
                        buffer_parts, buffer_tokens, buffer_contexts = [], 0, set()

                    token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                        encoding_name="cl100k_base",
                        chunk_size=max_tokens,
                        chunk_overlap=400,
                        separators=["\n\n", "\n|", "\n", ". ", "? ", "! ", " "],
                    )
                    sub_chunks = token_splitter.split_text(reconstructed_text)

                    current_table_header = None
                    for sub_text in sub_chunks:
                        detected_header = extract_table_header(sub_text)
                        if detected_header:
                            current_table_header = detected_header
                        elif (
                            "|" in sub_text
                            and "|---" not in sub_text.replace(" ", "")
                            and current_table_header
                        ):
                            sub_text = current_table_header + "\n" + sub_text

                        if len(sub_text) >= min_characters and not is_toc_chunk(sub_text):
                            if save_chunk_to_jsonl(
                                [sub_text], {context_string}, out_f, file_saved_count, max_garbled_risk
                            ):
                                file_saved_count += 1
                    continue

                # ── Case B: section doesn't fit → flush buffer ───────────────
                if buffer_tokens + doc_tokens > max_tokens:
                    if buffer_parts and buffer_parts[0].strip():
                        if save_chunk_to_jsonl(
                            buffer_parts, buffer_contexts, out_f, file_saved_count, max_garbled_risk
                        ):
                            file_saved_count += 1
                    buffer_parts, buffer_tokens, buffer_contexts = [], 0, set()

                # ── Accumulate ───────────────────────────────────────────────
                buffer_parts.append(reconstructed_text)
                buffer_tokens += doc_tokens
                buffer_contexts.add(context_string)

            # Flush remaining buffer
            if buffer_parts:
                save_chunk_to_jsonl(
                    buffer_parts, buffer_contexts, out_f, file_saved_count, max_garbled_risk
                )

        print(f"  → Zapisano ~{file_saved_count} chunków")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- ścieżki produkcyjne (kontener) — zakomentowane na czas lokalnych testów ---
    # process_pdf_to_clean_dataset(
    #     pdf_dir="/app/data_new",
    #     out_dir="/app/prepare-dataset/dataset-to-process",
    #     max_tokens=3700,
    #     min_characters=800,
    # )

    # --- ścieżki lokalne (pełny zestaw źródeł z materials/) ---
    process_pdf_to_clean_dataset(
        pdf_dir="/Users/ewalewski/python/fine-tuning-scripts/materials",
        out_dir="/Users/ewalewski/python/fine-tuning-scripts/_extract_out",
        max_tokens=3700,
        min_characters=800,
        max_garbled_risk=0.05,   # issue 4: odrzuć chunki z silną korupcją OCR (samogłoski)
    )
