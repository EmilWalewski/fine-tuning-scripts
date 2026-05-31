from flask import Flask, request, jsonify
import os
import json
import re
import tempfile
import tiktoken
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from llama_index.llms.ollama import Ollama

app = Flask(__name__)

# --- CONFIGURATION MATCHING TRAINING ---

INSTRUCTION_TEMPLATE = (
    "You are a Senior Strategic Corporate Analyst. Summarize the report excerpt into a high-density English executive summary.\n"
    "STRICT COMMANDS:\n"
    "1. SOURCE FIDELITY: Use ONLY the provided text. Never use external knowledge or general history.\n"
    "2. FINANCIAL ACCURACY: Extract all material figures (revenue, costs, profit). Normalize units: convert 'tys.' (thousands) to M (millions). Example: 4,156,476k PLN -> 4,156.5M PLN. Always state the currency.\n"
    "3. STRATEGIC EVENTS: Explicitly include milestones like acquisitions (zakup), dividends, or board changes.\n"
    "4. TELEGRAPHIC STYLE: Write a continuous, professional narrative. Skip introductions, transitions, and bullet points. Use dense noun phrases.\n"
    "5. NO HALLUCINATION: If a specific data point (e.g. effective tax rate) is not in the text, omit it. Do not invent boilerplate or advisory names."
)

OLLAMA_LLM_MODEL = "atlas"  # Nazwa Twojego utworzonego modelu w Ollamie

# Inicjalizacja LLM z parametrami z Modelfile
llm = Ollama(
    model=OLLAMA_LLM_MODEL,
    request_timeout=1800.0,
    additional_kwargs={
        "num_ctx": 5120,
        "temperature": 0,
        "num_predict": 1000,
    }
)

# ---------------------------------------------------------------------------
# Tokenizer & Regex Patterns (Exact copies from your script)
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


_PAGE_HEADER_SOFT = re.compile(
    r"  \n"
    r"(?:\*\*[^\n]+\*\* \n){1,4}"
    r"\d{1,4}  \n",
    re.MULTILINE,
)

_PAGE_HEADER_BARE = re.compile(
    r"\n(?:\*\*[^\n]+\*\*\s*\n){1,4}\d{1,4}\s*\n",
    re.MULTILINE,
)

_PAGE_FOOTER_NUM_URL = re.compile(
    r"\n\s*\d{1,4}\s*\n\s*(?:www\.)?[a-z0-9-]+\.(?:com|pl|eu|net|org)\s*\n",
    re.MULTILINE | re.IGNORECASE,
)
_PAGE_FOOTER_URL_NUM = re.compile(
    r"\n\s*(?:www\.)?[a-z0-9-]+\.(?:com|pl|eu|net|org)\s*\n\s*\d{1,4}\s*\n",
    re.MULTILINE | re.IGNORECASE,
)
_BARE_URL_LINE = re.compile(
    r"\n\s*(?:www\.)?[a-z0-9-]{2,}\.(?:com|pl|eu|net|org)\s*\n",
    re.MULTILINE | re.IGNORECASE,
)


def remove_page_headers(text: str) -> str:
    cleaned = _PAGE_HEADER_SOFT.sub(" ", text)
    cleaned = _PAGE_HEADER_BARE.sub("\n", cleaned)
    cleaned = _PAGE_FOOTER_NUM_URL.sub(" ", cleaned)
    cleaned = _PAGE_FOOTER_URL_NUM.sub(" ", cleaned)
    cleaned = _BARE_URL_LINE.sub("\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _seg_is_numeric(s: str) -> bool:
    s2 = s.strip().strip("_*").strip()
    if not any(c.isdigit() for c in s2):
        return False
    return bool(re.fullmatch(r"\(?-?[\d\s .,%]+\)?", s2))


def _is_value_stack(cell: str) -> bool:
    segs = [s.strip() for s in cell.split("<br>") if s.strip()]
    if len(segs) < 2:
        return False
    numeric = sum(1 for s in segs if _seg_is_numeric(s))
    return numeric >= max(2, int(len(segs) * 0.6))


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
    if re.match(r"[▪•]\s*<br>", cell):
        return cell
    if "<br>" in cell and _is_value_stack(cell):
        return re.sub(r"\s*<br>\s*", " ; ", cell)
    if cell.startswith("**") and "<br>" in cell:
        return re.sub(r"\*\*\s*<br>\s*\*\*", " ", cell).replace("<br>", " ")
    if cell.startswith("_") and "<br>" in cell:
        return cell.replace("<br>", " ")
    if re.match(r"\d{2}\.\d{2}\.\d{4}", cell):
        return cell.replace("<br>", " ; ")
    return cell.replace("<br>", " ")


def _row_has_bullet_br(row: str) -> bool:
    return bool(re.search(r"[▪•]\s*<br>", row))


def _convert_bullet_row_to_prose(row: str) -> str:
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
    rows = [l for l in block if not _is_sep_line(l)]

    def _append_backward(txt):
        pc = _split_cells(result[last_data])
        pc[0] = f"{pc[0]} {txt}".strip()
        result[last_data] = _join_cells(pc)

    result, pending_fwd, last_data = [], None, None
    for l in rows:
        cells = _split_cells(l)
        multi_empty = len(cells) > 1 and all(c == "" for c in cells[1:])
        single_cont = len(cells) == 1 and cells[0][:1].islower()

        if multi_empty:
            txt = cells[0]
            if not txt:
                continue
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

    if not result:
        return result
    counts = [len(_split_cells(r)) for r in result]
    width = max(set(counts), key=counts.count)
    grid = []
    for r in result:
        c = _split_cells(r)
        if len(c) > width:
            c = c[:width - 1] + [" ".join(x for x in c[width - 1:] if x).strip()]
        elif len(c) < width:
            c = c + [""] * (width - len(c))
        grid.append(c)
    norm = [_join_cells(c) for c in grid]
    if width >= 2:
        sep = "| " + " | ".join(["---"] * width) + " |"
        norm = [norm[0], sep] + norm[1:]
    return norm


def clean_loose_semicolons(text: str) -> str:
    out = []
    for ln in text.split("\n"):
        if " ; " in ln and not ln.strip().startswith("|"):
            toks = ln.split()
            numish = sum(1 for t in toks if re.fullmatch(r"\(?-?[\d.,%)]+;?", t))
            if toks and numish / len(toks) >= 0.6:
                continue
            ln = ln.replace(" ; ", " ")
        out.append(ln)
    return "\n".join(out)


def sanitize_tables(text: str) -> str:
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


_POLISH_VOWELS = set("aeiouąęóyAEIOUĄĘÓY")
_IGNORE_GARBLE = re.compile(
    r"^[-|_=*#>\[\](){}/\\]+$"
    r"|^\d"
    r"|^https?://"
    r"|^[A-Z]{2,6}$"
)
_KNOWN_ABBREVS = re.compile(r"Dvlpmt|XTB|GPW|MSSF|MSR|IFRS|ESG|PLN|EUR|USD|CEO|CFO")


def _token_consonant_ratio(token: str) -> float:
    clean = re.sub(r"<br>", "", token)
    clean = re.sub(r"^[*_.,;:!?'\"()\[\]|<>]+|[*_.,;:!?'\"()\[\]|<>]+$", "", clean)
    letters = [c for c in clean if c.isalpha()]
    if len(letters) < 5:
        return 0.0
    consonants = sum(1 for c in letters if c not in _POLISH_VOWELS)
    return consonants / len(letters)


def compute_garbled_score(text: str) -> float:
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
    cleaned = re.sub(r"\*\*==> picture \[\d+ x \d+\] intentionally omitted <==\*\*", "", text)
    cleaned = re.sub(r"\*\*----- End of picture text ----- \*\*", "", cleaned)
    cleaned = re.sub(r"\*\*----- Start of picture text -----\*\*\s*<br>?\s*", "", cleaned)
    cleaned = re.sub(r"\*\*----- End of picture text -----\*\*\s*<br>?\s*", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# FLASK ENDPOINT PROCESSING PIPELINE
# ---------------------------------------------------------------------------

@app.route('/analyze_report', methods=['POST'])
def analyze_report():
    if 'file' not in request.files:
        return jsonify({"error": "No file shared"}), 400

    file = request.files['file']

    # Parametry równe wartościom domyślnym z Twojego skryptu pipeline'u
    max_tokens = 3700
    min_characters = 800
    max_garbled_risk = 0.05

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, file.filename)
            file.save(pdf_path)

            # ── KROK 1: Ekstrakcja surowego Markdown za pomocą pymupdf4llm ──
            raw_markdown = pymupdf4llm.to_markdown(pdf_path)

            # ── KROK 2: Wszystkie filtry czyszczące w identycznej kolejności ──
            markdown_content = clean_markdown_artifacts(raw_markdown)
            markdown_content = remove_page_headers(markdown_content)
            markdown_content = expand_value_columns(markdown_content)
            markdown_content = fix_br_in_tables(markdown_content)
            markdown_content = sanitize_tables(markdown_content)
            markdown_content = clean_loose_semicolons(markdown_content)

            # ── KROK 3: Podział na sekcje na podstawie nagłówków Markdown ──
            headers_to_split_on = [
                ("#", "Naglowek_Glowny"),
                ("##", "Sekcja"),
                ("###", "Podsekcja"),
            ]
            md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            sections = md_splitter.split_text(markdown_content)

            # --- KOLEKCJONOWANIE I PAKOWANIE CHUNKÓW DO INFERENCJI ---
            inputs_to_process = []

            buffer_parts = []
            buffer_tokens = 0
            buffer_contexts = set()

            def process_and_add_buffer(parts, contexts):
                combined_text = "\n\n".join(parts)
                combined_context = " | ".join(sorted(list(contexts)))

                garbled = compute_garbled_score(combined_text)
                if garbled > max_garbled_risk:
                    print(f"    [skip buffer] garbled_risk={garbled:.4f} > {max_garbled_risk}")
                    return

                final_input = f"DOKUMENT SEKCJA: {combined_context}\n\n{combined_text}"
                inputs_to_process.append(final_input)

            for doc in sections:
                context_hierarchy = [
                    doc.metadata[h]
                    for h in ("Naglowek_Glowny", "Sekcja", "Podsekcja")
                    if h in doc.metadata
                ]
                context_string = (
                    " > ".join(context_hierarchy) if context_hierarchy else "Główny Dokument"
                )

                # Rekonstrukcja nagłówków wyjściowych
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

                # ── Przypadek A: Pojedyncza sekcja przekracza okno -> Awaryjny split (Tiktoken) ──
                if doc_tokens > max_tokens:
                    if buffer_parts:
                        process_and_add_buffer(buffer_parts, buffer_contexts)
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
                            garbled = compute_garbled_score(sub_text)
                            if garbled > max_garbled_risk:
                                print(f"    [skip sub_chunk] garbled_risk={garbled:.4f} > {max_garbled_risk}")
                                continue

                            final_input = f"DOKUMENT SEKCJA: {context_string}\n\n{sub_text}"
                            inputs_to_process.append(final_input)
                    continue

                # ── Przypadek B: Nowa sekcja nie mieści się w obecnym buforze -> opróżnij bufor ──
                if buffer_tokens + doc_tokens > max_tokens:
                    if buffer_parts and buffer_parts[0].strip():
                        process_and_add_buffer(buffer_parts, buffer_contexts)
                    buffer_parts, buffer_tokens, buffer_contexts = [], 0, set()

                # Akumulacja standardowa do bufora
                buffer_parts.append(reconstructed_text)
                buffer_tokens += doc_tokens
                buffer_contexts.add(context_string)

            # Czyszczenie pozostałości bufora po pętli
            if buffer_parts:
                process_and_add_buffer(buffer_parts, buffer_contexts)

            # ── KROK 4: Inferencja LLM (Dokładny format struktury treningowej) ──
            results = []
            print(f"Przetwarzanie {len(inputs_to_process)} gotowych chunków przez LLM...")

            for i, chunk_input in enumerate(inputs_to_process):
                full_prompt = (
                    f"### Instruction:\n{INSTRUCTION_TEMPLATE}\n\n"
                    f"### Input:\n{chunk_input}\n\n"
                    f"### Output:\n"
                )

                print(f"Generowanie analizy dla fragmentu {i+1}/{len(inputs_to_process)}...")
                response = llm.complete(full_prompt)
                results.append(response.text.strip())

            return jsonify({
                "status": "success",
                "analysis": results,
                "chunks_count": len(inputs_to_process)
            })

    except Exception as e:
        print(f"Error encountered: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
