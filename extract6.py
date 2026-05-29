import json
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import os

INSTRUCTION_TEMPLATE = (
    "You are a Senior Strategic Corporate Analyst. Summarize the report excerpt into a high-density English executive summary.\n"
    "STRICT COMMANDS:\n"
    "1. SOURCE FIDELITY: Use ONLY the provided text. Never use external knowledge or general history.\n"
    "2. FINANCIAL ACCURACY: Extract all material figures (revenue, costs, profit). Normalize units: convert 'tys.' (thousands) to M (millions). Example: 4,156,476k PLN -> 4,156.5M PLN. Always state the currency.\n"
    "3. STRATEGIC EVENTS: Explicitly include milestones like acquisitions (zakup), dividends, or board changes.\n"
    "4. TELEGRAPHIC STYLE: Write a continuous, professional narrative. Skip introductions, transitions, and bullet points. Use dense noun phrases.\n"
    "5. NO HALLUCINATION: If a specific data point (e.g. effective tax rate) is not in the text, omit it. Do not invent boilerplate or advisory names."
)

def is_toc_chunk(tekst):
    text_lower = tekst.lower()
    if "......." in text_lower or "spis treści" in text_lower or "contents" in text_lower:
        return True
    return False

def process_pdf_to_jsonl_fast_with_filter(pdf_dir, max_tokens=6000, min_characters=800, skip_existing=True):

    for pdf_file in os.listdir(pdf_dir):
        if not pdf_file.endswith(".pdf"): continue
        pdf_path = os.path.join(pdf_dir, pdf_file)
        name_only = os.path.splitext(pdf_file)[0]

        out_dir = "/app/prepare-dataset/dataset-to-process"
        os.makedirs(out_dir, exist_ok=True)
        file_to_process = os.path.join(out_dir, f"{name_only}.jsonl")

        if skip_existing and os.path.exists(file_to_process) and os.path.getsize(file_to_process) > 0:
            print(f"\n>>> Pomijam: {name_only}.pdf (Plik wynikowy już istnieje)")
            continue

        print(f"1. [KONWERSJA] Konwertuję PDF do Markdown...")
        markdown_content = pymupdf4llm.to_markdown(pdf_path)

        print("2. [PODZIAŁ STRUKTURALNY] Analizuję nagłówki...")
        headers_to_split_on = [
            ("#", "Naglowek_Glowny"),
            ("##", "Sekcja"),
            ("###", "Podsekcja"),
        ]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        sections = md_splitter.split_text(markdown_content)

        print(f"3. [KONTROLA TOKENÓW] Dzielę na chunki (max {max_tokens} tokenów)...")
        token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=max_tokens,
            chunk_overlap=300
        )
        final_chunks = token_splitter.split_documents(sections)

        print(f"4. [FILTROWANIE I ZAPIS] Buduję plik JSONL i odrzucam zbyt krótkie fragmenty...")

        saved_count = 0
        discarded_count = 0

        with open(file_to_process, "w", encoding="utf-8") as out_f:
            for idx, chunk in enumerate(final_chunks):

                # --- NOWY WARUNEK: Odrzucanie zbyt krótkich chunków ---
                chunk_length = len(chunk.page_content)
                if chunk_length < min_characters:
                    discarded_count += 1
                    # Opcjonalnie możesz odkomentować linijkę poniżej, żeby widzieć co usuwasz:
                    # print(f"   [Odrzucono] Chunk {idx+1} miał tylko {chunk_length} znaków.")
                    continue

                # Jeśli chunk spełnia warunek, budujemy dla niego kontekst z nagłówków
                context_hierarchy = []
                for header_type in ["Naglowek_Glowny", "Sekcja", "Podsekcja"]:
                    if header_type in chunk.metadata:
                        context_hierarchy.append(chunk.metadata[header_type])

                context_string = " > ".join(context_hierarchy)
                final_input = f"DOKUMENT SEKCJA: {context_string}\n\n{chunk.page_content}"

                if is_toc_chunk(final_input) or len(final_input) < 200:
                    continue

                jsonl_row = {
                    "id": saved_count + 1,  # ID nadajemy sekwencyjnie tylko dla zapisanych
                    "instruction": INSTRUCTION_TEMPLATE,
                    "input": final_input,
                    "output": ""
                }

                out_f.write(json.dumps(jsonl_row, ensure_ascii=False) + "\n")
                saved_count += 1

    print(f"\n--- PODSUMOWANIE PROCESU ---")
    print(f"Wszystkich wygenerowanych chunków: {len(final_chunks)}")
    print(f"Odrzuconych (poniżej {min_characters} znaków): {discarded_count}")
    print(f"Zapisanych pomyślnie do JSONL: {saved_count}")

# --- URUCHOMIENIE ---
if __name__ == "__main__":
    sciezka_pdf = "/app/data_new"

    # max_tokens=6000 (górny limit), min_characters=800 (dolny limit wielkości)
    process_pdf_to_jsonl_fast_with_filter(
        pdf_dir=sciezka_pdf,
        max_tokens=6000,
        min_characters=800,
        skip_existing=True
    )
