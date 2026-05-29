import json
import os
import tiktoken

def count_tokens(text):
    """Liczy tokeny przy użyciu kodowania cl100k_base (używanego przez GPT-4/Gemini)."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def process_all_files(input_dir, max_tokens=6000):
    # 1. Tworzenie folderu wyjściowego wewnątrz folderu wejściowego
    output_dir = os.path.join(input_dir, "merged_output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Rozpoczynam przetwarzanie. Wyniki znajdą się w: {output_dir}")

    # 2. Iteracja po plikach w folderze
    for filename in os.listdir(input_dir):
        if filename.endswith(".jsonl"):
            input_path = os.path.join(input_dir, filename)
            # Tworzenie nowej nazwy z dopiskiem -merged
            output_filename = filename.replace(".jsonl", "-merged.jsonl")
            output_path = os.path.join(output_dir, output_filename)

            print(f"Przetwarzam: {filename} -> {output_filename}")
            merge_single_file(input_path, output_path, max_tokens)

def merge_single_file(input_file, output_file, max_tokens):
    merged_entries = []
    current_input_parts = []
    current_token_count = 0
    current_id = 1
    system_instruction = None

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)

            if system_instruction is None:
                system_instruction = item.get("instruction", "")

            text_to_add = item["input"]
            tokens_in_text = count_tokens(text_to_add)

            # Sprawdzenie limitu
            if current_token_count + tokens_in_text > max_tokens and current_input_parts:
                # Zapisujemy to co mamy w buforze
                merged_entries.append({
                    "id": current_id,
                    "instruction": system_instruction,
                    "input": "\n\n".join(current_input_parts),
                    "output": ""
                })
                # Reset bufora
                current_id += 1
                current_input_parts = [text_to_add]
                current_token_count = tokens_in_text
            else:
                current_input_parts.append(text_to_add)
                current_token_count += tokens_in_text

        # Dodanie ostatniej paczki
        if current_input_parts:
            merged_entries.append({
                "id": current_id,
                "instruction": system_instruction,
                "input": "\n\n".join(current_input_parts),
                "output": ""
            })

    # Zapis do nowego pliku
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in merged_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # Tutaj podaj ścieżkę do folderu, w którym masz pliki .jsonl
    target_folder = '/app/prepare-dataset/dataset-to-process'
    process_all_files(target_folder)
