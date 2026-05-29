import json
import random
import os
import glob

def prepare_dataset(input_dir, output_file):
    combined_data = []

    # ... (logika wczytywania i tasowania pozostaje taka sama jak poprzednio) ...
    search_path = os.path.join(input_dir, "*.jsonl")
    files = glob.glob(search_path)

    if not files:
        print(f"Błąd: Nie znaleziono plików w {input_dir}")
        return

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    combined_data.append({
                        "instruction": entry['instruction'],
                        "input": entry['input'],
                        "output": entry['output']
                    })
                except: continue

    random.seed(42)
    random.shuffle(combined_data)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in combined_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Gotowe! Zbiór zapisano w: {output_file}")

# --- TUTAJ WPISUJESZ SWOJE ŚCIEŻKI ---
if __name__ == "__main__":
    MOJ_FOLDER_Z_DANYMI = "./dataset-analyzed-reports"
    MOJ_PLIK_WYNIKOWY = "./dataset.jsonl"

    prepare_dataset(MOJ_FOLDER_Z_DANYMI, MOJ_PLIK_WYNIKOWY)
