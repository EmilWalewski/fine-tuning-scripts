from flask import Flask, request, jsonify
import os
import tempfile
from llama_index.llms.ollama import Ollama

# Pełna logika ekstrakcji (Tier-1 + Tier-2 rescue + chunking + filtr garbled)
# jest JEDNYM źródłem w extract6.py — importujemy ją, żeby gwarantować parytet
# trening↔inferencja (zamiast duplikować ~250 linii, które mogłyby się rozjechać).
from extract6 import extract_inputs_from_pdf, INSTRUCTION_TEMPLATE

app = Flask(__name__)

OLLAMA_LLM_MODEL = "atlas"  # Nazwa Twojego modelu w Ollamie

llm = Ollama(
    model=OLLAMA_LLM_MODEL,
    request_timeout=1800.0,
    additional_kwargs={
        "num_ctx": 5120,
        "temperature": 0,
        "num_predict": 1000,
    },
)


@app.route('/analyze_report', methods=['POST'])
def analyze_report():
    if 'file' not in request.files:
        return jsonify({"error": "No file shared"}), 400

    file = request.files['file']

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, file.filename)
            file.save(pdf_path)

            # ── Ekstrakcja IDENTYCZNA jak przy budowie datasetu (wspólny adapter) ──
            # max_tokens / min_characters / max_garbled_risk = wartości domyślne
            # extract6.extract_inputs_from_pdf (3700 / 800 / 0.05).
            inputs_to_process = extract_inputs_from_pdf(pdf_path)

            # ── Inferencja LLM (dokładny format struktury treningowej) ──
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
                "chunks_count": len(inputs_to_process),
            })

    except Exception as e:
        print(f"Error encountered: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
