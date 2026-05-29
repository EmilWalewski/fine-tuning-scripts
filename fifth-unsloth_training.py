import psutil
import builtins
builtins.psutil = psutil
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1102"
os.environ["UNSLOTH_USE_COMPILED_TRAINER"] = "0"
os.environ["PIP_BREAK_SYSTEM_PACKAGES"] = "1"

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from unsloth.chat_templates import train_on_responses_only

# 1. Ładowanie modelu przez Unsloth (Magia oszczędzania VRAM)
print('Loading model and tokenizer')
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/workspace/correct-rocm/huggingface/hub/models--NousResearch--Meta-Llama-3-8B/snapshots/315b20096dc791d381d514deb5f8bd9c8d6d3061/",
    max_seq_length = 8192, # Zacznijmy bezpiecznie od 512
    load_in_4bit = True,
    dtype = torch.bfloat16, # RX 7600S kocha float16
    device_map = {"": 0},
    # Mówimy systemowi dokładnie, ile ma miejsca na GPU
    #max_memory = {0: "7.5GiB", "cpu": "16GiB"},
)

# 2. Dodawanie LoRA (Unsloth robi to optymalnie)
print('Loading Lora')
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Możesz dać nawet 16, Unsloth to udźwignie
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 64,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # KLUCZOWE
    random_state = 3407,
)

# 3. Dataset (tak jak wcześniej)
dataset = load_dataset("json", data_files="./dataset2.jsonl", split="train")
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        #text = f"Instruction: {instruction}\nInput: {input}\nOutput: {output}{EOS_TOKEN}"
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Output:\n{output}{EOS_TOKEN}"
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

print("DEBUG - Pierwszy przykład z datasetu:")
print(dataset[0]["text"])

# 4. Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 8192,
    dataset_num_proc = 2,#8,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5, #5
        #Zamiast zgadywać liczbę kroków, lepiej powiedzieć trenerowi: "Przejdź przez moje dane dokładnie 3 razy". To jest bezpieczniejsze przy małych zbiorach danych.
        num_train_epochs = 12,
        #max_steps = 2,
        learning_rate = 2e-5,
        fp16 = False,        # MUSI BYĆ FALSE
        bf16 = True,
        optim = "paged_adamw_8bit", #optim = "adamw_8bit", # Dzięki Twojemu bitsandbytes to zadziała!
        weight_decay = 0.01,
        output_dir = "outputs",
        remove_unused_columns = True,
        lr_scheduler_type = "linear",  # Dodane dla stabilności
        seed = 3407,                   # Stały seed pomaga w powtarzalności wyników
        # --- OSZCZĘDZANIE VRAM ---
        gradient_checkpointing = True, # Kluczowe przy 8GB VRAM!
        # --- LOGOWANIE I ZAPIS ---
        logging_steps = 1,             # Chcemy widzieć loss co każdy krok
        save_strategy = "no",          # Przy tak krótkim treningu nie musimy robić checkpointów
        # --- KOMPATYBILNOŚĆ ---
        report_to = "none"
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "### Instruction:\n",  # Zmienione z None
    response_part = "### Output:\n", # ...aż do tego konkretnego znacznika
)
trainer.train()

print('--- Trening zakończony. Rozpoczynam eksport do GGUF ---')

# Zapisujemy adaptery (na wszelki wypadek)

#model.save_pretrained("model_ollama_repo")
#tokenizer.save_pretrained("model_ollama_repo")

# Eksport do formatu GGUF (mergowanie + kwantyzacja)
# Uwaga: Może to potrwać kilka minut i wymaga wolnego miejsca na dysku (ok. 5-6 GB)
# model.save_pretrained_gguf(
#     "model_ollama_repo",
#     tokenizer,
#     quantization_method = "q4_k_m"
# )

model.save_pretrained_merged(
    "model_ollama_repo",
    tokenizer,
    save_method = "merged_16bit" # Najwyższa jakość
)

#python llama.cpp/convert_hf_to_gguf.py model_ollama_repo --outfile model.gguf
#./llama.cpp/llama-quantize model.gguf model-q4_k_m.gguf Q4_K_M
print('--- GOTOWE! Model GGUF znajduje się w folderze model_ollama_repo ---')
