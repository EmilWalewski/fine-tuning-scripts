import psutil
import builtins
builtins.psutil = psutil
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1102"
os.environ["UNSLOTH_USE_COMPILED_TRAINER"] = "0"
os.environ["PIP_BREAK_SYSTEM_PACKAGES"] = "1"
# --- Przeciwdziałanie fragmentacji VRAM (to sugeruje sam komunikat OOM) ---
# Na ROCm właściwa zmienna to PYTORCH_HIP_ALLOC_CONF; CUDA-owy wariant
# ustawiamy dla pewności, bo część buildów ROCm nadal go czyta.
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer  # (już nieużywane — zostawione na wszelki wypadek)
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset
from unsloth.chat_templates import train_on_responses_only

# 1. Ładowanie modelu przez Unsloth (Magia oszczędzania VRAM)
print('Loading model and tokenizer')
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/workspace/correct-rocm/huggingface/hub/models--NousResearch--Meta-Llama-3-8B/snapshots/315b20096dc791d381d514deb5f8bd9c8d6d3061/",
    max_seq_length = 5120, # TEST PAMIĘCI: szukamy największego okna mieszczącego się w 8GB (6144 OOM)
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
    lora_alpha = 32,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # KLUCZOWE
    random_state = 3407,
)

# 3. Dataset — ręczne tokenizowanie z maskowaniem promptu (deterministyczne).
# train_on_responses_only zawodziło: kolator maskował ~wszystkie przykłady na -100
# (30/31 eval) -> eval_loss=nan, a train uczył się na garstce, stąd fałszywie niski
# loss. Tu budujemy labels sami: maskujemy DOKŁADNIE tokeny promptu (-100), stratę
# liczymy tylko na odpowiedzi + EOS. Zero dopasowywania stringów.
dataset = load_dataset("json", data_files="./dataset.jsonl", split="train")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
MAX_LEN = 5120  # TEST PAMIĘCI: musi być równe max_seq_length z from_pretrained

def tokenize_and_mask(example):
    prompt = (f"### Instruction:\n{example['instruction']}\n\n"
              f"### Input:\n{example['input']}\n\n"
              f"### Output:\n")
    full = prompt + example["output"] + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens = True)["input_ids"]
    full_ids   = tokenizer(full,   add_special_tokens = True)["input_ids"]
    # liczba tokenów wspólnego prefiksu = długość promptu (odporne na merge na granicy)
    n = 0
    for a, b in zip(prompt_ids, full_ids):
        if a == b: n += 1
        else: break
    full_ids = full_ids[:MAX_LEN]
    labels = ([-100] * n + full_ids[n:])[:len(full_ids)]
    return {"input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels}

dataset = dataset.map(tokenize_and_mask, remove_columns = dataset.column_names)
dataset = dataset.train_test_split(test_size = 0.1, seed = 3407)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Potwierdzenie, że maskowanie działa: KAŻDY przykład musi mieć >0 tokenów straty
_u = [sum(1 for x in eval_dataset[i]["labels"] if x != -100) for i in range(len(eval_dataset))]
print(f"DEBUG maskowanie eval: min_odmaskowanych_tokenow={min(_u)} | "
      f"przyklady_z_0_strata={sum(1 for x in _u if x == 0)}/{len(_u)}  (musi być 0)")
print(f"DEBUG train[0]: dlugosc={len(train_dataset[0]['input_ids'])} tok, "
      f"odmaskowanych={sum(1 for x in train_dataset[0]['labels'] if x != -100)}")

# 4. Trainer (zwykły transformers.Trainer — dane już stokenizowane z labels)
data_collator = DataCollatorForSeq2Seq(tokenizer, padding = True, label_pad_token_id = -100)
# SFTTrainer (nie zwykły Trainer) — zachowuje pamięciooszczędną fused cross-entropy
# Unsloth, która NIE materializuje pełnych logitów [seq × 128k] (~2GB). Przy 8192 na
# 8GB VRAM to właśnie ta optymalizacja przesądza o tym, czy wejdzie czy OOM.
# skip_prepare_dataset: dane są już stokenizowane i zamaskowane ręcznie powyżej.
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    data_collator = data_collator,
    dataset_kwargs = {"skip_prepare_dataset": True},
    args = TrainingArguments(
        per_device_train_batch_size = 1,   # było 2 — przy 8GB VRAM jedna sekwencja na krok
        per_device_eval_batch_size = 1,    # ewaluacja po 1 — unika interakcji paddingu z maskowaniem
        gradient_accumulation_steps = 8,   # było 4 — efektywny batch zostaje 8 (1×8)
        warmup_steps = 5, #5
        #Zamiast zgadywać liczbę kroków, lepiej powiedzieć trenerowi: "Przejdź przez moje dane dokładnie 3 razy". To jest bezpieczniejsze przy małych zbiorach danych.
        num_train_epochs = 2,
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
        # --- WALIDACJA I CHECKPOINTY ---
        # Eval W TRAKCIE treningu WYŁĄCZONY: forward eval materializuje pełne logity
        # (~1.2GB dla 5120 tok), a trening i tak zjada ~całe 8GB -> OOM przy ewaluacji.
        # Trening (fused CE Unsloth) mieści się; ewaluację robimy OFFLINE na checkpointach.
        eval_strategy = "no",
        #eval_steps = 20,
        save_strategy = "steps",
        save_steps = 20,
        save_total_limit = 3,
        load_best_model_at_end = False,   # nie da się "best" bez eval w trakcie; wybierzemy offline
        #metric_for_best_model = "eval_loss",
        #greater_is_better = False,
        # --- KOMPATYBILNOŚĆ ---
        report_to = "none"
    ),
)

# (train_on_responses_only USUNIĘTE — maskowanie robimy ręcznie w tokenize_and_mask)

trainer.train()

print('--- Trening zakończony. Rozpoczynam eksport ---')
model.save_pretrained_merged(
    "model_ollama_repo",
    tokenizer,
    save_method = "merged_16bit" # Najwyższa jakość
)
#python llama.cpp/convert_hf_to_gguf.py model_ollama_repo --outfile model.gguf
#./llama.cpp/llama-quantize model.gguf model-q4_k_m.gguf Q4_K_M
print('--- GOTOWE! Model w folderze model_ollama_repo ---')
