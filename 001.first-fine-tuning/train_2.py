import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./tinyllama-medflashcards-lora--train-2"
SEED = 42
N_SAMPLES = 10_000

# ---- 1. Load tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---- 2. Load and format dataset ----
raw = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")

# Randomly subsample up to N_SAMPLES examples (defensive: won't crash if dataset is smaller)
n = min(N_SAMPLES, len(raw))
raw = raw.shuffle(seed=SEED).select(range(n))
print(f"Using {len(raw)} random samples for training (seed={SEED}).")

# TinyLlama-Chat uses the ChatML-ish template with <|system|>, <|user|>, <|assistant|>
def format_example(ex):
    messages = [
        {"role": "system", "content": "You are a helpful medical assistant. Answer the user's medical question accurately and concisely."},
        {"role": "user", "content": ex["instruction"] + (("\n" + ex["input"]) if ex.get("input") else "")},
        {"role": "assistant", "content": ex["output"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

dataset = raw.map(format_example, remove_columns=raw.column_names)

# Optional: small eval split (2% of the samples)
split = dataset.train_test_split(test_size=0.02, seed=SEED)
train_ds, eval_ds = split["train"], split["test"]
print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

# ---- 3. Load base model ----
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
model.config.use_cache = False

# ---- 4. Apply LoRA ----
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---- 5. Training config ----
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    bf16=True,
    max_seq_length=1024,
    packing=True,
    dataset_text_field="text",
    gradient_checkpointing=False,
    report_to="none",
    ddp_find_unused_parameters=False,
    seed=SEED,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

trainer.train()

# ---- 6. Save final adapter ----
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done. Adapter saved to {OUTPUT_DIR}")

