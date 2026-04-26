import os
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./tinyllama-medflashcards-lora"
SEED = 42
N_SAMPLES = 10_000   # set to None to use the full dataset

# ---- 0. Distributed init (fixes the NCCL barrier warning) ----
# torchrun sets LOCAL_RANK / RANK / WORLD_SIZE automatically.
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

if world_size > 1:
    # Pin this process to a single GPU BEFORE any CUDA / NCCL work.
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),  # <-- key line
        )

is_main_process = (local_rank == 0)

def log(msg):
    if is_main_process:
        print(msg)

# ---- 1. Load tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---- 2. Load and format dataset ----
raw = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")

# Optionally subsample. If N_SAMPLES is None or >= dataset size, use everything.
if N_SAMPLES is not None and N_SAMPLES < len(raw):
    raw = raw.shuffle(seed=SEED).select(range(N_SAMPLES))
    log(f"Using {len(raw)} random samples for training (seed={SEED}).")
else:
    log(f"Using the full dataset: {len(raw)} samples.")

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
log(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

# ---- 3. Load base model ----
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,         # H100s love bf16
    attn_implementation="sdpa",   # PyTorch's built-in scaled-dot-product attention
)
model.config.use_cache = False    # needed for training

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
if is_main_process:
    model.print_trainable_parameters()

# ---- 5. Training config ----
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=16,      # e.g. 4 GPUs × 16 = 64 effective batch
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    bf16=True,
    max_seq_length=1024,
    packing=True,                         # packs short samples → huge speedup
    dataset_text_field="text",
    gradient_checkpointing=False,         # not needed on H100 for 1.1B
    report_to="none",                     # change to "wandb" if you want logging
    ddp_find_unused_parameters=False,
    seed=SEED,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

trainer.train()

# ---- 6. Save final adapter (rank 0 only) ----
if is_main_process:
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done. Adapter saved to {OUTPUT_DIR}")

# ---- 7. Clean shutdown ----
if dist.is_initialized():
    dist.destroy_process_group()