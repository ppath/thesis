import os
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from accelerate import PartialState

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./tinyllama-medflashcards-lora"
SEED = 42
N_SAMPLES = 10_000

# ---- H100 perf knobs ----
torch.set_float32_matmul_precision("high")          # NEW
set_seed(SEED)                                       # NEW (full reproducibility)

# ---- 0. Distributed init ----
local_rank  = int(os.environ.get("LOCAL_RANK", 0))
global_rank = int(os.environ.get("RANK", 0))         # NEW
world_size  = int(os.environ.get("WORLD_SIZE", 1))

if world_size > 1:
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )

is_main_process = (global_rank == 0)                 # CHANGED: global, not local

def log(msg):
    if is_main_process:
        print(msg)

# ---- 1. Tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "right"                     # NEW: critical for causal LM training
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---- 2. Dataset ----
raw = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")

if N_SAMPLES is not None and N_SAMPLES < len(raw):
    raw = raw.shuffle(seed=SEED).select(range(N_SAMPLES))
    log(f"Using {len(raw)} random samples for training (seed={SEED}).")
else:
    log(f"Using the full dataset: {len(raw)} samples.")

def format_example(ex):
    messages = [
        {"role": "system", "content": "You are a helpful medical assistant. Answer the user's medical question accurately and concisely."},
        {"role": "user",   "content": ex["instruction"] + (("\n" + ex["input"]) if ex.get("input") else "")},
        {"role": "assistant", "content": ex["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

# CHANGED: only rank-0 builds the cache; others reuse it. Parallel CPU map.
with PartialState().main_process_first():
    dataset = raw.map(
        format_example,
        remove_columns=raw.column_names,
        num_proc=8,
        desc="Formatting chat examples",
    )

split = dataset.train_test_split(test_size=0.02, seed=SEED)
train_ds, eval_ds = split["train"], split["test"]
log(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

# ---- 3. Base model ----
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",   # swap to "flash_attention_3" if you `pip install flash-attn-3`
)
model.config.use_cache = False

# ---- 4. LoRA ----
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

    # bumped for 4× H100 80 GB on a 1.1B model with LoRA
    per_device_train_batch_size=32,        # CHANGED
    per_device_eval_batch_size=64,         # CHANGED (no grads in eval)
    gradient_accumulation_steps=1,

    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    # eval/save: less frequent so we don't spend the run on I/O
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=50,                         # CHANGED
    save_strategy="steps",
    save_steps=50,                         # CHANGED
    save_total_limit=2,

    bf16=True,
    bf16_full_eval=True,                   # NEW

    max_seq_length=1024,
    packing=True,

    gradient_checkpointing=False,
    report_to="tensorboard",               # CHANGED
    logging_dir=f"{OUTPUT_DIR}/logs",      # NEW
    ddp_find_unused_parameters=False,

    # data loader
    dataloader_num_workers=4,              # NEW
    dataloader_pin_memory=True,            # NEW
    dataloader_persistent_workers=True,    # NEW

    # checkpoint selection
    load_best_model_at_end=True,           # NEW
    metric_for_best_model="eval_loss",     # NEW
    greater_is_better=False,                # NEW

    # SFT-specific: only train on the assistant's tokens
    assistant_only_loss=True,              # NEW (KEY FIX)

    seed=SEED,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

# ---- 6. Train (with resume + safe shutdown) ----
resume = (
    os.path.isdir(OUTPUT_DIR)
    and any(d.startswith("checkpoint-") for d in os.listdir(OUTPUT_DIR))
)

try:
    trainer.train(resume_from_checkpoint=resume)

    if is_main_process:
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Done. Adapter saved to {OUTPUT_DIR}")
finally:
    if dist.is_initialized():
        dist.destroy_process_group()