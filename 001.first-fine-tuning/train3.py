import os
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from accelerate import PartialState

# Small (1.1B) chat-tuned LLaMA — fast to fine-tune, good for learning/demos.
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./tinyllama-medflashcards-lora"
SEED = 42
N_SAMPLES = 10_000   # set to None to use the full dataset (~33k rows)

# TF32 for non-bf16 matmuls (norms, optimizer) — free speedup on H100.
torch.set_float32_matmul_precision("high")
# Seeds Python/NumPy/Torch RNGs so dataset shuffling and init are reproducible.
set_seed(SEED)

# ---- 0. Distributed init ----
# torchrun sets these env vars; defaults to single-process if launched plainly.
local_rank  = int(os.environ.get("LOCAL_RANK", 0))
global_rank = int(os.environ.get("RANK", 0))
world_size  = int(os.environ.get("WORLD_SIZE", 1))

if world_size > 1:
    # Pin this process to one GPU BEFORE any CUDA/NCCL call to avoid contention.
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        # `device_id=` lets NCCL skip a slow barrier on init (PyTorch 2.3+).
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )

# Use global rank so logging/saving happens on exactly one process, even multi-node.
is_main_process = (global_rank == 0)

def log(msg):
    if is_main_process:
        print(msg)

# ---- 1. Tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# Causal LMs must pad on the RIGHT during training — left padding shifts labels and breaks the loss.
tokenizer.padding_side = "right"
# LLaMA tokenizers ship without a pad token; reusing EOS is the standard trick.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---- 2. Dataset ----
# Q&A pairs from medical flashcards — a clean instruction-tuning dataset.
raw = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")

if N_SAMPLES is not None and N_SAMPLES < len(raw):
    raw = raw.shuffle(seed=SEED).select(range(N_SAMPLES))
    log(f"Using {len(raw)} random samples for training (seed={SEED}).")
else:
    log(f"Using the full dataset: {len(raw)} samples.")

def format_example(ex):
    # Wrap each row in TinyLlama's chat template so the model sees the same
    # <|system|>/<|user|>/<|assistant|> structure it was pretrained on.
    messages = [
        {"role": "system", "content": "You are a helpful medical assistant. Answer the user's medical question accurately and concisely."},
        {"role": "user",   "content": ex["instruction"] + (("\n" + ex["input"]) if ex.get("input") else "")},
        {"role": "assistant", "content": ex["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

# Only rank-0 builds the dataset cache; other ranks wait, then reuse it.
# `num_proc=8` parallelizes the CPU-bound formatting across cores.
with PartialState().main_process_first():
    dataset = raw.map(
        format_example,
        remove_columns=raw.column_names,
        num_proc=8,
        desc="Formatting chat examples",
    )

# Tiny held-out split just to monitor overfitting; not used for hyperparameter search.
split = dataset.train_test_split(test_size=0.02, seed=SEED)
train_ds, eval_ds = split["train"], split["test"]
log(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

# ---- 3. Base model ----
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,         # bf16 = ~half the memory of fp32, native on H100, no loss scaling needed.
    attn_implementation="sdpa",   # PyTorch's fused attention; uses FlashAttention-2 on H100 automatically.
)
# KV-cache is for inference only; disabling it saves memory and silences a warning during training.
model.config.use_cache = False

# ---- 4. LoRA ----
# Instead of updating all 1.1B weights, we train tiny low-rank adapters (~0.5% of params).
# Result: a few MB to save, much less GPU memory, and quality close to full fine-tuning.
lora_config = LoraConfig(
    r=16,                # rank of the adapter — higher = more capacity, more memory.
    lora_alpha=32,       # scaling factor; the common rule of thumb is alpha = 2 * r.
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Apply LoRA to every linear layer in attention + MLP — the standard "all-linear" recipe.
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

    # 4× H100 80GB, 1.1B model + LoRA at seq_len=1024 → batch 32/GPU is comfortable.
    # Effective batch = 32 * 4 GPUs = 128 sequences per optimizer step.
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,        # Eval has no gradients, so we can ~2x the batch.
    gradient_accumulation_steps=1,

    # 2e-4 is a well-tested LR for LoRA; cosine decay + short warmup is a safe default.
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    logging_steps=20,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,                   # Keep only the 2 most recent checkpoints to save disk.

    bf16=True,
    bf16_full_eval=True,                  # Eval in bf16 too — faster, and accurate enough for monitoring.

    max_seq_length=1024,
    # Packing concatenates short samples up to max_seq_length, killing padding waste.
    # Big throughput win for datasets with variable, mostly-short examples like this one.
    packing=True,
    # Mask the prompt (system + user) from the loss so the model only learns to produce answers,
    # not to imitate user questions. Critical for instruction tuning quality.
    assistant_only_loss=True,

    gradient_checkpointing=False,         # Trades compute for memory — not needed for a 1.1B model on H100.
    report_to="tensorboard",
    logging_dir=f"{OUTPUT_DIR}/logs",
    # All LoRA targets are used every forward pass, so DDP doesn't need the (slow) unused-param scan.
    ddp_find_unused_parameters=False,

    # Keep the GPUs fed: parallel CPU loading, pinned memory for fast H2D copies,
    # persistent workers so we don't pay re-spawn cost every epoch.
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,

    # After training, automatically reload the checkpoint with the lowest eval loss.
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    seed=SEED,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,           # `tokenizer=` was renamed to `processing_class=` in transformers 5.x.
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

# ---- 6. Train ----
# Auto-resume if a previous run left checkpoints behind — handy after a crash or preemption.
resume = (
    os.path.isdir(OUTPUT_DIR)
    and any(d.startswith("checkpoint-") for d in os.listdir(OUTPUT_DIR))
)

# try/finally guarantees we tear down NCCL even on OOM/crash, preventing zombie processes.
try:
    trainer.train(resume_from_checkpoint=resume)

    # Save only on rank 0 — otherwise 4 processes race to write the same files.
    if is_main_process:
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Done. Adapter saved to {OUTPUT_DIR}")
finally:
    if dist.is_initialized():
        dist.destroy_process_group()