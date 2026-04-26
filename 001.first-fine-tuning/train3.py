import os
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from accelerate import PartialState

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./tinyllama-medflashcards-lora"
SEED = 42
N_SAMPLES = 10_000   # set to None to use the full dataset (~33k rows)

# Chat template wrapping the assistant turn in {% generation %} markers.
# TRL's assistant_only_loss reads these markers to build the loss mask so the
# prompt (system + user) does not contribute to the loss. TinyLlama's stock
# template lacks them, which is why we override here.
#
# IMPORTANT: for the {% generation %} markers to actually do anything, the
# dataset must reach SFTTrainer in *conversational* form (a "messages" column).
# If you pre-render with apply_chat_template(..., tokenize=False) the markers
# are consumed by Jinja and disappear, silently disabling the loss mask.
TINYLLAMA_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|system|>\n{{ message['content'] }}</s>\n"
    "{% elif message['role'] == 'user' %}"
    "<|user|>\n{{ message['content'] }}</s>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|assistant|>\n"
    "{% generation %}{{ message['content'] }}</s>{% endgeneration %}\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)

# TF32 for non-bf16 matmuls (norms, optimizer) — free speedup on H100.
# cuDNN autotuner picks the fastest kernel for each shape; cheap win because
# packing locks every batch to (per_device_batch, 1024).
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
set_seed(SEED)

# ---- 0. Distributed setup ----
# torchrun sets these env vars; defaults to single-process when launched plainly.
# We pin this rank to one GPU before any CUDA call, but leave NCCL init to
# Accelerate (Trainer constructs an Accelerator, which calls init_process_group
# with the right device_id internally).
local_rank  = int(os.environ.get("LOCAL_RANK", 0))
global_rank = int(os.environ.get("RANK", 0))
world_size  = int(os.environ.get("WORLD_SIZE", 1))

if world_size > 1:
    torch.cuda.set_device(local_rank)

is_main_process = (global_rank == 0)

def log(msg):
    if is_main_process:
        print(msg)

# ---- 1. Tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.chat_template = TINYLLAMA_CHAT_TEMPLATE
# Be explicit; LLaMA tokenizers are right-padded by default but some HF
# versions flip this for chat models, which would break SFT loss alignment.
tokenizer.padding_side = "right"

# LLaMA tokenizers ship without a pad token. Reusing EOS is the standard
# trick and is safe for *training* here because:
#   - packing=True produces no padding during training, and
#   - assistant_only_loss masks everything outside the assistant turn anyway.
# If you later call .generate() on the saved adapter, consider adding a
# dedicated pad token instead so the model can emit EOS unambiguously.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---- 2. Dataset ----
# Rank-0 builds the dataset cache; other ranks wait, then reuse it.
# We wrap *both* the load and the map so non-zero ranks never race rank 0
# for the Arrow cache files.
with PartialState().main_process_first():
    raw = load_dataset(
        "medalpaca/medical_meadow_medical_flashcards", split="train"
    )

    if N_SAMPLES is not None and N_SAMPLES < len(raw):
        raw = raw.shuffle(seed=SEED).select(range(N_SAMPLES))
        log(f"Using {len(raw)} random samples for training (seed={SEED}).")
    else:
        log(f"Using the full dataset: {len(raw)} samples.")

    def format_example(ex):
        # Emit conversational-format rows. SFTTrainer will detect the
        # "messages" column and call apply_chat_template(..., tokenize=True,
        # return_assistant_tokens_mask=True) itself — that's the only path
        # in which the {% generation %} markers actually drive the loss mask.
        # Pre-rendering to a "text" column here would silently disable
        # assistant_only_loss.
        return {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful medical assistant. Answer the "
                        "user's medical question accurately and concisely."
                    ),
                },
                {
                    "role": "user",
                    "content": ex["instruction"]
                    + (("\n" + ex["input"]) if ex.get("input") else ""),
                },
                {"role": "assistant", "content": ex["output"]},
            ]
        }

    dataset = raw.map(
        format_example,
        remove_columns=raw.column_names,
        num_proc=4,
        desc="Formatting chat examples",
    )

    # Tiny held-out split just to monitor overfitting; not used for HP search.
    split = dataset.train_test_split(test_size=0.02, seed=SEED)
    train_ds, eval_ds = split["train"], split["test"]

log(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

# ---- 3. Base model ----
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,         # bf16 is native on H100, no loss scaling needed.
    attn_implementation="flash_attention_3",   # PyTorch's fused attention, FlashAttention-2 backend on H100.
)
# KV-cache is for inference; disabling saves memory and silences a training warning.
model.config.use_cache = False
# Keep model and tokenizer in agreement on pad id, otherwise Trainer warns and
# generation/eval can mis-mask padded positions.
model.config.pad_token_id = tokenizer.pad_token_id

# ---- 4. LoRA ----
# Train low-rank adapters (~0.5% of params) instead of all 1.1B weights:
# tiny artifact, much less GPU memory, quality close to full fine-tuning.
lora_config = LoraConfig(
    r=16,                # adapter rank; higher = more capacity, more memory.
    lora_alpha=32,       # scaling; rule of thumb is alpha = 2 * r.
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # All linear layers in attention + MLP — the standard "all-linear" recipe.
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

    # 4× H100 80GB, 1.1B + LoRA at seq_len=1024: batch 32/GPU is comfortable.
    # Effective batch = 32 * 4 GPUs = 128 sequences per optimizer step.
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,        # Eval has no gradients, so ~2x is fine.
    gradient_accumulation_steps=1,

    # 2e-4 is well-tested for LoRA; cosine decay + short warmup is a safe default.
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    logging_steps=20,
    eval_strategy="steps",
    # Denser eval cadence: with ~9800 train samples and effective batch 128,
    # an epoch is ~76 optimizer steps, so 25 gives ~3 evals/epoch.
    eval_steps=25,
    save_strategy="steps",
    save_steps=25,                        # Must be a multiple of eval_steps.
    save_total_limit=2,                   # Keep only the 2 most recent checkpoints.

    bf16=True,
    bf16_full_eval=True,

    max_length=1024,
    # Train-time packing concatenates short samples up to 1024 tokens, killing
    # padding waste — big throughput win on this mostly-short dataset.
    # Eval is left unpacked so eval_loss has a stable per-example denominator,
    # which load_best_model_at_end relies on to pick a meaningful checkpoint.
    packing=True,
    eval_packing=False,
    # Loss only on assistant tokens (driven by the {% generation %} markers in
    # the chat template). Critical for instruction-tuning quality.
    # Requires the dataset to be in conversational ("messages") form — see above.
    assistant_only_loss=True,

    gradient_checkpointing=False,         # 1.1B on H100 — not needed.
    report_to="tensorboard",
    logging_dir=f"{OUTPUT_DIR}/logs",
    # All LoRA targets are hit every forward pass, so DDP can skip the slow
    # unused-parameter scan.
    ddp_find_unused_parameters=False,

    # Keep GPUs fed: parallel CPU loading, pinned memory for fast H2D copies,
    # persistent workers so we don't re-spawn each epoch.
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    seed=SEED,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,           # `tokenizer=` was renamed in transformers 5.x.
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

# ---- 6. Train ----
# get_last_checkpoint returns None for a non-existent or empty output dir,
# so the same call works for fresh runs and for resuming after a crash —
# without each rank doing its own racy os.listdir.
last_checkpoint = (
    get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
)

# try/finally guarantees NCCL teardown on OOM/crash, preventing zombie processes.
try:
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Both calls are already rank-0-only internally — no need to gate them.
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log(f"Done. Adapter saved to {OUTPUT_DIR}")
finally:
    if dist.is_initialized():
        # Wait for rank 0 to finish writing the final adapter/tokenizer files
        # before any rank tears down NCCL — otherwise non-zero ranks can exit
        # while rank 0 is still flushing to disk.
        dist.barrier()
        dist.destroy_process_group()