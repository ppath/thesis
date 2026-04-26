import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_DIR_1 = "tinyllama-medflashcards-lora--train-1"
LORA_DIR_2 = "tinyllama-medflashcards-lora--train-2"

TEST_QUESTIONS = [
    "What are the classic symptoms of diabetic ketoacidosis?",
    "Explain the mechanism of action of metformin.",
    "What is the difference between Crohn's disease and ulcerative colitis?",
    "What are the risk factors for pulmonary embolism?",
    "Describe the pathophysiology of myocardial infarction.",
]


def choose_model():
    print("\nChoose a model for inference:")
    print("  1) Base model (TinyLlama-1.1B-Chat-v1.0)")
    print("  2) Fine-tuned LoRA: tinyllama-medflashcards-lora--train-1")
    print("  3) Fine-tuned LoRA: tinyllama-medflashcards-lora--train-2")

    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice in {"1", "2", "3"}:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")


def load_model(choice):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"\nLoading base model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )

    if choice == "1":
        print("Using base model (no LoRA).")
        label = "base"
    else:
        lora_dir = LORA_DIR_1 if choice == "2" else LORA_DIR_2
        print(f"Attaching LoRA adapter from: {lora_dir}")
        model = PeftModel.from_pretrained(model, lora_dir)
        label = lora_dir

    model.eval()
    return model, tokenizer, device, label


def build_prompt(tokenizer, user_input):
    messages = [
        {"role": "system", "content": "You are a helpful medical study assistant. "
                                      "Answer concisely and accurately using flashcard-style responses."},
        {"role": "user", "content": user_input},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return (
            "<|system|>\nYou are a helpful medical study assistant.</s>\n"
            f"<|user|>\n{user_input}</s>\n<|assistant|>\n"
        )


@torch.no_grad()
def generate(model, tokenizer, device, user_input,
             max_new_tokens=256, temperature=0.7, top_p=0.9, repetition_penalty=1.1):
    prompt = build_prompt(tokenizer, user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    choice = choose_model()
    model, tokenizer, device, label = load_model(choice)
    print(f"\n>>> Ready. Using: {label}")
    print(f">>> Running {len(TEST_QUESTIONS)} test questions...\n")
    print("=" * 80)

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[Question {i}/{len(TEST_QUESTIONS)}]")
        print(f"Q: {question}\n")
        response = generate(model, tokenizer, device, question)
        print(f"A ({label}):\n{response}")
        print("-" * 80)

    print("\nDone.")


if __name__ == "__main__":
    main()

