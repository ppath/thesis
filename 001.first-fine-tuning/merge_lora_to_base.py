from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16
)
merged = PeftModel.from_pretrained(base, "./tinyllama-medflashcards-lora").merge_and_unload()
merged.save_pretrained("./tinyllama-medflashcards-merged")
AutoTokenizer.from_pretrained("./tinyllama-medflashcards-lora").save_pretrained("./tinyllama-medflashcards-merged")

