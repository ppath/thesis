from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

print("Downloading model...")
AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

print("Downloading dataset...")
load_dataset("medalpaca/medical_meadow_medical_flashcards")

print("✅ Everything cached locally.")

