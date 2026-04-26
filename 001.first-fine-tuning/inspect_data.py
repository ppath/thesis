from datasets import load_dataset

ds = load_dataset("medalpaca/medical_meadow_medical_flashcards")
print(ds)
print(ds["train"][0])

