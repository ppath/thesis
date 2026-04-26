
[TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) — a 1.1B chat model, small enough to make the exercise fast, large enough to show meaningful differences.

[medalpaca/medical_meadow_medical_flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards) — ~33k medical Q&A flashcards. TinyLlama's medical knowledge is shallow, so after fine-tuning you'll see a clear difference when asking medical questions.

Fine-tuning is stronger at teaching a model how to respond in a domain (style, terminology, format) than at reliably injecting new factual knowledge. For truly adding retrievable facts, RAG (Retrieval-Augmented Generation) is more appropriate. That said, with a small model like TinyLlama, domain-specific fine-tuning produces a very visible improvement — which is exactly what you want for a first experience.






## 1. Environment setup

Create a fresh virtual environment and install the needed libraries:
```
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.2 datasets==2.21.0 accelerate==0.34.2 \
            peft==0.12.0 trl==0.10.1 bitsandbytes==0.43.3 \
            sentencepiece wandb \
            rich tyro
```

Then configure accelerate (accept defaults, say "yes" to multi-GPU, pick 4 processes, no DeepSpeed for now):
```
accelerate config
```

```
In which compute environment are you running? **This machine**
Which type of machine are you using? **multi-GPU**
How many different machines will you use (use more than 1 for multi-node training)? [1]:
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]:
Do you wish to optimize your script with torch dynamo?[yes/NO]:
Do you want to use DeepSpeed? [yes/NO]:
Do you want to use FullyShardedDataParallel? [yes/NO]:
Do you want to use Megatron-LM ? [yes/NO]:
How many GPU(s) should be used for distributed training? [1]: **4**
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:
Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]:
Do you wish to use mixed precision? **bf16**

accelerate configuration saved at /home/<user>/.cache/huggingface/accelerate/default_config.yaml
```


## Execution times

time accelerate launch train_1.py | tee out/train_1.txt
real	2m54.988s
user	6m49.318s
sys	4m24.244s

time accelerate launch train_2.py | tee out/train_2.txt
real	1m4.711s
user	2m27.461s
sys	1m25.825s


