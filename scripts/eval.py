import os
import sys
import json
import random
import torch
import sacrebleu
from unsloth import FastLanguageModel
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MASTER_MODEL_DIR, INSTRUCTION, LORA_DATASET_DIR

# 1. Load the model
print("Loading model for evaluation...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MASTER_MODEL_DIR,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model)

# 2. Load Validation Data
val_file = os.path.join(LORA_DATASET_DIR, "val.jsonl")
val_data = []
with open(val_file, 'r', encoding='utf-8') as f:
    for line in f:
        val_data.append(json.loads(line))

# Pick a random sample to evaluate (e.g., 50 lines).
# Evaluating the entire val set auto-regressively can take a long time.
EVAL_SAMPLE_SIZE = 50
if len(val_data) > EVAL_SAMPLE_SIZE:
    random.seed(42)
    val_data = random.sample(val_data, EVAL_SAMPLE_SIZE)

predictions = []
references = []

print(f"\n--- Running Translation on {len(val_data)} Validation Samples ---")
for data in tqdm(val_data, desc="Translating"):
    jp_text = data["input"]
    true_ko = data["output"]

    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": jp_text}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    ko_translation = result.split("assistant\n")[-1].strip()

    predictions.append(ko_translation)
    # sacrebleu expects a list of references for each prediction (in case there are multiple valid translations)
    references.append([true_ko])

    prompt = tokenizer.apply_chat_template([
        {"role": "user", "content": f"{INSTRUCTION}\n\n{jp_text}"},
    ], tokenize=False, add_generation_prompt=True)  # Ensure add_generation_prompt=True for inference

# 3. Calculate Metrics
print("\n--- Translation Metrics ---")

# Calculate BLEU (using tokenize='ko-mecab' if installed, otherwise default)
bleu = sacrebleu.corpus_bleu(predictions, references)
print(f"BLEU Score: {bleu.score:.2f} (Scale: 0-100, Higher is better)")

# Calculate chrF (Better for Korean due to agglutinative grammar)
chrf = sacrebleu.corpus_chrf(predictions, references)
print(f"chrF Score: {chrf.score:.2f} (Scale: 0-100, Higher is better)")

# Print a few examples to manually verify
print("\n--- Sample Outputs ---")
for i in range(min(3, len(predictions))):
    print(f"JP (Input) : {val_data[i]['input']}")
    print(f"KO (Actual): {val_data[i]['output']}")
    print(f"KO (Model) : {predictions[i]}")
    print("-" * 30)