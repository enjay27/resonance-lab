import os
import sys
from unsloth import FastLanguageModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import CLEAN_MODEL_DIR, EVAL_DATASET_PATH, EVAL_OUTPUT_DIR, INSTRUCTION
from eval_metrics import comet_score, evaluate, format_report, load_eval_dataset, strip_think

# 1. Load the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CLEAN_MODEL_DIR,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# 2. Test Cases
test_queries = [
    "遺跡 １F～ T1 D2 28000↑募集中～",
    "スタレゾはよS2ならんか？服欲しい",
    "3竜EHN＠DたくさんH3T2 27k↑ギミック理解者のみ",
    "スカイ全然でる気配ないや",
    "ムクボ3돌 완료! 이제 90무기 제작하러 갑니다",
    "遺跡1Fから　29k↑　＠T1",
    "おやすみ！",
    "ムクボ3凸完了"
]

print("\n--- Translation Test ---")
for jp_text in test_queries:
    # Use the exact same format as your training script!
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": jp_text}
    ]

    # Let the tokenizer handle the <|im_start|> tags automatically
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(inputs, max_new_tokens=64)
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Extracting the final answer cleanly
    ko_translation = result.split("assistant\n")[-1].strip()
    print(f"JP: {jp_text}")
    print(f"KO: {ko_translation}\n")

# 3. Shared metrics on the eval dataset -- the same report the llamafactory pipeline prints,
#    so the two pipelines' numbers compare. Skipped (not an error) when there is no eval dataset.
if not os.path.exists(EVAL_DATASET_PATH):
    print(f"\nNo eval dataset at {EVAL_DATASET_PATH}; skipping the metrics.")
else:
    samples = load_eval_dataset(EVAL_DATASET_PATH)
    predictions, raw_outputs = [], []
    print(f"\n--- Running Translation on {len(samples)} Samples ---")
    for sample in samples:
        messages = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": sample["original"]}
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        outputs = model.generate(inputs, max_new_tokens=256, do_sample=False)
        new_tokens = outputs[0][inputs.shape[-1]:]
        raw_outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=False))
        predictions.append(strip_think(tokenizer.decode(new_tokens, skip_special_tokens=True)))

    report = format_report(evaluate(samples, predictions, raw_outputs), samples, predictions,
                           comet=comet_score(samples, predictions))
    print("\n" + report)
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    out = os.path.join(EVAL_OUTPUT_DIR, "unsloth-qwen3.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved: {out}")
