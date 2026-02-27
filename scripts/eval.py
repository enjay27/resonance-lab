import os
import sys
from unsloth import FastLanguageModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLEAN_MODEL_DIR, INSTRUCTION

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
    "ムクボ3凸完了",
    "15ch 銀ナッポ 転送・カナミヤ族集落の郊外の崖下あたり",
    "バグったので再起しましたー"
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