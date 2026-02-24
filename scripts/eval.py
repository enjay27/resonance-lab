from unsloth import FastLanguageModel
import torch

# 1. Load the model you just trained
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "BlueProtocol_JP_KO_Translator", # Or your output name
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable 2x faster inference

# 2. Test Cases (Try some tricky Star Resonance slang)
test_queries = [
    "遺跡 １F～ T1 D2 28000↑募集中～",
    "スタレゾはよS2ならんか？服欲しい",
    "3竜EHN＠DたくさんH3T2 27k↑ギミック理解者のみ"
]

print("\n--- Translation Test ---")
for jp_text in test_queries:
    inputs = tokenizer(
        [
            f"<|im_start|>system\nTranslate the following Blue Protocol chat log from Japanese into neutral Korean.<|im_end|>\n<|im_start|>user\n{jp_text}<|im_end|>\n<|im_start|>assistant\n"
        ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64)
    result = tokenizer.batch_decode(outputs)[0]

    # Extract only the assistant's response
    ko_translation = result.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
    print(f"JP: {jp_text}")
    print(f"KO: {ko_translation}\n")