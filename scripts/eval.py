import os
import sys
import json
import random
import torch
import sacrebleu
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR, INSTRUCTION, LORA_DATASET_DIR

# =============================================
# 1. Load Model
# =============================================
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True
)
model.eval()

# =============================================
# 2. Load Validation Data
# =============================================
val_file = os.path.join(LORA_DATASET_DIR, "val.jsonl")
val_data = []
with open(val_file, 'r', encoding='utf-8') as f:
    for line in f:
        val_data.append(json.loads(line))

EVAL_SAMPLE_SIZE = 50
if len(val_data) > EVAL_SAMPLE_SIZE:
    random.seed(42)
    val_data = random.sample(val_data, EVAL_SAMPLE_SIZE)

# =============================================
# 3. Localization Term Dictionary
# =============================================
TERM_DICT = {
    '消化': '숙제',
    '完凸': '풀돌',
    '火力': '딜러',
    'ファスト': '속공',
    '器用': '숙련',
    'リキャスト': '쿨타임',
    '盾': '탱커',
    '杖': '법사',
    '弓': '궁수',
    'ウルト': '궁',
    'イマジン': '이매진',
    'ガシャ': '뽑기',
    'ばんわ': '존밤',
    'ヒグマ': '산적 두목',
    'ムークボス': '무크 두목',
}

ALPHABET_TERMS = ['discord', 'Discord', 'NM', 'EH', 'T', 'H', 'D', 'DPS']

# =============================================
# 4. Run Inference
# =============================================
predictions = []
references = []
raw_outputs = []

print(f"\n--- Running Translation on {len(val_data)} Samples ---")
for data in tqdm(val_data):
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
        return_tensors="pt",
        enable_thinking=False
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    new_tokens = outputs[0][inputs.shape[-1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
    clean = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Strip empty think tags
    clean = re.sub(r'<think>\s*</think>\s*', '', clean).strip()

    predictions.append(clean)
    references.append([true_ko])
    raw_outputs.append(raw)

# =============================================
# 5. Standard Metrics (BLEU, chrF, TER)
# =============================================
print("\n--- Standard Metrics ---")

bleu = sacrebleu.corpus_bleu(predictions, references)
print(f"BLEU  : {bleu.score:.2f}  (higher is better, 0-100)")

chrf = sacrebleu.corpus_chrf(predictions, references)
print(f"chrF  : {chrf.score:.2f}  (higher is better, 0-100)")

ter = sacrebleu.corpus_ter(predictions, references)
print(f"TER   : {ter.score:.2f}  (lower is better, 0-100)")

# =============================================
# 6. COMET (Neural Metric — best for Korean)
# =============================================
print("\n--- COMET Score ---")
try:
    from comet import download_model, load_from_checkpoint
    comet_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_path)

    comet_data = [
        {"src": val_data[i]["input"], "mt": predictions[i], "ref": val_data[i]["output"]}
        for i in range(len(predictions))
    ]
    comet_scores = comet_model.predict(comet_data, batch_size=8, gpus=1)
    print(f"COMET : {comet_scores.system_score:.4f}  (higher is better, 0-1, >0.85 is good)")
except ImportError:
    print("COMET not installed. Run: pip install unbabel-comet")
except Exception as e:
    print(f"COMET failed: {e}")

# =============================================
# 7. Custom Metrics
# =============================================
print("\n--- Custom Metrics ---")

# 7-1. JP Leakage Rate
jp_pattern = re.compile(r'[\u3041-\u3096\u30A1-\u30FA\u4e00-\u9fff]')
jp_leakage = sum(1 for p in predictions if jp_pattern.search(p))
print(f"JP Leakage    : {jp_leakage}/{len(predictions)} ({jp_leakage/len(predictions)*100:.1f}%) — lower is better")

# 7-2. Think Tag Leakage
think_leakage = sum(1 for r in raw_outputs if '<think>' in r and len(r.split('<think>')[1].split('</think>')[0].strip()) > 0)
print(f"Think Leakage : {think_leakage}/{len(predictions)} ({think_leakage/len(predictions)*100:.1f}%) — lower is better")

# 7-3. Term Accuracy
term_hits = 0
term_total = 0
term_misses = []

for i, data in enumerate(val_data):
    jp = data["input"]
    pred = predictions[i]
    ref = data["output"]

    for jp_term, ko_term in TERM_DICT.items():
        if jp_term in jp:
            term_total += 1
            if ko_term in pred:
                term_hits += 1
            else:
                term_misses.append((jp, pred, ref, jp_term, ko_term))

if term_total > 0:
    print(f"Term Accuracy : {term_hits}/{term_total} ({term_hits/term_total*100:.1f}%) — higher is better")
    if term_misses:
        print("\n  Term Misses (first 5):")
        for jp, pred, ref, jp_t, ko_t in term_misses[:5]:
            print(f"    JP : {jp}")
            print(f"    Expected '{ko_t}' for '{jp_t}' but got: {pred}")
            print()
else:
    print("Term Accuracy : No term-containing samples in eval set")

# 7-4. Alphabet Preservation (discord → discord, not 디스코드)
alpha_pattern = re.compile(r'[A-Za-z]{3,}')  # 3+ char english words
alpha_violations = 0
for i, data in enumerate(val_data):
    jp = data["input"]
    # Find english words in original
    eng_words = alpha_pattern.findall(jp)
    for word in eng_words:
        # Check if it was transliterated (simplified heuristic)
        if word.lower() == 'discord' and '디스코드' in predictions[i]:
            alpha_violations += 1

print(f"Alpha Preserved: {alpha_violations} discord violations found")

# 7-5. Exact Match Rate
exact = sum(1 for p, r in zip(predictions, references) if p == r[0])
print(f"Exact Match   : {exact}/{len(predictions)} ({exact/len(predictions)*100:.1f}%)")

# =============================================
# 8. Sample Outputs
# =============================================
print("\n--- Sample Outputs (first 5) ---")
for i in range(min(5, len(predictions))):
    print(f"JP  : {val_data[i]['input']}")
    print(f"REF : {val_data[i]['output']}")
    print(f"PRED: {predictions[i]}")
    print("-" * 40)