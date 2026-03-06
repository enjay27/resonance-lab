import os
import sys
import json
import torch
import sacrebleu
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR, INSTRUCTION, EVAL_DATASET_PATH

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
# 2. Load Eval Dataset
# =============================================
eval_data = []
with open(EVAL_DATASET_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        eval_data.append(json.loads(line))

print(f"Loaded {len(eval_data)} eval samples")

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

# =============================================
# 4. Run Inference
# =============================================
predictions = []
references = []
raw_outputs = []

print(f"\n--- Running Translation on {len(eval_data)} Samples ---")
for data in tqdm(eval_data):
    jp_text = data["original"]
    true_ko = data["translated"]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": "ja",
                    "target_lang_code": "ko",
                    "text": jp_text
                }
            ]
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,           # ← returns BatchEncoding
        return_tensors="pt",
        enable_thinking=False,
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,               # ← unpack dict
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[-1]
    new_tokens = outputs[0][input_length:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
    clean = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
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
        {"src": eval_data[i]["original"], "mt": predictions[i], "ref": eval_data[i]["translated"]}
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

JP_PATTERN = re.compile(r'[\u3041-\u3096\u30A1-\u30FA\u4e00-\u9fff]')

# 7-1. JP Leakage Rate
jp_leakage = sum(1 for p in predictions if JP_PATTERN.search(p))
print(f"JP Leakage    : {jp_leakage}/{len(predictions)} ({jp_leakage/len(predictions)*100:.1f}%) — lower is better")

# 7-2. Think Tag Leakage
think_leakage = sum(
    1 for r in raw_outputs
    if '<think>' in r and len(r.split('<think>')[1].split('</think>')[0].strip()) > 0
)
print(f"Think Leakage : {think_leakage}/{len(predictions)} ({think_leakage/len(predictions)*100:.1f}%) — lower is better")

# 7-3. Term Accuracy
term_hits = 0
term_total = 0
term_misses = []

for i, data in enumerate(eval_data):
    jp = data["original"]
    pred = predictions[i]
    ref = data["translated"]

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
            print(f"    JP  : {jp}")
            print(f"    REF : {ref}")
            print(f"    PRED: {pred}")
            print(f"    Expected '{ko_t}' for '{jp_t}'")
            print()
else:
    print("Term Accuracy : No term-containing samples in eval set")

# 7-4. Alphabet Preservation
alpha_pattern = re.compile(r'[A-Za-z]{3,}')
alpha_violations = 0
for i, data in enumerate(eval_data):
    jp = data["original"]
    eng_words = alpha_pattern.findall(jp)
    for word in eng_words:
        if word.lower() == 'discord' and '디스코드' in predictions[i]:
            alpha_violations += 1

print(f"Alpha Preserved: {alpha_violations} discord violations found")

# 7-5. Exact Match Rate
exact = sum(1 for p, r in zip(predictions, references) if p == r[0])
print(f"Exact Match   : {exact}/{len(predictions)} ({exact/len(predictions)*100:.1f}%)")

# =============================================
# 8. Category Breakdown
# =============================================
print("\n--- Category Breakdown ---")

categories = {}
for i, data in enumerate(eval_data):
    cat = data.get("category", "unknown")
    if cat not in categories:
        categories[cat] = {"total": 0, "jp_leak": 0, "term_miss": 0, "discord_viol": 0}

    pred = predictions[i]
    jp = data["original"]

    categories[cat]["total"] += 1

    if JP_PATTERN.search(pred):
        categories[cat]["jp_leak"] += 1

    for jp_term, ko_term in TERM_DICT.items():
        if jp_term in jp and ko_term not in pred:
            categories[cat]["term_miss"] += 1

    if 'discord' in jp.lower() and '디스코드' in pred:
        categories[cat]["discord_viol"] += 1

for cat, stats in categories.items():
    total = stats["total"]
    print(f"\n  [{cat}] ({total} samples)")
    print(f"    JP Leakage     : {stats['jp_leak']}/{total}")
    print(f"    Term Misses    : {stats['term_miss']}/{total}")
    print(f"    Discord Viol   : {stats['discord_viol']}/{total}")

# =============================================
# 9. Full Output Log
# =============================================
print("\n--- Full Output Log ---")
for i, data in enumerate(eval_data):
    cat = data.get("category", "?")
    jp = data["original"]
    ref = data["translated"]
    pred = predictions[i]

    has_jp = bool(JP_PATTERN.search(pred))
    has_discord_viol = 'discord' in jp.lower() and '디스코드' in pred

    flags = []
    if has_jp: flags.append("⚠️ JP")
    if has_discord_viol: flags.append("⚠️ discord")
    flag_str = "  " + " ".join(flags) if flags else "  ✅"

    print(f"[{cat}]")
    print(f"  JP  : {jp}")
    print(f"  REF : {ref}")
    print(f"  PRED: {pred}{flag_str}")
    print()