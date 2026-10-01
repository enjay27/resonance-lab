"""Evaluate the merged model on the eval dataset: `python scripts/llamafactory/eval.py [--prompt ...]`.

Generation lives here (it needs the model and a GPU); the scoring and the report are the shared
eval_metrics module, the same one the unsloth pipeline uses. The report is printed and saved to
outputs/eval/<profile>-<prompt>.txt so runs of different models can be compared.
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import EVAL_DATASET_PATH, EVAL_OUTPUT_DIR, LF_PROFILE
from eval_metrics import comet_score, evaluate, format_report, load_eval_dataset, strip_think
from lf_tools import chat_messages, generate_inputs, load_profile, training_prompt, with_bos

PROMPTS = {
    "chat-template": "the model's own documented prompt through its chat template (TranslateGemma: language codes -> long "
    "English prompt, what resonance-stream sends today; Hy-MT2: its English 'Translate the following text into Korean')",
    "training": "the raw line in the training turn format -- exactly what fine-tuning saw",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the merged model on the eval dataset.")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="chat-template",
                        help="which prompt to evaluate with (default: %(default)s)")
    args = parser.parse_args(argv)

    profile = load_profile(LF_PROFILE)
    try:
        samples = load_eval_dataset(EVAL_DATASET_PATH)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    if not os.path.isdir(profile.merged_dir):
        print(f"[ERROR] No merged model at {profile.merged_dir}. Run the Merge LoRA stage first.")
        sys.exit(1)

    import torch  # heavy imports after the cheap checks
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Profile {profile.name} | prompt: {args.prompt} ({PROMPTS[args.prompt]})")
    print(f"Loading {profile.merged_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(profile.merged_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        profile.merged_dir, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    ).eval()

    def encode(jp_text):
        if args.prompt == "training":
            ids = with_bos(tokenizer(training_prompt(profile.template, jp_text))["input_ids"], tokenizer.bos_token_id)
            return {"input_ids": torch.tensor([ids]).to("cuda"), "attention_mask": torch.ones(1, len(ids), dtype=torch.long).to("cuda")}
        return generate_inputs(tokenizer.apply_chat_template(
            chat_messages(profile.template, jp_text),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to("cuda"))

    predictions, raw_outputs = [], []
    print(f"\n--- Running Translation on {len(samples)} Samples ---")
    for sample in tqdm(samples):
        inputs = encode(sample["original"])
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        raw_outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=False))
        predictions.append(strip_think(tokenizer.decode(new_tokens, skip_special_tokens=True)))

    report = format_report(evaluate(samples, predictions, raw_outputs), samples, predictions, comet=comet_score(samples, predictions))
    print("\n" + report)

    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    out = os.path.join(EVAL_OUTPUT_DIR, f"{profile.name}-{args.prompt}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    main()
