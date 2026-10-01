"""Show the exact training example LLaMA-Factory builds from a row: `python scripts/llamafactory/inspect_pair.py [--model <profile>] [--rows 3]`.

Uses LLaMA-Factory's own template code (the one `llamafactory-cli train` uses) with the profile's
tokenizer and template, on the first rows of data/processed/lora_train_data.jsonl (`--format pair`).
For each row it prints the masked part (the prompt: not trained on), the trained part (the response, with
its end-of-turn token if the template adds one), and the token count against the profile's cutoff_len.
No GPU needed; needs llamafactory + transformers. Run it as a file, not `python -c`.
"""

import argparse
import os
import sys

import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import PROCESSED_LOGS
from lf_tools import add_model_argument, first_pairs, load_profile, model_name


def main(argv=None):
    parser = argparse.ArgumentParser(description="Show the training example built from the first rows of the pair file.")
    add_model_argument(parser)
    parser.add_argument("--rows", type=int, default=3, help="how many rows to show (default: %(default)s)")
    parser.add_argument("--file", default=PROCESSED_LOGS, help="pair file (default: the processed training file)")
    args = parser.parse_args(argv)

    profile = load_profile(model_name(args.model))
    try:
        pairs = first_pairs(args.file, args.rows)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    with open(profile.train_yaml, encoding="utf-8") as f:
        cutoff = yaml.safe_load(f).get("cutoff_len")

    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(profile.base_model, trust_remote_code=True)
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template=profile.template))
    print(f"profile {profile.name} | template {profile.template} | cutoff_len {cutoff} | file {args.file}")
    print(f"eos_token={tokenizer.eos_token!r} stop words={template.stop_words!r} efficient_eos={template.efficient_eos}")

    for i, (original, translated) in enumerate(pairs, 1):
        messages = [{"role": "user", "content": original}, {"role": "assistant", "content": translated}]
        prompt_ids, response_ids = template.encode_oneturn(tokenizer, messages)
        total = len(prompt_ids) + len(response_ids)
        print(f"\n=== row {i}: {total} tokens ({len(prompt_ids)} masked + {len(response_ids)} trained)" + (" -- OVER cutoff, truncated" if cutoff and total > cutoff else ""))
        print(f"input column  : {original!r}")
        print(f"output column : {translated!r}")
        print(f"MASKED  (prompt)  : {tokenizer.decode(prompt_ids, skip_special_tokens=False)!r}")
        print(f"TRAINED (response): {tokenizer.decode(response_ids, skip_special_tokens=False)!r}")


if __name__ == "__main__":
    main()
