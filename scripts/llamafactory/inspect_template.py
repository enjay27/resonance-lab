"""Check the prompt strings against the real tokenizer: `python scripts/llamafactory/inspect_template.py [--model <hf id or dir>]`.

Needs only `transformers` and the model's tokenizer files (no GPU). Prints, for the active profile's
template, (1) the model's own chat template output, (2) what `training_prompt` says training saw and
how the tokenizer encodes it (is a BOS added? the special tokens), and exits 1 if (2) does not end with
the generation prompt of (1)'s user turn -- i.e. if lf_tools' TRAINING_PROMPTS disagree with the tokenizer.
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import LF_PROFILE
from lf_tools import chat_messages, load_profile, training_prompt

SAMPLE = "遺跡1Fから　29k↑　＠T1"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect the chat template and training prompt of a profile's model.")
    parser.add_argument("--model", help="HF id or local dir (default: the profile's base model)")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    profile = load_profile(LF_PROFILE)
    model = args.model or profile.base_model
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    print(f"profile {profile.name} | template {profile.template} | tokenizer {model}")
    print(f"bos_token={tokenizer.bos_token!r} eos_token={tokenizer.eos_token!r} pad_token={tokenizer.pad_token!r}")

    chat = tokenizer.apply_chat_template(chat_messages(profile.template, SAMPLE), tokenize=False, add_generation_prompt=True)
    print(f"\n[chat template, generation prompt]\n{chat!r}")

    train = training_prompt(profile.template, SAMPLE)
    ids = tokenizer(train)["input_ids"]
    print(f"\n[training_prompt]\n{train!r}\nfirst ids {ids[:6]} -> {tokenizer.convert_ids_to_tokens(ids[:6])}")
    print(f"tokenizer adds BOS on its own: {ids[:1] == [tokenizer.bos_token_id]}")

    # The raw-line part: the chat prompt of the official instruction contains the line, the training one is the line alone.
    plain = tokenizer.apply_chat_template([{"role": "user", "content": SAMPLE}], tokenize=False, add_generation_prompt=True)
    print(f"\n[chat template, raw line as the user turn]\n{plain!r}")
    ok = plain.removeprefix(tokenizer.bos_token or "") == train
    print("\nMATCH: training_prompt == the template's raw-line prompt (BOS aside)" if ok else "\nMISMATCH: fix TRAINING_PROMPTS in lf_tools.py")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
