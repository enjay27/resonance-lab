"""Stub transformers: a char-level tokenizer and a model that echoes the reference translation.

The tokenizer and the model check the arguments eval.py passes them (the ones the real classes need), so a
change that breaks the call fails the mock run. The reference comes from the eval set the pipeline copied
into the repo (data/eval/bp-eval-dataset.jsonl): the line whose `original` appears in the prompt.
"""

import json
import os

import torch

BOS, EOS = 1, 2
EVAL_SET = os.path.join("data", "eval", "bp-eval-dataset.jsonl")


def _check_model_dir(path):
    missing = [name for name in ("config.json", "model.safetensors", "tokenizer_config.json") if not os.path.isfile(os.path.join(path, name))]
    if missing:
        raise OSError(f"[MOCK CONTRACT] transformers: {path!r} lacks {', '.join(missing)}: it is not a merged Hugging Face model")


def _text_of(content):
    """The prompt text of a chat message: a string, or TranslateGemma's list of typed parts."""
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content)


class MockEncoding(dict):
    def to(self, device):
        return self


class AutoTokenizer:
    bos_token_id = BOS
    eos_token_id = EOS

    @classmethod
    def from_pretrained(cls, path, trust_remote_code=False):
        _check_model_dir(path)
        return cls()

    def __call__(self, text):
        return {"input_ids": [ord(ch) for ch in text]}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict, return_tensors, enable_thinking):
        assert tokenize and add_generation_prompt and return_dict and return_tensors == "pt", "unexpected apply_chat_template arguments"
        ids = [ord(ch) for message in messages for ch in _text_of(message["content"])]
        return MockEncoding(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long))

    def decode(self, tokens, skip_special_tokens=False):
        ids = list(tokens)
        if skip_special_tokens:
            ids = [i for i in ids if i not in (BOS, EOS)]
        return "".join({BOS: "<s>", EOS: "</s>"}.get(i, chr(i)) for i in ids)


class AutoModelForCausalLM:
    @classmethod
    def from_pretrained(cls, path, torch_dtype=None, device_map=None, trust_remote_code=False):
        _check_model_dir(path)
        return cls()

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask=None, max_new_tokens=None, pad_token_id=None, do_sample=None):
        assert do_sample is False and isinstance(max_new_tokens, int) and max_new_tokens > 0 and pad_token_id == EOS, "unexpected generate arguments"
        prompt = "".join(chr(i) for i in input_ids[0])
        with open(EVAL_SET, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        reply = next((row["translated"] for row in rows if row["original"] in prompt), "?")
        return torch.tensor([list(input_ids[0]) + [ord(ch) for ch in reply] + [EOS]])
