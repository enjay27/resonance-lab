"""What the Gate judge is asked: the question's instructions and the options, with the judge-facing text kept apart from the taxonomy.

`configs/category_taxonomy.json` describes the categories for people and for the dataset recipes; its root descriptions used to be
sent to the judge as they are, including notes meant for the translator's training data. A prompt variant
(`configs/judge_prompts/<name>.json`) changes only what the judge reads, so different wordings can be tried and scored against each
other (`categorize.py --probe ... --judge-prompt NAME --save-probe LABEL`, then `compare_judges.py`):

    {"description": "what this variant is for",          required for a shipped file
     "instructions": "replaces the question sentence",    optional
     "descriptions": {"chat": "text the judge reads"},    optional: replaces the description of a root
     "drop": ["other"]}                                   optional: roots left out of the question (the cutoff abstains on unclear lines)

"default" is no file: today's question exactly. The judge id hashes the instructions and the options, so every variant is another judge
(its journal and its saved probe never mix with another's). Pure; tested against the real taxonomy.
"""

import json
import os
import re
from typing import NamedTuple

from config import JUDGE_PROMPT_DIR
from gate_judge import INSTRUCTIONS, choice_options

DEFAULT = "default"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_KEYS = {"description", "instructions", "descriptions", "drop"}


class PromptError(ValueError):
    """A prompt variant that cannot be used; the message says which and why."""


class Variant(NamedTuple):
    description: str | None = None
    instructions: str | None = None
    descriptions: dict = {}
    drop: tuple = ()


def parse_variant(mapping):
    if not isinstance(mapping, dict):
        raise PromptError("a prompt variant is a JSON object")
    unknown = sorted(set(mapping) - _KEYS)
    if unknown:
        raise PromptError(f"unknown field {', '.join(unknown)} (known: {', '.join(sorted(_KEYS))})")
    instructions = mapping.get("instructions")
    if instructions is not None and (not isinstance(instructions, str) or not instructions.strip()):
        raise PromptError("instructions must be a non-empty text")
    descriptions = mapping.get("descriptions", {})
    if not isinstance(descriptions, dict):
        raise PromptError("descriptions must be an object {root: text}")
    for root, text in descriptions.items():
        if not isinstance(text, str) or not text.strip():
            raise PromptError(f"the description of {root!r} must be a non-empty text")
    drop = mapping.get("drop", [])
    if not isinstance(drop, list) or not all(isinstance(root, str) for root in drop):
        raise PromptError("drop must be a list of root names")
    return Variant(mapping.get("description"), instructions, dict(descriptions), tuple(drop))


def list_variants(directory=JUDGE_PROMPT_DIR):
    """The names there are: `default` and every <name>.json of the folder, sorted."""
    found = {DEFAULT}
    if os.path.isdir(directory):
        found |= {name[:-5] for name in os.listdir(directory) if name.endswith(".json") and _NAME.match(name[:-5])}
    return sorted(found)


def load_variant(name, directory=JUDGE_PROMPT_DIR):
    """The Variant called `name`; `default` is no file (today's question)."""
    if not isinstance(name, str) or not _NAME.match(name):
        raise PromptError(f"{name!r} is not a prompt variant name (letters, digits, '.', '_', '-')")
    if name == DEFAULT:
        return Variant()
    path = os.path.join(directory, f"{name}.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise PromptError(f"no prompt variant {name!r} (there are: {', '.join(list_variants(directory))})") from None
    except (OSError, json.JSONDecodeError) as error:
        raise PromptError(f"{path}: not a readable JSON file ({error})") from None
    try:
        return parse_variant(data)
    except PromptError as error:
        raise PromptError(f"{path}: {error}") from None


def judge_question(taxonomy, variant=None):
    """(instructions, {root: description}) the judge is asked, with the variant applied to the taxonomy's roots (None = default)."""
    variant = variant or Variant()
    options = choice_options(taxonomy)
    for root in (*variant.descriptions, *variant.drop):
        if root not in options:
            raise PromptError(f"{root!r} is not a root category of the taxonomy (roots: {', '.join(options)})")
    both = sorted(set(variant.descriptions) & set(variant.drop))
    if both:
        raise PromptError(f"{', '.join(both)} is both described and dropped")
    options = {root: variant.descriptions.get(root, text) for root, text in options.items() if root not in variant.drop}
    if len(options) < 2:
        raise PromptError("a question needs at least two options to choose between")
    return variant.instructions or INSTRUCTIONS, options
