"""The categories of configs/category_taxonomy.json, and how many eval lines each root has. Pure, no torch.

A category is a lower-case path (`game/combat`); its root is the part before the first slash (`game`). Gate 1 chooses
between roots, a deeper gate between a root's children. The eval set's `category` field uses the same names, so the eval
report can score each root (eval_metrics.py).
"""

import json
from typing import NamedTuple

from config import CATEGORY_TAXONOMY


class TaxonomyError(Exception):
    """The taxonomy file cannot be used; the message names it."""


class Taxonomy(NamedTuple):
    roots: tuple  # the roots, in file order
    paths: dict  # every category path (roots and children) -> its description


def _flatten(categories, prefix, found):
    for name, node in categories.items():
        found[f"{prefix}{name}"] = node.get("description", "")
        _flatten(node.get("children", {}), f"{prefix}{name}/", found)
    return found


def load_taxonomy(path=None):
    path = path or CATEGORY_TAXONOMY
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise TaxonomyError(f"{path}: taxonomy not found") from None
    except (OSError, ValueError) as error:
        raise TaxonomyError(f"{path}: not a readable JSON file ({error})") from None
    categories = data.get("categories") if isinstance(data, dict) else None
    if not isinstance(categories, dict) or not categories or not all(isinstance(node, dict) for node in categories.values()):
        raise TaxonomyError(f"{path}: needs \"categories\": {{root: {{\"description\": ..., \"children\": {{...}}}}}}")
    return Taxonomy(roots=tuple(categories), paths=_flatten(categories, "", {}))


def root_of(category):
    return category.split("/", 1)[0]


def coverage(samples, taxonomy):
    """How many eval lines each root has: {"counts": {root: n} (every root, also the empty ones), "unknown": {label: n} for
    labels the taxonomy does not have (not counted under a root), "unlabeled": n lines without a category}."""
    counts = dict.fromkeys(taxonomy.roots, 0)
    unknown, unlabeled = {}, 0
    for sample in samples:
        label = sample.get("category")
        if not label:
            unlabeled += 1
        elif label in taxonomy.paths:
            counts[root_of(label)] += 1
        else:
            unknown[label] = unknown.get(label, 0) + 1
    return {"counts": counts, "unknown": unknown, "unlabeled": unlabeled}
