"""Which lines go into the training file, by category weight. Pure, no torch, no network.

A recipe (`configs/datasets/<name>.json`) says how much of the dataset each category gets, as a weight:

    {"seed": 42, "total": null, "keep": [], "categories": {"greetings": {"weight": 1}, "chat": {"weight": 9}}}

- A category's share of the dataset is its weight / the total weight (here 10% and 90%). Weights are relative: they
  need not add up to anything.
- A category is a lower-case path (`recruitment/party`). A key of the recipe covers its children; the most specific
  key wins. Lines whose category no key covers are left out. Lines with no entry in the categories file are
  `uncategorized`, which can get a weight like any category.
- `total` null = the largest dataset the weights allow: it ends where the first category runs out of lines.
- Lines are drawn per category by a salted hash rank, so the draw is the same on every machine and a smaller `total`
  is always inside a bigger one (a learning curve over the data is nested).
- A line belongs to a category by its key: the sha1 of the normalised line, as valsplit.py and overlap.py normalise it.
"""

import hashlib
import heapq
import json
import math
import os
import re
from fractions import Fraction
from typing import NamedTuple

from config import RECIPE_DIR
from overlap import match_key

UNCATEGORIZED = "uncategorized"
DEFAULT_SEED = 42
# preprocess.py drops these on its own; a recipe can let them through (Gate 1 is trusted to have categorised them).
KEEPABLE = ("recruitment spam",)

_CATEGORY = re.compile(r"^[a-z0-9_-]+(/[a-z0-9_-]+)*$")
_LINE_KEY = re.compile(r"^[0-9a-f]{40}$")
_RECIPE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_RECIPE_KEYS = {"description", "seed", "total", "keep", "categories"}


class RecipeError(ValueError):
    """A recipe or categories file that cannot be used; the message says which and why."""


class Recipe(NamedTuple):
    seed: int
    total: int | None
    keep: tuple
    weights: dict  # category key -> Fraction, relative sizes
    description: str | None = None

    @property
    def shares(self):
        """category key -> its share of the dataset: weight / total weight."""
        total = sum(self.weights.values())
        return {name: weight / total for name, weight in self.weights.items()}


class Allocation(NamedTuple):
    total: int  # lines that will be selected
    targets: dict  # category key -> lines
    limited_by: str | None  # the key that ran out of lines and ended the dataset, None when `total` was reached
    requested: int | None  # the recipe's `total`


class Selection(NamedTuple):
    keys: frozenset  # the line keys selected
    allocation: Allocation
    available: dict  # category key -> lines the pool has for it


# --- the recipe file ----------------------------------------------------------------------------------------------


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _weight(name, value):
    if not isinstance(value, dict):
        raise RecipeError(f"category {name!r}: expected {{\"weight\": number}}")
    for other in ("share", "fraction"):
        if other in value:
            raise RecipeError(f"category {name!r}: only 'weight' is supported (its relative size: the share is weight / total weight), not {other!r}")
    unknown = sorted(set(value) - {"weight"})
    if unknown:
        raise RecipeError(f"category {name!r} has unknown key(s): {', '.join(unknown)}")
    weight = value.get("weight")
    if not _is_number(weight) or not weight > 0:
        raise RecipeError(f"category {name!r}: weight must be a number above 0, got {weight!r}")
    return Fraction(str(weight))


def parse_recipe(mapping):
    """The recipe in `mapping` (the parsed JSON), or a RecipeError saying what is wrong."""
    if not isinstance(mapping, dict):
        raise RecipeError("a recipe is a JSON object")
    unknown = sorted(set(mapping) - _RECIPE_KEYS)
    if unknown:
        raise RecipeError(f"unknown key(s) in the recipe: {', '.join(unknown)} (known: {', '.join(sorted(_RECIPE_KEYS))})")

    categories = mapping.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise RecipeError("the recipe needs 'categories': {category: {\"weight\": number}, ...}")
    weights = {}
    for name, value in categories.items():
        if not isinstance(name, str) or not _CATEGORY.match(name):
            raise RecipeError(f"category {name!r} is not a lower-case path like 'recruitment/party'")
        weights[name] = _weight(name, value)

    seed = mapping.get("seed", DEFAULT_SEED)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise RecipeError(f"seed must be a whole number, got {seed!r}")
    total = mapping.get("total")
    if total is not None and (not isinstance(total, int) or isinstance(total, bool) or total <= 0):
        raise RecipeError(f"total must be a positive whole number or null, got {total!r}")

    keep = mapping.get("keep", [])
    if not isinstance(keep, (list, tuple)):
        raise RecipeError("keep must be a list of filters to let through")
    for reason in keep:
        if reason not in KEEPABLE:
            raise RecipeError(f"keep: {reason!r} cannot be kept (allowed: {', '.join(KEEPABLE)})")

    description = mapping.get("description")
    if description is not None and not isinstance(description, str):
        raise RecipeError("description must be text")
    return Recipe(seed=seed, total=total, keep=tuple(keep), weights=weights, description=description)


def recipe_path(name, directory=RECIPE_DIR):
    if not _RECIPE_NAME.match(name):
        raise RecipeError(f"{name!r} is not a recipe name (letters, digits, '_', '-', '.')")
    return os.path.join(directory, f"{name}.json")


def load_recipe(path):
    """(Recipe, sha256) of a recipe file. The hash is of the content, so reformatting the file does not change it."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise RecipeError(f"{path}: recipe not found") from None
    except (OSError, json.JSONDecodeError) as error:
        raise RecipeError(f"{path}: not a readable JSON file ({error})") from None
    try:
        recipe = parse_recipe(data)
    except RecipeError as error:
        raise RecipeError(f"{path}: {error}") from None
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return recipe, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- lines and their categories -------------------------------------------------------------------------------------


def line_key(text):
    """The key of a line: sha1 of its normalised text (the hash valsplit.bucket takes its number from)."""
    return hashlib.sha1((match_key(text) or (text or "")).encode("utf-8")).hexdigest()


def line_rank(seed, key):
    """A number in [0, 1) that depends on the seed and the line. Salted, so it is not the validation bucket."""
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / 16**12


def read_categories(path):
    """{line key: category} from a JSONL file of {"key": sha1, "category": path, ...} (other fields are ignored)."""
    categories = {}
    try:
        f = open(path, encoding="utf-8")
    except OSError as error:
        raise RecipeError(f"{path}: categories file cannot be read ({error.strerror or error})") from None
    with f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise RecipeError(f"{path}: line {number}: not JSON") from None
            key, category = (row.get("key"), row.get("category")) if isinstance(row, dict) else (None, None)
            if not isinstance(key, str) or not _LINE_KEY.match(key) or not isinstance(category, str) or not _CATEGORY.match(category):
                raise RecipeError(f"{path}: line {number}: needs a 40-character sha1 \"key\" and a lower-case path \"category\"")
            if categories.setdefault(key, category) != category:
                raise RecipeError(f"{path}: line {number}: {key[:8]} is in both {categories[key]!r} and {category!r}")
    return categories


def category_of(text, categories):
    return categories.get(line_key(text), UNCATEGORIZED)


def recipe_key(category, keys):
    """The most specific of `keys` that is `category` or one of its parents, None when no key covers it."""
    best = None
    for key in keys:
        if (category == key or category.startswith(key + "/")) and (best is None or len(key) > len(best)):
            best = key
    return best


# --- how many lines each key gets -----------------------------------------------------------------------------------


def allocate(weights, available, total=None):
    """Lines per key. `weights` are relative sizes (shares work too: only the ratio matters). Line j of a key enters
    the dataset at time (2j-1)/weight, and the dataset is the lines that enter
    first (Sainte-Lague: counts follow the shares, and a longer dataset only ever adds lines). It ends when `total`
    lines are in, or when the next line to enter does not exist: that key is `limited_by`."""
    names = sorted(weights)
    denominator = math.lcm(*(weights[name].denominator for name in names))
    whole = {name: int(weights[name] * denominator) for name in names}
    scale = math.lcm(*whole.values())
    step = {name: scale // whole[name] for name in names}

    targets = dict.fromkeys(names, 0)
    queue = [(step[name], name, 1) for name in names]  # (entry time, key, line number); ties go to the key name
    heapq.heapify(queue)
    taken, limited_by = [], None  # (entry time, key) of the lines in, in order
    while total is None or len(taken) < total:
        entry, name, number = heapq.heappop(queue)
        if number > available.get(name, 0):
            limited_by = name
            while taken and taken[-1][0] == entry:  # lines entering at the same time as the missing one stay out too,
                taken.pop()  # so the shares are exact at the end instead of favouring the key that sorts first
            break
        taken.append((entry, name))
        heapq.heappush(queue, ((2 * number + 1) * step[name], name, number + 1))
    for _, name in taken:
        targets[name] += 1
    return Allocation(total=len(taken), targets=targets, limited_by=limited_by, requested=total)


def select_lines(recipe, pairs):
    """The line keys the recipe selects from `pairs` of (line key, category)."""
    groups = {name: set() for name in recipe.weights}
    for key, category in pairs:
        name = recipe_key(category, recipe.weights)
        if name is not None:
            groups[name].add(key)
    available = {name: len(keys) for name, keys in groups.items()}
    allocation = allocate(recipe.weights, available, recipe.total)

    chosen = set()
    for name, keys in groups.items():
        ordered = sorted(keys, key=lambda key: (line_rank(recipe.seed, key), key))
        chosen.update(ordered[: allocation.targets[name]])
    return Selection(frozenset(chosen), allocation, available)
