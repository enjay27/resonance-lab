"""The logic behind notebooks/gate_pipeline.ipynb: the dataset pipeline's Gate steps, from a raw log to a recipe preview.

The notebook is thin; what is here is pure and tested (no torch, no network, no judge): the stratified draft of the labelled
sample that a human then corrects, the channel mix of the raw log, what a dataset recipe would take from the categories, the
text of the decision for the memory notes, and whether the model fits the card. The judge itself is `gate_judge.GateJudge`
over either client (judge_client.SystemOneClient or judge_local.LocalKevClient); the scores are gate_eval's.
"""

import json
import os
import re

from dataset_recipe import line_rank, load_recipe, recipe_path, select_lines

UNGUESSED = "?"  # the draft's label for a line the judge gave no answer for: not a category, so read_sample refuses it until a human labels the line


# --- the draft of the labelled sample ---------------------------------------------------------------------------------


def draft_sample(lines, size, seed, guess, exclude=frozenset()):
    """About `size` rows {original, category [, channel]} to be corrected by hand and saved as the labelled sample.

    `lines` are (key, original, channel) (categorize.distinct_lines); `guess(original, channel) -> category or None` is the
    judge's (or the rules') answer, which becomes the draft's label. The draw is stratified by the guess, the same number of lines
    from each category (a category that runs out gives its share to the others): a sample of the natural mix would hold a
    handful of rare categories, and the rare ones are where a judge can be wrong. Seeded, and independent of the order of
    `lines`; lines whose key is in `exclude` (the sample already labelled) are never drawn. The result is grouped by category."""
    strata, seen = {}, set(exclude)
    for key, original, channel in lines:
        if key in seen:
            continue
        seen.add(key)
        strata.setdefault(guess(original, channel) or UNGUESSED, []).append((line_rank(seed, key), key, original, channel))
    for rows in strata.values():
        rows.sort(key=lambda row: (row[0], row[1]))
    taken = dict.fromkeys(strata, 0)
    total = 0
    while total < size and any(taken[name] < len(strata[name]) for name in strata):
        for name in sorted(strata):
            if total < size and taken[name] < len(strata[name]):
                taken[name] += 1
                total += 1
    rows = []
    for name in sorted(strata):
        for _, _, original, channel in strata[name][: taken[name]]:
            rows.append({"original": original, "category": name, **({"channel": channel} if channel else {})})
    return rows


def draft_path(sample_path):
    """`data/eval/gate1-sample.jsonl` -> `data/eval/gate1-sample.draft.jsonl`: the draft never overwrites the corrected file."""
    root, extension = os.path.splitext(sample_path)
    return f"{root}.draft{extension}"


def write_draft(path, rows, overwrite=False):
    """The draft as JSONL (UTF-8). Refuses to replace a file that exists: it may be the draft a human is correcting."""
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} exists (it may hold labels you corrected): move it, or pass overwrite=True")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --- the channel mix --------------------------------------------------------------------------------------------------


def channel_report(lines):
    """{total, with_channel, channels: {name: lines}} over (key, original, channel) lines."""
    channels, total = {}, 0
    for _, _, channel in lines:
        total += 1
        if channel:
            channels[channel] = channels.get(channel, 0) + 1
    return {"total": total, "with_channel": sum(channels.values()), "channels": dict(sorted(channels.items(), key=lambda item: -item[1]))}


def format_channel_report(report):
    if not report["with_channel"]:
        return (f"{report['total']:,} distinct lines, no line has a channel (the unified file of the pinned dataset revision has none; "
                "per-channel files do): --use-channel has nothing to give the judge.")
    lines = [f"{report['with_channel']:,} of {report['total']:,} distinct lines have a channel:"]
    lines += [f"  {name:<14}{count:>9,}" for name, count in report["channels"].items()]
    return "\n".join(lines)


# --- a recipe against the categories ----------------------------------------------------------------------------------


def recipe_preview(recipe, categories, keys):
    """What `recipe` would take from the categorised lines: per recipe key its share, the share the lines naturally have, the lines
    available and the lines it would select. `categories` is {line key: category}, `keys` the distinct line keys of the log.
    An upper bound on the training file: preprocess drops more lines (duplicates, eval overlap, spam filters) after this."""
    keys = list(keys)
    pairs = [(key, categories[key]) for key in keys if key in categories]
    selection = select_lines(recipe, pairs)
    covered = sum(selection.available.values())
    shares = recipe.shares
    rows = [{"key": key, "share": float(shares[key]), "natural": selection.available[key] / covered if covered else 0.0,
             "available": selection.available[key], "target": selection.allocation.targets[key]}
            for key in sorted(recipe.weights, key=lambda key: (-shares[key], key))]
    return {"rows": rows, "total": selection.allocation.total, "limited_by": selection.allocation.limited_by,
            "requested": selection.allocation.requested, "distinct": len(keys), "uncategorized": len(keys) - len(pairs),
            "outside": len(pairs) - covered}


def recipe_preview_by_name(name, categories, keys):
    """`recipe_preview` of a recipe file of configs/datasets/ by its name (RecipeError when it is not there or not usable)."""
    recipe, _ = load_recipe(recipe_path(name))
    return recipe_preview(recipe, categories, keys)


def format_recipe_preview(preview, name):
    lines = [f"--- Recipe {name} on {preview['distinct']:,} distinct lines: would select {preview['total']:,} ---",
             f"  {'category':<16}{'recipe':>8}{'natural':>9}{'available':>11}{'selected':>10}"]
    for row in preview["rows"]:
        mark = "  <- limits the dataset" if row["key"] == preview["limited_by"] else ""
        lines.append(f"  {row['key']:<16}{row['share']:>8.0%}{row['natural']:>9.0%}{row['available']:>11,}{row['target']:>10,}{mark}")
    lines.append(f"  {preview['uncategorized']:,} lines have no category (uncertain or unjudged), {preview['outside']:,} are in a category the recipe does not name.")
    return "\n".join(lines)


# --- the decision ---------------------------------------------------------------------------------------


def _percent(value):
    return "-" if value is None else f"{value:.0%}"


def decision_text(by, cutoff, judge, rules, seconds_per_line=None, lines=None, categorized=None):
    """A markdown block for the memory notes: which judge, the cutoff, and its scores on the labelled sample against the rules'.
    `judge` and `rules` are gate_eval reports (at the chosen cutoff); `lines` / `categorized` are the full pass's counts."""
    beats = judge["accuracy"] > rules["accuracy"]
    out = [f"- judge `{by}`, cutoff {cutoff} (picked on the labelled sample: {judge['n']} lines)",
           f"- on the {judge['n']} lines: accuracy {_percent(judge['accuracy'])}, coverage {_percent(judge['coverage'])}, "
           f"precision {_percent(judge['precision'])}; the rules {_percent(rules['accuracy'])} / {_percent(rules['coverage'])} / {_percent(rules['precision'])}",
           f"- the judge {'beats' if beats else 'does NOT beat'} the rules (accuracy {_percent(judge['accuracy'])} vs {_percent(rules['accuracy'])})"]
    if seconds_per_line is not None:
        out.append(f"- speed: {seconds_per_line:g} s per line")
    if lines is not None and categorized is not None:
        out.append(f"- full pass: {categorized:,} of {lines:,} distinct lines categorised ({_percent(categorized / lines if lines else None)}), the rest uncertain or unjudged")
    return "\n".join(out)


# --- will the model fit the card --------------------------------------------------------------------------------------

_SIZE = re.compile(r"kev-(\d+(?:\.\d+)?)b", re.IGNORECASE)
_BYTES_PER_PARAMETER = {"bf16": 2, "fp32": 4}
_OVERHEAD_GB = 1.0  # the pointer head, the activations of a short state, the CUDA context: kev's README measures ~9 GB for Kev-4B in bf16


def estimated_gb(run, dtype):
    """About how much GPU memory the model takes, from the size in the run's name (`kev-9b`) and the precision; None when the name
    does not say (a local directory). The weights are merged into the base, so it is the base's size."""
    found = _SIZE.search(run)
    if not found or dtype not in _BYTES_PER_PARAMETER:
        return None
    return float(found.group(1)) * _BYTES_PER_PARAMETER[dtype] + _OVERHEAD_GB


def memory_warning(run, dtype, free_gb):
    """Text when the model will not fit in `free_gb` of GPU memory, None when it fits or cannot be judged (CPU, unknown size).
    kev has no 4-bit / 8-bit loading: the way down is a smaller model."""
    needed = estimated_gb(run, dtype)
    if needed is None or free_gb is None or needed <= free_gb:
        return None
    return (f"{run} ({dtype}) needs about {needed:.0f} GB and {free_gb:.1f} GB are free: it will probably run out of memory. "
            "Free the GPU (a training run, llama-server) or use a smaller model (jaredpalmer/kev-4b is ~9 GB, kev-0.8b ~3 GB).")
