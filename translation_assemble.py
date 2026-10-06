"""The mechanics of translating chat lines with agents, apart from the agents: choosing the lines, batching them, putting the agents' outputs
together, correcting what is known to go wrong, the term table, the brief, and the guard that a glossary changed in between is noticed.

A season's work, in rounds: round 1 translates every selected line; a later round revises the lines the checks flagged (its input rows carry `prev`,
the earlier Korean). Assembling takes the LATEST round's row per line, applies the deterministic corrections of the glossary (`fixes`) once at the end,
and the checks (`translation_check.py`) run on the result. Pure: no files, no network, no model; `scripts/translate_agents.py` does the I/O.
"""

import collections
import datetime
import re

from glossary import VERSION_RX

# What is not translated by default: the maintainer's guild is Korean (guild adverts), the line is not Japanese, or it is the app's placeholder.
DEFAULT_SKIP = ("recruitment/guild", "non_japanese", "other/placeholder")
DROPPED_SECTIONS = ("Changelog", "Observed terms", "Updating for a new season")  # the brief is the rules, not the maintenance of the document

SLOT_ATTACHED = re.compile(r"(@[THD]+\d*)많이")
SLOT_SPACED = re.compile(r"(@[THD]+\d*) 많이")
ARROW_AFTER_SCORE = re.compile(r"(\d[\d,.]*[kKmM]?)\s*이상")


class AssembleError(Exception):
    """The input cannot be assembled; the message says which line or what to do."""


class StaleBatches(AssembleError):
    """The glossary document is not the version the batches were prepared with."""


# --- the lines and the batches -----------------------------------------------------------------------------------------------


def select_lines(labelled, skip=DEFAULT_SKIP):
    """(input rows, {skipped category: count}) from labelled lines `{original, category, channel}`. The id is the 1-based position in `labelled`,
    so ids do not move when the skip list changes. A skip entry covers its children (`recruitment` skips `recruitment/guild`)."""
    selected, skipped = [], collections.Counter()
    for number, row in enumerate(labelled, 1):
        original, category = row.get("original"), row.get("category")
        if not isinstance(original, str) or not original.strip() or not isinstance(category, str):
            raise AssembleError(f"labelled line {number}: needs `original` and `category`")
        if any(category == s or category.startswith(s + "/") for s in skip):
            skipped[category] += 1
            continue
        selected.append({"i": number, "ch": str(row.get("channel") or "?")[:1].upper(), "cat": category, "ja": original})
    return selected, dict(skipped)


def split_batches(rows, size, prefix):
    """[(name, rows)]: as few batches as hold `size` lines each at most, of even size, in order (`prefix1`, `prefix2`, ...)."""
    if size < 1:
        raise AssembleError(f"batch size must be at least 1, not {size}")
    count = max(1, -(-len(rows) // size))
    base, extra = divmod(len(rows), count)
    batches, start = [], 0
    for n in range(count):
        end = start + base + (n < extra)
        batches.append((f"{prefix}{n + 1}", rows[start:end]))
        start = end
    return batches


# --- the corrections ---------------------------------------------------------------------------------------------------------


def apply_fixes(ja, ko, terms, glossary):
    """(ko, terms, [why of each correction applied]): the glossary's `fixes`, the middle dot of the source kept, an arrow kept as an arrow."""
    terms, applied = [list(t) for t in terms], []

    def replace(old, new, why):
        nonlocal ko
        if old in ko:
            ko = ko.replace(old, new)
            for term in terms:
                term[1] = term[1].replace(old, new)
            applied.append(why)

    for fix in glossary.fixes:
        if fix.when.search(ja) and not (fix.unless and fix.unless.search(ja)):
            replace(fix.old, fix.new, fix.why)
    if "・" in ja and "･" not in ja and "·" not in ja:  # agents turn the katakana middle dot of a kaomoji or bullet into another dot
        for other in ("･", "·"):
            replace(other, "・", "restored ・ of the source")
    if "↑" in ja and "↑" not in ko and "以上" not in ja:
        arrowed = ARROW_AFTER_SCORE.sub(r"\1↑", ko)
        if arrowed != ko:
            ko = arrowed
            applied.append("↑ kept (was 이상)")
    return ko, terms, applied


def normalise_slot_spacing(rows):
    """`@D 많이` or `@D많이`: the form most of the lines use wins. Changes `translated` in place; returns how many rows changed."""
    spaced = sum(bool(SLOT_SPACED.search(r["translated"])) for r in rows)
    attached = sum(bool(SLOT_ATTACHED.search(r["translated"])) for r in rows)
    if not attached:
        return 0
    changed = 0
    for r in rows:
        text = SLOT_ATTACHED.sub(r"\1 많이", r["translated"]) if spaced >= attached else SLOT_SPACED.sub(r"\1많이", r["translated"])
        if text != r["translated"]:
            r["translated"] = text
            changed += 1
    return changed


# --- putting the rounds together ---------------------------------------------------------------------------------------------


def assemble(inputs, rounds, glossary):
    """(final rows, Counter of corrections applied): `inputs` {i: input row}, `rounds` the outputs of each round in order, the latest round's row
    per line wins (its flag replaces the earlier one)."""
    latest = {}
    for outputs in rounds:
        for r in outputs:
            if r.get("i") in inputs:
                latest[r["i"]] = r
    missing = sorted(set(inputs) - set(latest))
    if missing:
        raise AssembleError(f"not translated by any round: {missing[:30]}{' ...' if len(missing) > 30 else ''} ({len(missing)} lines)")
    rows, applied = [], collections.Counter()
    for i in sorted(inputs):
        source, out = inputs[i], latest[i]
        ko, terms, whys = apply_fixes(source["ja"], out["ko"], out.get("terms", []), glossary)
        applied.update(whys)
        row = {"i": i, "channel": source["ch"], "category": source["cat"], "original": source["ja"], "translated": ko, "terms": terms}
        if out.get("flag"):
            row["flag"] = out["flag"]
        rows.append(row)
    slots = normalise_slot_spacing(rows)
    if slots:
        applied["slot spacing normalised"] = slots
    return rows, applied


# --- the terms ---------------------------------------------------------------------------------------------------------------


def term_table(rows):
    """{Japanese term: {Korean rendering: uses}} from the `terms` of the rows."""
    table = {}
    for r in rows:
        for ja, ko in r.get("terms", []):
            table.setdefault(ja, collections.Counter())[ko] += 1
    return {ja: dict(counts.most_common()) for ja, counts in table.items()}


def multiple_renderings(table):
    """The terms rendered more than one way, the most used first: each is legitimate (say why in the glossary) or a mistake."""
    return [ja for ja, renderings in sorted(table.items(), key=lambda kv: -sum(kv[1].values())) if len(renderings) > 1]


def format_terms_tsv(table):
    lines = ["japanese\tuses\trenderings (count)\tconsistent"]
    for ja, renderings in sorted(table.items(), key=lambda kv: -sum(kv[1].values())):
        shown = " | ".join(f"{ko} ({n})" for ko, n in renderings.items())
        lines.append(f"{ja}\t{sum(renderings.values())}\t{shown}\t{'yes' if len(renderings) == 1 else 'NO'}")
    return "\n".join(lines) + "\n"


# --- what the batches were made with -----------------------------------------------------------------------------------------


def require_current(record, current_doc_version):
    """The guard: the glossary document must still be the version the run's batches were prepared with."""
    prepared = record["doc_version"]
    if prepared != current_doc_version:
        raise StaleBatches(f"the batches were prepared with the glossary document {prepared}, but it is {current_doc_version} now: "
                           "prepare them again (or restore the document) so the translations follow one glossary")


def run_record(season, doc_version, glossary_sha1, batch_size, round_no, counts, batches, git_sha):
    return {"season": season, "doc_version": doc_version, "glossary_sha1": glossary_sha1, "batch_size": batch_size, "git_sha": git_sha,
            "counts": counts, "rounds": {str(round_no): list(batches)},
            "prepared": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


def add_round(record, round_no, batches, doc_version):
    """The record with another round added; a round must be prepared with the glossary version the run began with."""
    require_current(record, doc_version)
    return {**record, "rounds": {**record["rounds"], str(round_no): list(batches)}}


# --- the brief -----------------------------------------------------------------------------------------------------------------

OUTPUT_FORMAT = """
## Output (this task)

Translate every input row. Write one JSON object per input row to your output file (UTF-8, one per line, `ensure_ascii` off), the same `i`:

    {"i": 123, "ko": "...", "terms": [["巨塔", "침식 거탑"], ["継", "계속"]], "flag": "only when unsure"}

- `ko`: the Korean translation (a string). No kana, no kanji.
- `terms`: every GAME TERM in the line as [Japanese, the Korean you used]; the Korean string must appear verbatim in `ko`. `[]` when none.
- `flag`: a short English note when the line is garbled, cut off or ambiguous (best reading in `ko`). Never skip a line.
- Each input row also has `ch` (channel W/P/L/B) and `cat` (a category label) as hints for what kind of line it is.
- When you are done, run the check you were given on your output file and fix every problem it lists.
"""

REVISION_FORMAT = """
## This is a revision round

Each input row has `prev`: an earlier Korean translation. For every row read the Japanese yourself, check `prev` against the glossary above, and write the final `ko`.
Apply the glossary exactly. If `prev` was already right, keep it unchanged character for character (do not paraphrase for the sake of it). Fix anything else that is wrong.
"""


def build_brief(doc_text, round_no):
    """The brief for the agents of a round: the glossary document without its maintenance sections (changelog, observed terms, season steps),
    the version it is, and the output format (plus, from round 2, how to treat `prev`)."""
    version = VERSION_RX.search(doc_text)
    sections = re.split(r"(?m)^(?=## )", doc_text)
    kept = [s for s in sections if not s.startswith("## ") or not any(s[3:].startswith(name) for name in DROPPED_SECTIONS)]
    brief = "".join(kept).rstrip() + "\n"
    brief += f"\n(Glossary document version {version.group(1) if version else 'unknown'}.)\n" + OUTPUT_FORMAT
    if round_no >= 2:
        brief += REVISION_FORMAT
    return brief
