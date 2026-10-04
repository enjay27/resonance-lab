"""Gate 1 with a judge: a decision model on a local llama-server puts a chat line in a root category. Pure, no torch.

The judge pass writes one journal row per judged line ({key, by, choice, margin, p, probabilities}) and flushes it, so a
pass over a big log can stop and resume, and the cutoff is chosen afterwards without asking the model again: the categories
file (`{key, category, by, p, margin}`, what `preprocess.py --recipe` reads) is *derived* from the journal, keeping the lines
whose margin (top probability minus the second) reaches the cutoff. A line below the cutoff is left uncategorized, never
forced. `by` names the model and a hash of the question, so two judges never mix in one journal.
"""

import hashlib
import json
import os

from gate_eval import evaluate_categorizer
from judge_client import JudgeError
from taxonomy import TaxonomyError

INSTRUCTIONS = (
    "Which category does this in-game chat message belong to? The message is Japanese text written by a player "
    "of the online game Blue Protocol: Star Resonance."
)


def choice_options(taxonomy, root=None):
    """{category: description} to choose between: the roots, or the direct children of `root` (by path, `game/combat`)."""
    if root is None:
        return {name: taxonomy.paths[name] for name in taxonomy.roots}
    if root not in taxonomy.paths:
        raise TaxonomyError(f"'{root}' is not a category of the taxonomy")
    prefix = root + "/"
    options = {path: text for path, text in taxonomy.paths.items() if path.startswith(prefix) and "/" not in path[len(prefix):]}
    if not options:
        raise TaxonomyError(f"'{root}' has no children to choose between")
    return options


def judge_id(model, instructions, options):
    """`systemone:<model>:<hash8>`: the model (its file name, without .gguf) and a hash of the question it answered."""
    name = os.path.basename(model or "unknown").removesuffix(".gguf")
    question = json.dumps({"instructions": instructions, "options": options}, ensure_ascii=False, sort_keys=True)
    return f"systemone:{name}:{hashlib.sha1(question.encode('utf-8')).hexdigest()[:8]}"


def state_for(text, channel=None):
    """What the judge reads: the line, or the line with its chat channel (a party channel recruits, world chat chats)."""
    return {"channel": channel, "message": text} if channel else text


class GateJudge:
    """A client + the options + the cutoff. Asks once per (line, channel); a failed request is no answer, counted in `failures`."""

    def __init__(self, client, options, cutoff, instructions=INSTRUCTIONS):
        self.client, self.options, self.cutoff, self.instructions = client, options, cutoff, instructions
        self.failures = 0
        self._answers = {}

    def answer(self, text, channel=None):
        key = (text, channel or None)
        if key not in self._answers:
            try:
                self._answers[key] = self.client.choice(state_for(text, channel), self.instructions, self.options)
            except JudgeError:
                self.failures += 1
                return None
        return self._answers[key]

    def predict(self, text, channel=None):
        """The category, or None when the judge failed or its margin is below the cutoff."""
        answer = self.answer(text, channel)
        return answer.choice if answer is not None and answer.margin >= self.cutoff else None


def cutoff_sweep(rows, judge, cutoffs):
    """The judge's scores on a labelled sample at each cutoff, from one pass (every line is asked once): a list of
    {cutoff, accuracy, coverage, precision, covered, correct}."""
    answers = {row["original"]: judge.answer(row["original"], row.get("channel")) for row in rows}
    sweep = []
    for cutoff in cutoffs:
        def predict(text, cutoff=cutoff):
            answer = answers[text]
            return answer.choice if answer is not None and answer.margin >= cutoff else None

        report = evaluate_categorizer(rows, predict)
        sweep.append({"cutoff": cutoff, **{k: report[k] for k in ("accuracy", "coverage", "precision", "covered", "correct")}})
    return sweep


SWEEP_CUTOFFS = [round(0.1 * n, 1) for n in range(10)]


def format_sweep(sweep, name):
    """The cutoff sweep as a table: what the judge would score at each cutoff (counts next to the percentages)."""
    lines = [f"Cutoff sweep for {name} (one pass over the sample; a sample of a few hundred lines overfits the cutoff):",
             f"  {'cutoff':>6}{'answered':>10}{'right':>7}{'accuracy':>10}{'coverage':>10}{'precision':>11}"]
    for s in sweep:
        precision = "-" if s["precision"] is None else f"{s['precision']:.0%}"
        lines.append(f"  {s['cutoff']:>6.1f}{s['covered']:>10}{s['correct']:>7}{s['accuracy']:>10.0%}{s['coverage']:>10.0%}{precision:>11}")
    return "\n".join(lines)


# --- the journal ----------------------------------------------------------------------------------------------------


class JournalError(Exception):
    """The judge journal cannot be used; the message names the file."""


def journal_row(key, answer, by):
    top = max(answer.probabilities.values())
    return {"key": key, "by": by, "choice": answer.choice, "margin": round(answer.margin, 4), "p": round(top, 4),
            "probabilities": {option: round(p, 4) for option, p in answer.probabilities.items()}}


def read_journal(path, by):
    """{line key: row} of the judge journal (the first row of a key wins; a missing file is empty). A torn last line (a crash
    in the middle of a write) is ignored; any other damaged line, or a row by another judge, is a JournalError."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return {}
    rows = {}
    last = max((n for n, line in enumerate(lines, 1) if line.strip()), default=0)
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            if number == last:
                continue
            raise JournalError(f"{path} line {number}: not valid JSON") from None
        if not isinstance(row, dict) or not isinstance(row.get("key"), str):
            raise JournalError(f"{path} line {number}: not a journal row")
        if row.get("by") != by:
            raise JournalError(f"{path} line {number}: written by {row.get('by')!r}, this run is {by!r}: "
                               "use another --journal, or --force to start the journal again")
        rows.setdefault(row["key"], row)
    return rows


def trim_torn_tail(path):
    """Make the journal end on a complete line before rows are appended: cut a torn last line, add a missing newline."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return
    if not data or data.endswith(b"\n"):
        return
    head, _, tail = data.rpartition(b"\n")
    try:
        json.loads(tail.decode("utf-8"))
        fixed = data + b"\n"
    except ValueError:
        fixed = head + b"\n" if head else b""
    with open(path, "wb") as f:
        f.write(fixed)


def categories_from(rows, cutoff):
    """(category rows for the categories file, number of lines below the cutoff) from journal rows."""
    categories, uncertain = [], 0
    for row in rows.values():
        if row["margin"] >= cutoff:
            categories.append({"key": row["key"], "category": row["choice"], "by": row["by"], "p": row["p"], "margin": row["margin"]})
        else:
            uncertain += 1
    return categories, uncertain


def write_categories(path, categories):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in categories:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
