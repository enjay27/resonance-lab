"""A table of the experiment's runs, for comparing sweeps without clicking through the MLflow UI. Pure, no mlflow import.

`scripts/mlflow_compare.py` reads the runs from the NAS and prints this table. A run is a plain dict
(`name`, `status`, `start_ms`, `params`, `metrics`, `tags`) with the names track_records.py / tracking.py log; what a run
never logged (no eval stage yet) is a dash.
"""

from datetime import datetime, timezone


class CompareError(Exception):
    """The runs cannot be listed (no such experiment, an unknown sort)."""


# (key, header): the columns, in order
COLUMNS = [
    ("run", "run"), ("status", "status"), ("profile", "profile"), ("lr", "lr"), ("epochs", "epochs"), ("batch", "batch"),
    ("eval_loss", "eval loss"), ("at_step", "@step"), ("train_loss", "train loss"), ("chrf", "chrF"), ("term_acc", "term acc"),
    ("jp_leak", "JP leak"), ("prompt", "prompt"), ("rows", "rows"), ("recipe", "recipe"), ("data", "data"), ("when", "when (UTC)"),
]
SORT_KEYS = ("created", "eval-loss", "chrf", "term")


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fixed(value, digits):
    return "-" if value is None else f"{value:.{digits}f}"


def _percent(value):
    return "-" if value is None else f"{value * 100:.1f}%"


def _general(text):
    """'0.0002' -> '0.0002', '1e-05' -> '1e-05', '3.0' -> '3'; text that is no number as it is; missing -> '-'."""
    if text is None:
        return "-"
    number = _number(text)
    return text if number is None else format(number, "g")


def summarize(run):
    """One run as a row of display text."""
    params, metrics, tags = run["params"], run["metrics"], run["tags"]
    revision = tags.get("dataset.revision")
    return {
        "run": run["name"], "status": run["status"], "profile": params.get("profile", "-"),
        "lr": _general(params.get("learning_rate")), "epochs": _general(params.get("num_train_epochs")),
        "batch": params.get("effective_batch", "-"),
        "eval_loss": _fixed(metrics.get("eval.best_loss"), 4), "at_step": _general(metrics.get("eval.best_loss_step")),
        "train_loss": _fixed(metrics.get("train.loss"), 4), "chrf": _fixed(metrics.get("eval.chrf"), 1),
        "term_acc": _percent(metrics.get("eval.term_accuracy")), "jp_leak": _percent(metrics.get("eval.jp_leak_rate")),
        "prompt": tags.get("eval.prompt", "-"), "rows": params.get("data.rows_passed", "-"), "recipe": tags.get("dataset.recipe", "-"),
        "data": revision[:8] if revision else "-",
        "when": datetime.fromtimestamp(run["start_ms"] / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M"),
    }


def _sort_key(sort):
    """(function giving a sortable key, newest first within ties): the best first, a run without the value last."""
    if sort == "created":
        return lambda run: (0, -run["start_ms"])
    metric, best_is_high = {"eval-loss": ("eval.best_loss", False), "chrf": ("eval.chrf", True), "term": ("eval.term_accuracy", True)}[sort]

    def key(run):
        value = run["metrics"].get(metric)
        if value is None:
            return (1, 0, -run["start_ms"])
        return (0, -value if best_is_high else value, -run["start_ms"])

    return key


def build_rows(runs, profile=None, sort="created", limit=None, include_unfinished=False):
    """The rows to show: finished runs (all with `include_unfinished`), of the profiles/runs whose name contains `profile`,
    sorted (`SORT_KEYS`), at most `limit`."""
    if sort not in SORT_KEYS:
        raise CompareError(f"unknown sort {sort!r}; choose one of: {', '.join(SORT_KEYS)}")
    chosen = [r for r in runs if include_unfinished or r["status"] == "FINISHED"]
    if profile:
        needle = profile.lower()
        chosen = [r for r in chosen if needle in r["params"].get("profile", "").lower() or needle in r["name"].lower()]
    chosen.sort(key=_sort_key(sort))
    return [summarize(r) for r in (chosen[:limit] if limit else chosen)]


def render_text(rows):
    """The table as aligned plain text (every line the same width)."""
    if not rows:
        return "no runs to show (the experiment has none finished; --all includes the others)"
    widths = [max(len(header), *(len(row[key]) for row in rows)) for key, header in COLUMNS]
    lines = ["  ".join(header.ljust(width) for (key, header), width in zip(COLUMNS, widths))]
    lines += ["  ".join(row[key].ljust(width) for (key, header), width in zip(COLUMNS, widths)) for row in rows]
    return "\n".join(lines)


def render_markdown(rows):
    """The table as GitHub markdown, for the memory notes."""
    def cell(text):
        return text.replace("|", "\\|")

    lines = ["| " + " | ".join(header for _, header in COLUMNS) + " |", "|" + "|".join(" --- " for _ in COLUMNS) + "|"]
    lines += ["| " + " | ".join(cell(row[key]) for key, _ in COLUMNS) + " |" for row in rows]
    return "\n".join(lines)


def fetch_runs(client, experiment, max_results=500):
    """The experiment's runs, newest first, as plain dicts. `client` is an MlflowClient (or a stand-in)."""
    found = client.get_experiment_by_name(experiment)
    if found is None:
        raise CompareError(f"no experiment {experiment!r} on the server")
    runs = client.search_runs([found.experiment_id], max_results=max_results, order_by=["attributes.start_time DESC"])
    return [{"name": r.info.run_name, "status": r.info.status, "start_ms": r.info.start_time,
             "params": dict(r.data.params), "metrics": dict(r.data.metrics), "tags": dict(r.data.tags)} for r in runs]
