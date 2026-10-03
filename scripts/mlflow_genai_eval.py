"""Per-sample evaluation in MLflow: `python scripts/mlflow_genai_eval.py [--model M] [--fast] [--prompt training|chat-template]`.

Run it after eval.py, which saves the translations of the eval set to outputs/eval/<profile>-<prompt>.jsonl. Every eval line becomes
a trace in the MLflow server of `.env.mlflow` with its own scores (chrF, JP leakage, term check, ...; genai_eval.py), in an
evaluation run of the experiment `resonance-lab-eval` that is tagged with the training run the merged model came from. The
training run itself keeps its run-level eval.* metrics; this adds the per-line view and replaces nothing.

A separate script, not a step of eval.py: `mlflow.genai.evaluate` needs the server at that moment, while the stages' tracker is
offline-first and never raises into a stage. No LLM and no API key is involved: the scores are computed here, offline.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "scripts", "llamafactory"))  # lf_tools: the model profiles are the llamafactory pipeline's
from config import EVAL_OUTPUT_DIR, MLFLOW_EVAL_EXPERIMENT
from genai_eval import build_scorers, evaluation_data, predictions_path, read_predictions
from lf_tools import add_model_argument, load_profile, model_name
from mlflow_compare import default_client
from runs import read_merge_record
from track_records import stage_run_id

PROMPT_CHOICES = ("chat-template", "training")  # eval.py's --prompt


def run_evaluation(data, scorers, experiment):
    """Evaluate the precomputed outputs with `scorers` in `experiment`; returns the evaluation run's id."""
    import mlflow

    mlflow.set_experiment(experiment)
    return mlflow.genai.evaluate(data=data, scorers=scorers).run_id


def main(argv=None, client=None, run_evaluation=run_evaluation, scorers=build_scorers):
    parser = argparse.ArgumentParser(description="Evaluate the saved predictions of the merged model, line by line, in MLflow.")
    parser.add_argument("--prompt", choices=PROMPT_CHOICES, default="chat-template",
                        help="which eval.py run to read (default: %(default)s)")
    parser.add_argument("--predictions", help="a predictions file (default: outputs/eval/<profile>-<prompt>.jsonl)")
    parser.add_argument("--experiment", default=MLFLOW_EVAL_EXPERIMENT, help="experiment name (default: %(default)s)")
    add_model_argument(parser)
    args = parser.parse_args(argv)

    profile = load_profile(model_name(args.model, fast=args.fast))
    path = args.predictions or predictions_path(EVAL_OUTPUT_DIR, profile.name, args.prompt)
    try:
        rows = read_predictions(path)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    if not rows:
        print(f"[ERROR] {path} has no rows")
        sys.exit(1)

    client = client or default_client()
    print(f"Profile {profile.name} | prompt: {args.prompt} | {len(rows)} lines from {path}")
    try:
        run_id = run_evaluation(evaluation_data(rows), scorers(), args.experiment)
    except ImportError as e:
        print(f"[ERROR] a package is missing: {e} (mlflow.genai.evaluate needs pandas next to mlflow-skinny: "
              "pip install -r requirements-llamafactory.txt)")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 - a server that does not answer is a message, not a traceback
        print(f"[ERROR] the evaluation failed: {' '.join(str(e).split())[:300]}")
        sys.exit(1)

    tags = {"profile": profile.name, "eval.prompt": args.prompt, "eval.n": str(len(rows))}
    training_run = stage_run_id(read_merge_record(profile.merged_dir))  # None: a model from before runs existed
    if training_run:
        tags["training_run"] = training_run
    try:
        for key, value in tags.items():
            client.set_tag(run_id, key, value)
    except Exception as e:  # noqa: BLE001 - the scores are already in; a failed tag must not hide that
        print(f"[WARN] could not tag the run {run_id}: {' '.join(str(e).split())[:200]}")
    print(f"Evaluation run {run_id} in experiment '{args.experiment}'.")


if __name__ == "__main__":
    main()
