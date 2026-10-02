"""Compare the runs of the experiment in one table: `python scripts/mlflow_compare.py [--profile hy] [--sort eval-loss]`.

Reads the runs from the MLflow server of `.env.mlflow` (the same settings the training uses) and prints, per finished run, the
profile, learning rate, epochs, best eval loss, the eval scores (chrF, term accuracy, JP leakage and which prompt) and the data
revision. `--markdown` prints a GitHub table for the memory notes; `--all` includes running, failed and killed runs.
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tracking
from compare_runs import SORT_KEYS, CompareError, build_rows, fetch_runs, render_markdown, render_text
from config import MLFLOW_ENV_FILE, MLFLOW_EXPERIMENT


def default_client():
    """An MlflowClient for the server named in .env.mlflow; exits 1 with the fix when there is none."""
    try:
        settings = tracking.tracking_settings(env_file=MLFLOW_ENV_FILE)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    if settings is None:
        print("[ERROR] tracking is not set up: copy .env.mlflow.example to .env.mlflow and fill in the NAS URL and password "
              "(deploy/mlflow/README.md); RESONANCE_MLFLOW=0 turns it off.")
        sys.exit(1)
    os.environ.update(tracking.client_environment(settings))
    try:
        import mlflow
    except ImportError:
        print("[ERROR] mlflow-skinny is not installed in this environment (pip install -r requirements-llamafactory.txt)")
        sys.exit(1)
    return mlflow.MlflowClient()


def main(argv=None, client=None):
    parser = argparse.ArgumentParser(description="Compare the runs of the experiment in one table.")
    parser.add_argument("--experiment", default=MLFLOW_EXPERIMENT, help="experiment name (default: %(default)s)")
    parser.add_argument("--profile", help="only runs whose profile or name contains this text, e.g. hy-mt2-1.8b-fast")
    parser.add_argument("--sort", choices=SORT_KEYS, default="created", help="best first for eval-loss / chrf / term (default: newest first)")
    parser.add_argument("--limit", type=int, default=20, help="at most this many rows (default: %(default)s)")
    parser.add_argument("--all", action="store_true", help="include runs that are not finished (running, failed, killed)")
    parser.add_argument("--markdown", action="store_true", help="a GitHub markdown table instead of plain text")
    args = parser.parse_args(argv)

    client = client or default_client()
    try:
        rows = build_rows(fetch_runs(client, args.experiment), profile=args.profile, sort=args.sort, limit=args.limit,
                          include_unfinished=args.all)
    except CompareError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 - a server that does not answer is a message, not a traceback
        print(f"[ERROR] could not read the runs from MLflow: {' '.join(str(e).split())[:200]}")
        sys.exit(1)
    print(render_markdown(rows) if args.markdown else render_text(rows))


if __name__ == "__main__":
    main()
