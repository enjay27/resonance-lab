"""Is the judge server up, new enough, and serving a decision model?

    python scripts/judge_check.py [--url http://127.0.0.1:8080]

Asks one trivial yes/no question of llama-server's /v1/systemone and prints the model that answered; exits 1 with the reason
(connection refused, llama.cpp too old, not a decision model, ...) otherwise.
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import JUDGE_URL
from judge_client import JudgeError, SystemOneClient


def main(argv=None, post=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default=JUDGE_URL, help=f"llama-server's base URL (default {JUDGE_URL})")
    args = parser.parse_args(argv)
    try:
        model = SystemOneClient(args.url, post=post).check_server()
    except JudgeError as error:
        print(f"Judge check failed: {error}", file=sys.stderr)
        sys.exit(1)
    print(f"Judge OK: {args.url} answers with {model}")


if __name__ == "__main__":
    main()
