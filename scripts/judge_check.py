"""Is the judge server up, new enough, and serving a decision model?

    python scripts/judge_check.py [--url http://127.0.0.1:8080]
    python scripts/judge_check.py --local [RUN] [--dtype bf16|fp32] [--device cuda|cpu]     (in the .venv-kev virtualenv)

Asks one trivial yes/no question of llama-server's /v1/systemone (or, with --local, of the Kev model loaded in this process:
no server, `judge_local.py`) and prints the model that answered; exits 1 with the reason (connection refused, llama.cpp too old,
not a decision model, the model does not load, ...) otherwise.
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import JUDGE_LOCAL_RUN, JUDGE_URL
from judge_client import JudgeError, SystemOneClient
from judge_local import DTYPES, LocalKevClient


def main(argv=None, post=None, loader=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--url", default=JUDGE_URL, help=f"llama-server's base URL (default {JUDGE_URL})")
    where.add_argument("--local", nargs="?", const=JUDGE_LOCAL_RUN, metavar="RUN",
                       help=f"load the Kev model in this process instead (a Hub id[@revision] or a directory; default {JUDGE_LOCAL_RUN})")
    parser.add_argument("--dtype", choices=DTYPES, default="bf16", help="with --local: the precision (default bf16)")
    parser.add_argument("--device", help="with --local: cuda or cpu (default: cuda when there is one)")
    args = parser.parse_args(argv)
    try:
        if args.local:
            client = LocalKevClient(args.local, device=args.device, dtype=args.dtype, loader=loader)
        else:
            client = SystemOneClient(args.url, post=post)
        model = client.check_server()
    except JudgeError as error:
        print(f"Judge check failed: {error}", file=sys.stderr)
        sys.exit(1)
    print(f"Judge OK: {args.local or args.url} answers with {model}")


if __name__ == "__main__":
    main()
