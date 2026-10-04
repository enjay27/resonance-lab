"""Which Gate judge is better? Compares the saved probes (one per judge) on the labelled sample.

    python scripts/categorize.py --probe --judge-url http://127.0.0.1:8080 --save-probe 9b-q8        # once per judge
    python scripts/categorize.py --probe --judge-local jaredpalmer/kev-4b@v1.0 --save-probe 4b-bf16
    python scripts/compare_judges.py [--cutoff 0.3] [--only 9b-q8,4b-bf16]

Needs no model and no server: it reads data/eval/gate1-compare/*.json (what each judge answered) and the CURRENT labelled
sample, so correcting a label needs no new run. Prints, per judge, the accuracy / coverage / precision at the cutoff, the model's
own accuracy without a cutoff (with its 95% interval), the most it answers at 90% precision, its speed, and for each pair of
judges where they differ and who is right. Exits 1 with the reason when there is nothing to compare.
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GATE1_COMPARE_DIR, GATE1_SAMPLE, JUDGE_CUTOFF
from gate_compare import CompareError, compare_probes, format_comparison, load_probes
from gate_eval import SampleError, read_sample


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sample", default=GATE1_SAMPLE, help="the hand-labelled sample (default: %(default)s)")
    parser.add_argument("--dir", default=GATE1_COMPARE_DIR, help="the saved probes (default: %(default)s)")
    parser.add_argument("--cutoff", type=float, default=JUDGE_CUTOFF, help="the margin a pass would use (default: %(default)s)")
    parser.add_argument("--target-precision", type=float, default=0.9, help="precision for the 'coverage at' column (default: %(default)s)")
    parser.add_argument("--only", metavar="A,B", help="compare only these labels")
    args = parser.parse_args(argv)
    try:
        rows = read_sample(args.sample)
        records = load_probes(args.dir)
        if args.only:
            wanted = [label.strip() for label in args.only.split(",") if label.strip()]
            unknown = sorted(set(wanted) - {record["label"] for record in records})
            if unknown:
                raise CompareError(f"no saved probe named {', '.join(unknown)} in {args.dir} (there are: {', '.join(r['label'] for r in records) or 'none'})")
            records = [record for record in records if record["label"] in wanted]
        print(format_comparison(compare_probes(records, rows, args.cutoff, args.target_precision)))
    except (CompareError, SampleError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
