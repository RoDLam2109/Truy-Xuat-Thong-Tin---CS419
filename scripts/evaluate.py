"""Evaluate an exported run without fitting or running any retrieval model."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.runs import evaluate_files, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--statuses", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output already exists; choose a new directory")
    per_query, summary = evaluate_files(args.rankings, args.statuses,
                                         args.queries, args.qrels)
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "per_query.csv", per_query)
    write_json(args.output_dir / "metrics.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
