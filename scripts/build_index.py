"""Chuẩn bị word terms/count/TF-IDF từ corpus và transform query sạch."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.terms import build_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data/zalo_legal_2021/corpus.jsonl")
    parser.add_argument("--queries", type=Path, default=PROJECT_ROOT / "data/zalo_legal_2021/queries_clean.jsonl")
    parser.add_argument("--qrels", type=Path, help="Nếu có, kiểm tra mọi query có nhãn và ID tài liệu hợp lệ")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/terms.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/terms/word_clean")
    args = parser.parse_args()
    try:
        stats = build_index(args.corpus, args.queries, args.config, args.output_dir, args.qrels)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
