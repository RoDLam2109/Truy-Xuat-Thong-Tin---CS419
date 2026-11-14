"""Tải đúng dataset đã chọn và xuất schema chung cho pipeline clean."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_io import read_qrels, sha256, write_json, write_jsonl

DATASET = "minhnguyent546/zalo-ai-legal-text-retrieval-2021"


def export_dataset(corpus, queries, qrels_by_split, output_dir, provenance):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("Thư mục dữ liệu đã tồn tại; chọn thư mục mới để tránh trộn dataset")
    if set(qrels_by_split) != {"train", "test"}:
        raise ValueError("Cần cả qrels train và test")
    query_map = {}
    duplicate_queries = 0
    for row in queries:
        qid, text = row.get("query_id"), row.get("question")
        if not isinstance(qid, str) or not qid.strip() or not isinstance(text, str) or not text.strip():
            raise ValueError("Query cần query_id và question không rỗng")
        if qid in query_map:
            if query_map[qid]["text"] != text:
                raise ValueError(f"Query trùng ID nhưng khác nội dung: {qid}")
            duplicate_queries += 1
        else:
            query_map[qid] = {"id": qid, "text": text}
    if not query_map:
        raise ValueError("Không có query")
    split_ids, split_qrels = {}, {}
    for split, rows in qrels_by_split.items():
        converted, pairs, ids = [], set(), set()
        for row in rows:
            qid, did, raw_score = row.get("query_id"), row.get("corpus_id"), row.get("score")
            if qid not in query_map or not isinstance(did, str) or not did.strip():
                raise ValueError("Qrels thiếu query hoặc corpus ID không hợp lệ")
            score = float(raw_score)
            if not math.isfinite(score) or score < 0 or isinstance(raw_score, bool):
                raise ValueError("Score qrels phải hữu hạn và >= 0")
            if (qid, did) in pairs:
                raise ValueError("Qrels trùng cặp query/document")
            pairs.add((qid, did))
            ids.add(qid)
            converted.append((qid, did, int(score) if score.is_integer() else score))
        split_ids[split], split_qrels[split] = ids, converted
    if split_ids["train"] & split_ids["test"]:
        raise ValueError("Train và test trùng query ID")
    output_dir.mkdir(parents=True)
    doc_ids = set()

    def documents():
        for row in corpus:
            did = row.get("id")
            if not isinstance(did, str) or not did.strip() or did in doc_ids:
                raise ValueError("Corpus ID rỗng, sai kiểu hoặc bị trùng")
            title, text = row.get("title", ""), row.get("text")
            if not isinstance(title, str) or not isinstance(text, str) or not (title + text).strip():
                raise ValueError("Corpus title/text không hợp lệ hoặc nội dung rỗng")
            doc_ids.add(did)
            yield {"id": did, "title": title, "text": text}

    write_jsonl(output_dir / "corpus.jsonl", documents())
    if not doc_ids:
        raise ValueError("Corpus rỗng")
    write_jsonl(output_dir / "queries_all.jsonl", query_map.values())
    for split in ("train", "test"):
        query_file = "queries_train.jsonl" if split == "train" else "queries_clean.jsonl"
        write_jsonl(output_dir / query_file, (r for qid, r in query_map.items() if qid in split_ids[split]))
        path = output_dir / f"qrels_{split}.tsv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(["query-id", "corpus-id", "score"])
            writer.writerows(split_qrels[split])
        read_qrels(path, split_ids[split], doc_ids)
    stats = {
        "documents": len(doc_ids), "queries_all": len(query_map),
        "queries_train": len(split_ids["train"]), "queries_clean_test": len(split_ids["test"]),
        "qrels_train_rows": len(split_qrels["train"]), "qrels_test_rows": len(split_qrels["test"]),
        "duplicate_identical_query_rows_removed": duplicate_queries,
        "unlabelled_queries": len(set(query_map) - split_ids["train"] - split_ids["test"]),
    }
    write_json(output_dir / "manifest.json", {
        "source": provenance, "stats": stats,
        "clean_definition": "original test queries, not proof of perfect spelling",
        "files_sha256": {p.name: sha256(p) for p in sorted(output_dir.iterdir()) if p.is_file()},
        "script_sha256": sha256(Path(__file__)),
    })
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET, choices=[DATASET])
    parser.add_argument("--revision", default="main", help="Branch/tag hoặc commit; manifest luôn lưu commit thực tế")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/zalo_legal_2021")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data/hf_cache")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Thư mục output đã tồn tại; chọn đường dẫn mới")
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi

        # Resolve branch một lần để các config/split thuộc cùng snapshot.
        revision = HfApi().dataset_info(args.dataset, revision=args.revision, token=False).sha
        kwargs = {"path": args.dataset, "revision": revision, "cache_dir": str(args.cache_dir), "token": False}
        corpus = load_dataset(name="corpus", split="train", **kwargs)
        queries = load_dataset(name="queries", split="train", **kwargs)
        qrels = load_dataset(name="qrels", **kwargs)
        stats = export_dataset(corpus, queries, {split: qrels[split] for split in ("train", "test")},
                               args.output_dir, {"dataset": args.dataset, "revision": revision})
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
