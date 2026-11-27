"""Bổ sung query/qrels train Zalo Legal từ nguồn cố định, loại trùng test.

Chỉ dùng thư viện chuẩn. Không ghi vào corpus, query test hoặc query nhiễu.
"""

import argparse
import csv
import hashlib
import json
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = "GreenNode/zalo-ai-legal-text-retrieval-vn"
REVISION = "12d76d4d04b94ceada970fcfbe7fec20bfa97389"
SOURCE_HASHES = {
    "queries.jsonl": "7cf566b867fbd5cd6428058dea1115593c08c4b0ce0d7b0b00d6b40ae24ee509",
    "qrels/train.jsonl": "8f21b5bb07a34bd67c6df1c31435feab44986c8be717854f92291faa67329b1f",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: bản ghi phải là object")
            yield row


def required_string(row, key):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} phải là chuỗi không rỗng")
    return value


def text_key(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def prepare_rows(source_queries, source_qrels, test_queries, doc_ids):
    """Giữ thứ tự query gốc và toàn bộ nhãn của query được giữ lại."""
    queries, duplicate_queries = {}, 0
    for row in source_queries:
        qid = required_string(row, "_id")
        text = unicodedata.normalize("NFC", required_string(row, "text"))
        if qid in queries:
            if queries[qid]["text"] != text:
                raise ValueError(f"Query ID có nội dung mâu thuẫn: {qid}")
            duplicate_queries += 1
        else:
            queries[qid] = {"id": qid, "text": text}

    test_ids, test_texts = set(), set()
    for row in test_queries:
        qid = required_string(row, "id")
        if qid in test_ids:
            raise ValueError(f"Query test trùng ID: {qid}")
        test_ids.add(qid)
        test_texts.add(text_key(required_string(row, "text")))
    if not test_ids:
        raise ValueError("Tập test rỗng; không thể kiểm tra rò rỉ dữ liệu")

    qrels, pairs, train_ids = [], set(), set()
    for row in source_qrels:
        qid = required_string(row, "query-id")
        doc_id = required_string(row, "corpus-id")
        if row.get("score") != 1 or isinstance(row.get("score"), bool):
            raise ValueError("Qrels Zalo Legal phải có score = 1")
        if qid not in queries:
            raise ValueError(f"Qrels tham chiếu query không tồn tại: {qid}")
        if doc_id not in doc_ids:
            raise ValueError(f"Qrels tham chiếu tài liệu không tồn tại: {doc_id}")
        pair = (qid, doc_id)
        if pair in pairs:
            raise ValueError(f"Qrels trùng cặp query/tài liệu: {pair}")
        pairs.add(pair)
        train_ids.add(qid)
        qrels.append({"query-id": qid, "corpus-id": doc_id, "score": 1})

    id_overlap = train_ids & test_ids
    text_overlap = {
        qid for qid in train_ids - id_overlap
        if text_key(queries[qid]["text"]) in test_texts
    }
    keep = train_ids - id_overlap - text_overlap
    if not keep:
        raise ValueError("Không còn query train sau khi loại trùng test")
    train_queries = [row for qid, row in queries.items() if qid in keep]
    train_qrels = [row for row in qrels if row["query-id"] in keep]
    counts = Counter(row["query-id"] for row in train_qrels)
    report = {
        "source_query_rows": len(queries) + duplicate_queries,
        "source_unique_queries": len(queries),
        "duplicate_source_query_rows_removed": duplicate_queries,
        "source_train_queries": len(train_ids),
        "source_train_qrels": len(qrels),
        "excluded_test_id_overlap": sorted(id_overlap),
        "excluded_test_text_overlap": sorted(text_overlap),
        "excluded_qrels": len(qrels) - len(train_qrels),
        "train_queries": len(train_queries),
        "train_qrels": len(train_qrels),
        "relevant_documents_per_query": dict(sorted(Counter(counts.values()).items())),
        "test_queries": len(test_ids),
        "remaining_test_id_overlap": len(keep & test_ids),
        "remaining_test_text_overlap": sum(
            text_key(row["text"]) in test_texts for row in train_queries
        ),
    }
    return train_queries, train_qrels, report


def ensure_sources(source_dir, download):
    for name, expected in SOURCE_HASHES.items():
        path = source_dir / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Thiếu {path}; chạy lại với --download")
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"SHA-256 nguồn tải về không khớp: {name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        if sha256(path) != expected:
            raise ValueError(f"SHA-256 nguồn không khớp: {path}")


def prepare(data_dir, source_dir, download=False):
    ensure_sources(source_dir, download)
    corpus_path = data_dir / "corpus.jsonl"
    test_path = data_dir / "queries_clean.jsonl"
    doc_ids = set()
    for row in read_jsonl(corpus_path):
        doc_id = required_string(row, "id")
        if doc_id in doc_ids:
            raise ValueError(f"Corpus trùng ID: {doc_id}")
        doc_ids.add(doc_id)
    queries, qrels, report = prepare_rows(
        read_jsonl(source_dir / "queries.jsonl"),
        read_jsonl(source_dir / "qrels/train.jsonl"),
        read_jsonl(test_path),
        doc_ids,
    )
    # Hoàn tất kiểm tra trước khi ghi; chỉ ba file train dưới đây được tạo lại.
    queries_path = data_dir / "queries_train.jsonl"
    qrels_path = data_dir / "qrels_train.tsv"
    with queries_path.open("w", encoding="utf-8", newline="\n") as output:
        for row in queries:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    with qrels_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=["query-id", "corpus-id", "score"],
            delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(qrels)
    report.update({
        "dataset": DATASET,
        "revision": REVISION,
        "source_sha256": SOURCE_HASHES,
        "local_sha256": {
            path.name: sha256(path)
            for path in (corpus_path, test_path, queries_path, qrels_path)
        },
        "normalization_for_overlap": "Unicode NFC + casefold + collapse whitespace",
    })
    report_path = data_dir / "train_manifest.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/zalo_legal")
    parser.add_argument("--source-dir", type=Path, help="Cache JSONL gốc; mặc định <data-dir>/raw")
    parser.add_argument("--download", action="store_true", help="Tải file nguồn còn thiếu")
    args = parser.parse_args()
    report = prepare(args.data_dir, args.source_dir or args.data_dir / "raw", args.download)
    print(json.dumps({key: report[key] for key in (
        "train_queries", "train_qrels", "remaining_test_id_overlap", "remaining_test_text_overlap"
    )}))


if __name__ == "__main__":
    main()
