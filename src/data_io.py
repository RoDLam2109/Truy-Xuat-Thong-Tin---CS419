"""Đọc dữ liệu canonical; ID luôn là chuỗi, không chuẩn hóa ID."""

import csv
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_records(path, corpus=False):
    seen = set()
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: JSON không hợp lệ") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: cần object JSON")
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError(f"{path}:{line_number}: ID phải là chuỗi không rỗng")
            if identifier in seen:
                raise ValueError(f"{path}:{line_number}: ID trùng: {identifier}")
            seen.add(identifier)
            if not isinstance(row.get("text"), str):
                raise ValueError(f"{path}:{line_number}: text phải là chuỗi")
            if corpus and not isinstance(row.get("title", ""), str):
                raise ValueError(f"{path}:{line_number}: title phải là chuỗi")
            yield row
    if not seen:
        raise ValueError(f"{path}: dữ liệu rỗng")


def read_qrels(path, query_ids, doc_ids):
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["query-id", "corpus-id", "score"]:
            raise ValueError("Qrels cần header query-id, corpus-id, score phân cách bằng tab")
        for row in reader:
            qid, did = row["query-id"], row["corpus-id"]
            if qid not in query_ids or did not in doc_ids:
                raise ValueError(f"Qrels tham chiếu ID không tồn tại: {qid}, {did}")
            if (qid, did) in seen:
                raise ValueError(f"Qrels trùng cặp query/document: {qid}, {did}")
            try:
                score = float(row["score"])
            except (TypeError, ValueError) as error:
                raise ValueError("Score qrels không hợp lệ") from error
            if not math.isfinite(score) or score < 0:
                raise ValueError("Score qrels phải hữu hạn và >= 0")
            seen.add((qid, did))
            rows.append((qid, did, score))
    if not rows:
        raise ValueError("Qrels rỗng")
    relevant_ids = {qid for qid, _, score in rows if score > 0}
    if relevant_ids != set(query_ids):
        raise ValueError("Mỗi query cần ít nhất một nhãn liên quan dương")
    return rows
