"""Validate exported rankings and reuse the project's existing metric formulas."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from src.retrieval.evaluation import aggregate, query_metrics

METRICS = ("mrr@10", "ndcg@10", "recall@100")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_qrels(path, qids):
    result = {qid: {} for qid in qids}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["query-id", "corpus-id", "score"]:
            raise ValueError("Invalid qrels header")
        for row in reader:
            qid, doc_id = row["query-id"], row["corpus-id"]
            score = float(row["score"])
            if qid not in result or doc_id in result[qid]:
                raise ValueError("Unknown query or duplicate qrel")
            if not math.isfinite(score) or score <= 0:
                raise ValueError("Expected finite positive qrels")
            result[qid][doc_id] = score
    if any(not values for values in result.values()):
        raise ValueError("Every query needs positive qrels")
    return result


def evaluate_rankings(rankings, statuses, qids, qrels):
    """Explicit empty statuses count as zero; missing queries are errors.

    Each ranking follows {qid, doc_id, rank, score}; status rows follow
    {qid, status: ok|empty, retrieved}. Ties use document ID ascending.
    """
    qids = list(qids)
    if not qids or len(qids) != len(set(qids)) or set(qrels) != set(qids):
        raise ValueError("Invalid query IDs or qrels coverage")
    by_query = {qid: [] for qid in qids}
    status_by_id = {}
    for row in statuses:
        qid = row["qid"]
        if qid not in by_query or qid in status_by_id:
            raise ValueError("Unknown/duplicate query status")
        status_by_id[qid] = row
    if set(status_by_id) != set(qids):
        raise ValueError("Missing query status; cannot silently omit queries")
    for row in rankings:
        qid = row["qid"]
        if qid not in by_query:
            raise ValueError("Unknown query in ranking")
        if not isinstance(row["doc_id"], str) or not row["doc_id"]:
            raise ValueError("Invalid document ID")
        if type(row["rank"]) is not int:
            raise ValueError("Rank must be an integer")
        if not math.isfinite(row["score"]) or row["score"] <= 0:
            raise ValueError("Ranking score must be finite and positive")
        by_query[qid].append(row)
    per_query = []
    for qid in qids:
        rows = sorted(by_query[qid], key=lambda row: row["rank"])
        if [row["rank"] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError("Ranks must be consecutive from one")
        if len(rows) > 100:
            raise ValueError("Run schema permits at most 100 documents per query")
        if rows != sorted(rows, key=lambda row: (-row["score"], row["doc_id"])):
            raise ValueError("Score order or document-ID tie order is invalid")
        status = status_by_id[qid]
        expected = "ok" if rows else "empty"
        if (status["status"] != expected or type(status["retrieved"]) is not int
                or status["retrieved"] != len(rows)):
            raise ValueError("Query status does not match ranking")
        scores = query_metrics([row["doc_id"] for row in rows], qrels[qid])
        per_query.append(dict(qid=qid, status=expected, retrieved=len(rows), **scores))
    return per_query, aggregate(per_query)


def evaluate_files(rankings_path, statuses_path, queries_path, qrels_path):
    qids = [row["id"] for row in read_jsonl(queries_path)]
    qrels = read_qrels(qrels_path, qids)
    return evaluate_rankings(read_jsonl(rankings_path), read_jsonl(statuses_path),
                             qids, qrels)
