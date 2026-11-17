"""Evaluation of full-corpus top-k ranked retrieval with fixed test qrels."""
from __future__ import annotations

import math
from statistics import mean


def query_metrics(ranked_doc_ids, relevant_scores):
    """Compute IR metrics over all judged positive documents, not just top 10."""
    positive = {doc_id: score for doc_id, score in relevant_scores.items() if score > 0}
    if not positive:
        raise ValueError("Every query must have positive qrels")
    ranking = list(ranked_doc_ids)
    if len(ranking) != len(set(ranking)):
        raise ValueError("Repeated document in ranking")
    mrr = 0.0
    dcg = 0.0
    for position, doc_id in enumerate(ranking[:10], 1):
        score = positive.get(doc_id, 0.0)
        if score:
            if mrr == 0.0:
                mrr = 1.0 / position
            dcg += (2.0 ** score - 1.0) / math.log2(position + 1)
    ideal = sorted(positive.values(), reverse=True)[:10]
    ideal_dcg = sum((2.0 ** score - 1.0) / math.log2(i + 2)
                    for i, score in enumerate(ideal))
    recall = sum(doc_id in positive for doc_id in ranking[:100]) / len(positive)
    return {"mrr@10": mrr, "ndcg@10": dcg / ideal_dcg, "recall@100": recall}


def aggregate(per_query):
    if not per_query:
        raise ValueError("No query results")
    return {metric: mean(row[metric] for row in per_query) for metric in
            ("mrr@10", "ndcg@10", "recall@100")}


def degradation(clean, noisy):
    result = {}
    for metric in ("mrr@10", "ndcg@10", "recall@100"):
        drop = clean[metric] - noisy[metric]
        result[metric + "_drop_abs"] = drop
        result[metric + "_drop_pct"] = (100 * drop / clean[metric]
                                         if clean[metric] else None)
    return result
