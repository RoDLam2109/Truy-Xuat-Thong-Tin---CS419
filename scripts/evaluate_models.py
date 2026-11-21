"""Evaluate VSM, BM25 and LSI with word/character n-gram representations.

Validation is the default split; test evaluation requires explicit confirmation.
Vocabulary, IDF and SVD are fitted on corpus data, not relevance labels.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_retrieval_index import sha256
from scripts.generate_noise import read_queries
from src.retrieval.feature_matrix import load_index, to_sparse, tfidf_queries
from src.retrieval.models import SparseDotRetriever, BM25Retriever, LSIRetriever
from src.retrieval.evaluation import query_metrics, aggregate, degradation

CONDITIONS = (
    "clean", "no_diacritics", "telex_typo", "phonetic", "abbrev",
    "mixed_low", "mixed_medium", "mixed_high",
)
METRICS = ("mrr@10", "ndcg@10", "recall@100")


def qrels_by_query(path, ids, doc_ids, allow_subset=False):
    relevant = {qid: {} for qid in ids}
    seen = set()
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != ["query-id", "corpus-id", "score"]:
            raise ValueError(f"Invalid qrels columns: {path}")
        for row in reader:
            qid, did = row["query-id"], row["corpus-id"]
            pair = (qid, did)
            if qid not in relevant or pair in seen:
                raise ValueError(f"Unknown/duplicate qrel: {pair}")
            if not allow_subset and did not in doc_ids:
                raise ValueError(f"Relevant document not found in corpus: {did}")
            value = int(row["score"])
            if value < 1:
                raise ValueError("This dataset expects positive qrels")
            relevant[qid][did] = value
            seen.add(pair)
    if any(not rel for rel in relevant.values()):
        raise ValueError("Every query needs at least one positive document")
    return relevant


def data_paths(data_dir, split_dir, split):
    if split == "val":
        base, qrels = split_dir / "val", split_dir / "val/qrels_val.tsv"
    elif split == "test":
        base, qrels = data_dir, data_dir / "qrels_test.tsv"
    else:
        raise ValueError(f"Invalid split: {split}")
    paths = {"clean": base / "queries_clean.jsonl"}
    paths.update({name: base / "noise" / f"queries_{name}.jsonl"
                  for name in CONDITIONS if name != "clean"})
    return paths, qrels


def load_evaluation(data_dir, split_dir, split, doc_ids, allow_subset=False):
    paths, qrels_file = data_paths(data_dir, split_dir, split)
    query_sets = {name: read_queries(path) for name, path in paths.items()}
    clean_ids = [row["id"] for row in query_sets["clean"]]
    for name, rows in query_sets.items():
        if [row["id"] for row in rows] != clean_ids:
            raise ValueError(f"{name}: query IDs/order differ from clean")
    judged = qrels_by_query(qrels_file, clean_ids, set(doc_ids), allow_subset)
    return paths, qrels_file, query_sets, clean_ids, judged


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(index_dir, data_dir, split_dir, split, output_dir, models=("vsm", "bm25", "lsi"),
             batch_size=16, svd_dim=64, seed=42, k1=1.2, b=0.75,
             confirm_test=False, allow_subset_smoke=False):
    index_dir, data_dir = Path(index_dir), Path(data_dir)
    split_dir, output_dir = Path(split_dir), Path(output_dir)
    if split == "test" and not confirm_test:
        raise ValueError("Final test requires --confirm-test-use. Tune ONLY on val.")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; select a unique run directory: {output_dir}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    meta, doc_ids, vocab, counts, tfidf, lengths, idf = load_index(index_dir)
    if not meta["full_corpus"] and not allow_subset_smoke:
        raise ValueError("Subset corpus index is smoke-only; full evaluation rejected")
    if meta["raw_corpus_sha256"] != sha256(data_dir / "corpus.jsonl"):
        raise ValueError("Index built from a different corpus")
    paths, qrels_file, query_sets, ids, judged = load_evaluation(
        data_dir, split_dir, split, doc_ids, allow_subset_smoke)
    branch = meta["branch"]
    if branch not in {"word", "char_ngram"}:
        raise ValueError("Unknown term branch in index")
    output_dir.mkdir(parents=True)
    configuration = {
        "dataset": "GreenNode Lam v1",
        "split": split, "branch": branch,
        "n_documents": len(doc_ids), "n_queries": len(ids),
        "n_qrels": sum(map(len, judged.values())),
        "full_corpus": meta["full_corpus"],
        "allow_subset_smoke": allow_subset_smoke,
        "model_names": list(models), "bm25_k1": k1, "bm25_b": b,
        "svd_dim": svd_dim, "seed": seed, "batch_size": batch_size,
        "index_meta_sha256": sha256(index_dir / "index_meta.json"),
        "qrels_sha256": sha256(qrels_file),
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
        "selection_rule": "Tune on validation; test held out. Corpus-only TF-IDF, IDF, SVD.",
    }
    (output_dir / "config.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    per_query, summaries = [], []
    for model_name in models:
        start = time.perf_counter()
        if model_name == "vsm":
            retriever = SparseDotRetriever(tfidf, doc_ids, top_k=100)
        elif model_name == "bm25":
            retriever = BM25Retriever(counts, lengths, doc_ids, k1=k1, b=b, top_k=100)
        elif model_name == "lsi":
            retriever = LSIRetriever(tfidf, doc_ids, dimensions=svd_dim, seed=seed, top_k=100)
        else:
            raise ValueError(f"Unknown model {model_name}")
        fit_time = time.perf_counter() - start
        clean_baseline = None
        for name in CONDITIONS:
            started = time.perf_counter()
            qcounts = to_sparse(query_sets[name], vocab, branch,
                                meta["term_config"]["stopwords"],
                                meta["term_config"]["ngram_range"])
            qmatrix = qcounts if model_name == "bm25" else tfidf_queries(qcounts, idf)
            empty = int(np.count_nonzero(np.diff(qcounts.indptr) == 0))
            ranking_rows = retriever.rank(qmatrix, batch_size=batch_size)
            metrics = []
            for qid, ranking in zip(ids, ranking_rows):
                ranked = [doc_ids[column] for column, _ in ranking]
                scores = query_metrics(ranked, judged[qid])
                metrics.append(scores)
                per_query.append(dict(model=model_name, branch=branch,
                                      condition=name, qid=qid, retrieved=len(ranking),
                                      **scores))
            if len(metrics) != len(ids):
                raise AssertionError("Ranking count differs from query count")
            agg = aggregate(metrics)
            if name == "clean":
                clean_baseline = agg
            row = dict(branch=branch, model=model_name, split=split, condition=name,
                       queries=len(ids),
                       changed_queries=sum(x["text"] != y["text"] for x, y in
                                           zip(query_sets["clean"], query_sets[name])),
                       empty_queries=empty,
                       fit_seconds=round(fit_time, 3),
                       retrieval_seconds=round(time.perf_counter() - started, 3),
                       **agg, **degradation(clean_baseline, agg))
            summaries.append(row)
            (output_dir / "progress.json").write_text(
                json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"{branch} {model_name} {name}: nDCG@10={agg['ndcg@10']:.4f}, "
                  f"MRR@10={agg['mrr@10']:.4f}, Recall@100={agg['recall@100']:.4f}",
                  flush=True)
    fields = ["split", "branch", "model", "condition", "queries", "changed_queries",
              "empty_queries", *METRICS,
              *(m + "_drop_abs" for m in METRICS),
              *(m + "_drop_pct" for m in METRICS), "fit_seconds", "retrieval_seconds"]
    write_csv(output_dir / "summary.csv", summaries, fields)
    write_csv(output_dir / "per_query.csv", per_query,
              ["branch", "model", "condition", "qid", "retrieved", *METRICS])
    lines = [
        f"# Retrieval evaluation — {split} / {branch}", "",
        f"Corpus: {len(doc_ids)} docs; queries: {len(ids)}; qrels: {configuration['n_qrels']}.",
        f"Full corpus: {meta['full_corpus']}. No query labels used for model fitting.", "",
        "| Model | Condition | MRR@10 | nDCG@10 | Recall@100 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(f"| {row['model']} | {row['condition']} | {row['mrr@10']:.4f} | "
                     f"{row['ndcg@10']:.4f} | {row['recall@100']:.4f} |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/zalo_legal")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "outputs/splits/seed42")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=["vsm", "bm25", "lsi"],
                        default=["vsm", "bm25", "lsi"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--svd-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bm25-k1", type=float, default=1.2)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--confirm-test-use", action="store_true")
    parser.add_argument("--allow-subset-smoke", action="store_true")
    args = parser.parse_args()
    evaluate(args.index_dir, args.data_dir, args.split_dir, args.split, args.output_dir,
             args.models, args.batch_size, args.svd_dim, args.seed, args.bm25_k1,
             args.bm25_b, args.confirm_test_use, args.allow_subset_smoke)


if __name__ == "__main__":
    main()
