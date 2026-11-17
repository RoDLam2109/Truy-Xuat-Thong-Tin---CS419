"""Compare VSM, BM25, and LSI on paired clean/noisy Drive queries.

No qrels or queries are used to fit IDF, vocabulary, BM25 statistics or SVD.
All hyperparameters are fixed up front; this is a held-out benchmark, not tuning.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

from scipy.sparse import load_npz
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_io import read_qrels, read_records, sha256
from src.terms import TermIndex
from src.retrieval.models import BM25Retriever, LSIRetriever, SparseDotRetriever
from src.retrieval.evaluation import query_metrics, aggregate, degradation

CONDITIONS = (
    "clean", "no_diacritics", "telex_typo", "phonetic", "abbrev",
    "mixed_low", "mixed_medium", "mixed_high",
)
METRICS = ("mrr@10", "ndcg@10", "recall@100")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/zalo_legal")
    p.add_argument("--index-dir", type=Path, default=ROOT / "outputs/terms/drive_word_clean")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/retrieval/drive_word_v1")
    p.add_argument("--models", nargs="+", choices=("vsm", "bm25", "lsi"),
                   default=("vsm", "bm25", "lsi"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bm25-k1", type=float, default=1.2)
    p.add_argument("--bm25-b", type=float, default=0.75)
    return p.parse_args()


def load_data(data_dir, index):
    paths = {"clean": data_dir / "queries_clean.jsonl"}
    paths.update({name: data_dir / "noise" / f"queries_{name}.jsonl"
                  for name in CONDITIONS if name != "clean"})
    query_sets = {name: list(read_records(path)) for name, path in paths.items()}
    clean_ids = [x["id"] for x in query_sets["clean"]]
    for name, rows in query_sets.items():
        if [x["id"] for x in rows] != clean_ids:
            raise ValueError(f"{name}: IDs/order differ from clean; aborting")
    judged_rows = read_qrels(data_dir / "qrels_test.tsv", set(clean_ids), set(index.doc_ids))
    judged = {qid: {} for qid in clean_ids}
    for qid, did, score in judged_rows:
        judged[qid][did] = score
    return paths, query_sets, clean_ids, judged


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >=1")
    if args.output_dir.exists():
        raise ValueError(f"Output exists, choose a new run ID: {args.output_dir}")
    index = TermIndex(args.index_dir)
    paths, query_sets, ids, qrels = load_data(args.data_dir, index)
    print(f"Validated {len(ids)} queries, {len(index.doc_ids)} documents, "
          f"{sum(map(len, qrels.values()))} qrels.", flush=True)
    # Every condition is transformed by the same corpus-fitted vocabulary and IDF.
    matrices = {}
    for name in CONDITIONS:
        texts = [row["text"] for row in query_sets[name]]
        matrices[name] = index.transform(texts)
        print(f"Transformed {name}: {matrices[name][0].shape}, "
              f"empty={int(np.count_nonzero(np.diff(matrices[name][0].indptr) == 0))}",
              flush=True)
    corpus_counts = load_npz(args.index_dir / "counts.npz")
    corpus_tfidf = load_npz(args.index_dir / "tfidf.npz")
    lengths = np.load(args.index_dir / "doc_lengths.npy")
    if corpus_counts.shape != corpus_tfidf.shape or corpus_counts.shape[0] != len(index.doc_ids):
        raise ValueError("Corpus index shapes incompatible")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "runs").mkdir()
    settings = {
        "source": "user-provided Google Drive, folder 18fC4b6ve3zxCzdqMIWADmpmja1yP2XoD",
        "evaluation_split": "788 paired clean/noisy test queries",
        "n_queries": len(ids), "n_documents": len(index.doc_ids),
        "n_qrels": sum(map(len, qrels.values())),
        "models": args.models, "seed": args.seed, "svd_dimensions": args.svd_dim,
        "bm25_k1": args.bm25_k1, "bm25_b": args.bm25_b, "top_k": 100,
        "batch_size": args.batch_size,
        "index_manifest_sha256": sha256(args.index_dir / "manifest.json"),
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
        "qrels_sha256": sha256(args.data_dir / "qrels_test.tsv"),
        "hyperparameter_selection": "fixed before test; no test-set tuning",
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    changed = {name: sum(a["text"] != b["text"] for a, b in
                zip(query_sets["clean"], query_sets[name])) for name in CONDITIONS}
    all_summary, all_perquery = [], []
    for model_name in args.models:
        print(f"Fitting {model_name}...", flush=True)
        t0 = time.perf_counter()
        if model_name == "vsm":
            model = SparseDotRetriever(corpus_tfidf, index.doc_ids, top_k=100)
            query_matrix_pos = 1
        elif model_name == "bm25":
            model = BM25Retriever(corpus_counts, lengths, index.doc_ids,
                                  k1=args.bm25_k1, b=args.bm25_b, top_k=100)
            query_matrix_pos = 0
        else:
            model = LSIRetriever(corpus_tfidf, index.doc_ids, dimensions=args.svd_dim,
                                 seed=args.seed, top_k=100)
            query_matrix_pos = 1
        fit_seconds = round(time.perf_counter() - t0, 3)
        print(f"{model_name} fit complete in {fit_seconds}s.", flush=True)
        baseline = None
        for condition in CONDITIONS:
            t0 = time.perf_counter()
            query_matrix = matrices[condition][query_matrix_pos]
            rankings = list(model.rank(query_matrix, batch_size=args.batch_size))
            if len(rankings) != len(ids):
                raise AssertionError("Missing query rankings")
            per_query = []
            run_path = args.output_dir / "runs" / f"{model_name}_{condition}.jsonl"
            with run_path.open("w", encoding="utf-8", newline="\n") as run:
                for qid, ranking in zip(ids, rankings):
                    doc_ids = [index.doc_ids[col] for col, _ in ranking]
                    scores = [score for _, score in ranking]
                    metrics = query_metrics(doc_ids, qrels[qid])
                    per_query.append(metrics)
                    all_perquery.append({"model": model_name, "condition": condition,
                                         "qid": qid, "retrieved": len(ranking), **metrics})
                    for rank, (did, score) in enumerate(zip(doc_ids, scores), 1):
                        run.write(json.dumps({"qid": qid, "doc_id": did, "rank": rank,
                                              "score": score}, ensure_ascii=False) + "\n")
            summary = aggregate(per_query)
            if condition == "clean":
                baseline = summary
            row = {"model": model_name, "term": "word", "condition": condition,
                   "queries": len(ids), "changed_queries": changed[condition],
                   "fit_seconds": fit_seconds, "retrieval_seconds": round(time.perf_counter() - t0, 3),
                   **summary, **degradation(baseline, summary)}
            all_summary.append(row)
            # Preserve verified partial results if a later heavy model fails.
            (args.output_dir / "progress.json").write_text(
                json.dumps(all_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"{model_name:<5} {condition:<15} MRR={summary['mrr@10']:.4f} "
                  f"nDCG={summary['ndcg@10']:.4f} Recall={summary['recall@100']:.4f} "
                  f"drop={row['ndcg@10_drop_pct']:.1f}%", flush=True)
    cols = ["term", "model", "condition", "queries", "changed_queries", *METRICS,
            *(m + "_drop_abs" for m in METRICS),
            *(m + "_drop_pct" for m in METRICS), "fit_seconds", "retrieval_seconds"]
    write_csv(args.output_dir / "summary.csv", all_summary, cols)
    write_csv(args.output_dir / "per_query.csv", all_perquery,
              ["model", "condition", "qid", "retrieved", *METRICS])
    lines = [
        "# Model evaluation on Drive queries (no test-set tuning)", "",
        f"Corpus: {len(index.doc_ids)} | Queries: {len(ids)} | Qrels: {settings['n_qrels']}", "",
        "| Model | Condition | MRR@10 | nDCG@10 | Recall@100 | nDCG drop % |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in all_summary:
        lines.append("| {model} | {condition} | {mrr@10:.4f} | {ndcg@10:.4f} | "
                     "{recall@100:.4f} | {ndcg@10_drop_pct:.1f}% |".format(**row))
    lines += ["", "All metrics average over all 788 queries. The same full corpus and "
              "qrels were used for every condition. Hyperparameters fixed before evaluation."]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Results saved to " + str(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
