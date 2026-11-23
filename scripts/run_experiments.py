"""Run Lam's 2 representations x 3 models x 8 query conditions.

Uses existing term/index/model/metric APIs without changing their behavior.
Each condition is committed by a completion marker only after export and audit.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Limit BLAS oversubscription for the large SVD while preserving its parameters.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "4")

import joblib
import numpy as np

from scripts.build_retrieval_index import sha256
from scripts.evaluate_models import CONDITIONS, data_paths, load_evaluation
from src.evaluation.runs import evaluate_files, write_csv, write_json
from src.retrieval.evaluation import degradation
from src.retrieval.feature_matrix import load_index, to_sparse, tfidf_queries
from src.retrieval.models import BM25Retriever, LSIRetriever, SparseDotRetriever

PROTOCOL = {"bm25_k1": 1.2, "bm25_b": 0.75, "svd_dim": 64,
            "seed": 42, "top_k": 100}
SELECTION_RULE = "mean noisy nDCG@10 across all seven conditions; tie: clean nDCG@10, branch, model"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def completed_run(directory, expected_config):
    marker = directory / "complete.json"
    if not marker.is_file():
        raise ValueError(f"Incomplete run at {directory}; move it aside before resuming")
    done = json.loads(marker.read_text(encoding="utf-8"))
    if done["config"] != expected_config:
        raise ValueError(f"Run configuration changed at {directory}")
    for name, digest in done["sha256"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Run artifact changed: {directory / name}")
    return json.loads((directory / "summary.json").read_text(encoding="utf-8"))


def run(index_root, data_dir, split_dir, output_dir, split="val", resume=False,
        branches=("word", "char_ngram"), models=("vsm", "bm25", "lsi"),
        batch_size=16, protocol=None, selection=None, confirm_test=False):
    protocol = dict(PROTOCOL if protocol is None else protocol)
    index_root, data_dir, split_dir, output_dir = map(
        Path, (index_root, data_dir, split_dir, output_dir))
    branches, models = list(branches), list(models)
    if (not branches or len(set(branches)) != len(branches)
            or not set(branches) <= {"word", "char_ngram"}
            or not models or len(set(models)) != len(models)
            or not set(models) <= {"vsm", "bm25", "lsi"} or batch_size < 1):
        raise ValueError("Invalid branches, models or batch size")
    if split == "test":
        if not confirm_test or selection is None:
            raise ValueError("Test requires --confirm-test-use and a frozen validation --selection")
        selected = json.loads(Path(selection).read_text(encoding="utf-8"))
        if (selected["split"] != "val" or selected["completed_runs"] != 48
                or selected["protocol"] != protocol):
            raise ValueError("Test requires complete validation and unchanged model parameters")
    paths, qrels_path = data_paths(data_dir, split_dir, split)
    config = {
        "split": split, "branches": branches, "models": models,
        "conditions": list(CONDITIONS), "protocol": protocol, "batch_size": batch_size,
        "selection_rule": SELECTION_RULE,
        "queries_sha256": {name: sha256(path) for name, path in paths.items()},
        "qrels_sha256": sha256(qrels_path),
        "raw_corpus_sha256": sha256(data_dir / "corpus.jsonl"),
        "indexes": {branch: sha256(index_root / branch / "index_meta.json") for branch in branches},
        "selection_sha256": sha256(selection) if selection is not None else None,
        "code_sha256": {name: sha256(ROOT / name) for name in (
            "scripts/run_experiments.py", "src/evaluation/runs.py", "src/retrieval/models.py",
            "src/retrieval/feature_matrix.py", "src/retrieval/evaluation.py",
            "scripts/prepare_terms.py", "scripts/evaluate_models.py")},
        "versions": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "pyvi", "joblib")},
        "python": platform.python_version(),
    }
    if split == "test" and (selected["indexes"] != config["indexes"]
                             or selected["raw_corpus_sha256"] != config["raw_corpus_sha256"]):
        raise ValueError("Test indexes/corpus differ from frozen validation")
    if output_dir.exists():
        if not resume:
            raise FileExistsError("Output exists; use a new directory or --resume")
        previous = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        if previous["config"] != config:
            raise ValueError("Resume rejected: source/configuration changed")
    else:
        output_dir.mkdir(parents=True)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True, check=False).stdout.strip()
        write_json(output_dir / "manifest.json", {"config": config, "git_revision": revision,
                   "expected_runs": len(branches) * len(models) * len(CONDITIONS)})
    all_summaries = []
    for branch in branches:
        pending = any(not (output_dir / branch / model / name / "complete.json").is_file()
                      for model in models for name in CONDITIONS)
        if pending:
            meta, doc_ids, vocab, counts, tfidf, lengths, idf = load_index(index_root / branch)
            if not meta["full_corpus"] or meta["raw_corpus_sha256"] != config["raw_corpus_sha256"]:
                raise ValueError("A full index of the canonical corpus is required")
            _, _, query_sets, ids, _ = load_evaluation(data_dir, split_dir, split, doc_ids)
            query_counts = {name: to_sparse(query_sets[name], vocab, branch,
                            meta["term_config"]["stopwords"], meta["term_config"]["ngram_range"])
                            for name in CONDITIONS}
        for model_name in models:
            model_dir = output_dir / branch / model_name
            missing = any(not (model_dir / name / "complete.json").is_file() for name in CONDITIONS)
            retriever = None
            fit_time = 0.0
            if missing:
                started = time.perf_counter()
                if model_name == "vsm":
                    retriever = SparseDotRetriever(tfidf, doc_ids, top_k=protocol["top_k"])
                elif model_name == "bm25":
                    retriever = BM25Retriever(counts, lengths, doc_ids, k1=protocol["bm25_k1"],
                                               b=protocol["bm25_b"], top_k=protocol["top_k"])
                else:
                    retriever = LSIRetriever(tfidf, doc_ids, dimensions=protocol["svd_dim"],
                                              seed=protocol["seed"], top_k=protocol["top_k"])
                fit_time = time.perf_counter() - started
                # The identical fitted instance serves every clean/noisy query set.
                model_dir.mkdir(parents=True, exist_ok=True)
                model_file = model_dir / "model.joblib"
                if model_file.exists():
                    # Refit after interruption must reproduce the same serialized model.
                    temporary = model_dir / "model.resume.joblib"
                    joblib.dump(retriever, temporary, compress=0)
                    if sha256(temporary) != sha256(model_file):
                        raise ValueError("Refitted model differs from stored model")
                    temporary.unlink()
                else:
                    joblib.dump(retriever, model_file, compress=0)
            clean = None
            for name in CONDITIONS:
                directory = model_dir / name
                run_config = dict(experiment=fingerprint(config), branch=branch,
                                  model=model_name, condition=name)
                if directory.exists():
                    row = completed_run(directory, run_config)
                else:
                    directory.mkdir(parents=True)
                    started = time.perf_counter()
                    qcounts = query_counts[name]
                    qmatrix = qcounts if model_name == "bm25" else tfidf_queries(qcounts, idf)
                    with (directory / "rankings.jsonl").open("w", encoding="utf-8") as ranks, \
                         (directory / "queries.jsonl").open("w", encoding="utf-8") as statuses:
                        iterator = iter(retriever.rank(qmatrix, batch_size=batch_size))
                        for qid in ids:
                            try:
                                ranking = next(iterator)
                            except StopIteration as error:
                                raise ValueError("Model omitted queries") from error
                            statuses.write(json.dumps(dict(qid=qid, status="ok" if ranking else "empty",
                                                           retrieved=len(ranking))) + "\n")
                            for position, (column, score) in enumerate(ranking, 1):
                                ranks.write(json.dumps(dict(qid=qid, doc_id=doc_ids[column],
                                                            rank=position, score=score)) + "\n")
                        if next(iterator, None) is not None:
                            raise ValueError("Model returned extra queries")
                    per_query, metrics = evaluate_files(directory / "rankings.jsonl",
                        directory / "queries.jsonl", paths[name], qrels_path)
                    if name == "clean":
                        clean = metrics
                    row = dict(split=split, branch=branch, model=model_name, condition=name,
                        queries=len(ids), empty_queries=sum(r["status"] == "empty" for r in per_query),
                        changed_queries=sum(a["text"] != b["text"] for a, b in
                                            zip(query_sets["clean"], query_sets[name])),
                        fit_seconds=round(fit_time, 3),
                        retrieval_and_evaluation_seconds=round(time.perf_counter() - started, 3),
                        **metrics, **degradation(clean, metrics))
                    write_csv(directory / "per_query.csv", per_query)
                    write_json(directory / "summary.json", row)
                    files = ("rankings.jsonl", "queries.jsonl", "per_query.csv", "summary.json")
                    write_json(directory / "complete.json", dict(config=run_config,
                               sha256={file: sha256(directory / file) for file in files}))
                if name == "clean":
                    clean = row
                all_summaries.append(row)
                write_csv(output_dir / "summary.csv", all_summaries)
                print(f"{split} {branch} {model_name} {name}: nDCG@10={row['ndcg@10']:.4f}", flush=True)
            del retriever
            gc.collect()
        if pending:
            del counts, tfidf, lengths, idf, query_counts, query_sets
            gc.collect()
    write_json(output_dir / "complete.json", {"completed_runs": len(all_summaries),
               "summary_sha256": sha256(output_dir / "summary.csv")})
    return all_summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=ROOT / "outputs/lam/index")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/zalo_legal")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "outputs/lam/splits/seed42")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--branches", nargs="+", choices=["word", "char_ngram"], default=["word", "char_ngram"])
    parser.add_argument("--models", nargs="+", choices=["vsm", "bm25", "lsi"], default=["vsm", "bm25", "lsi"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--confirm-test-use", action="store_true")
    args = parser.parse_args()
    run(args.index_root, args.data_dir, args.split_dir, args.output_dir, args.split,
        args.resume, args.branches, args.models, args.batch_size,
        selection=args.selection, confirm_test=args.confirm_test_use)


if __name__ == "__main__":
    main()
