"""Build word or character n-gram sparse indexes from processed corpus."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_terms import read_rows
from src.retrieval.feature_matrix import load_vocabulary, to_sparse, fit_corpus

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(processed_dir, output_dir, branch, max_docs=None):
    processed_dir, output_dir = Path(processed_dir), Path(output_dir)
    if branch not in {"word", "char_ngram"}:
        raise ValueError("Branch must be 'word' or 'char_ngram'")
    if max_docs is not None and max_docs < 2:
        raise ValueError("max_docs must be at least two")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite index: {output_dir}")
    vocab_path = processed_dir / f"vocabulary_{branch}.json"
    vocab = load_vocabulary(vocab_path)
    config = json.loads((processed_dir / "config_used.json").read_text(encoding="utf-8"))
    summary = json.loads((processed_dir / "summary.json").read_text(encoding="utf-8"))
    corpus_path = processed_dir / "corpus.jsonl"
    ids = []
    def corpus_rows():
        source = read_rows(corpus_path)
        if max_docs is not None:
            source = itertools.islice(source, max_docs)
        for row in source:
            ids.append(row["id"])
            yield row
    matrix = to_sparse(corpus_rows(), vocab, branch, config["stopwords"], config["ngram_range"])
    if matrix.shape[0] != len(ids) or len(ids) != len(set(ids)):
        raise ValueError("Duplicate/misaligned corpus document IDs")
    if max_docs is None and len(ids) != summary["rows"]["corpus.jsonl"]:
        raise ValueError("Processed corpus count differs from summary")
    lengths = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    if not np.any(lengths):
        raise ValueError("Empty index: no documents contain selected terms")
    tfidf, idf = fit_corpus(matrix)
    output_dir.mkdir(parents=True)
    sparse.save_npz(output_dir / "counts.npz", matrix)
    sparse.save_npz(output_dir / "tfidf.npz", tfidf)
    np.save(output_dir / "doc_lengths.npy", lengths)
    np.save(output_dir / "idf.npy", idf)
    (output_dir / "doc_ids.json").write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    (output_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    meta = {
        "branch": branch, "n_documents": len(ids), "n_features": len(vocab),
        "nnz": int(matrix.nnz), "full_corpus": max_docs is None,
        "idf_fit": "corpus only; sklearn TfidfTransformer, smooth_idf=True",
        "vocabulary_fit": "corpus DF only via scripts/prepare_terms.py",
        "processed_corpus_sha256": sha256(corpus_path),
        "vocabulary_sha256": sha256(vocab_path),
        "raw_corpus_sha256": summary["source_sha256"]["corpus.jsonl"],
        "term_config": config,
    }
    (output_dir / "index_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "outputs/processed")
    parser.add_argument("--branch", choices=["word", "char_ngram"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-docs", type=int, help="Smoke test ONLY; index is not full corpus")
    args = parser.parse_args()
    print(json.dumps(build(args.processed_dir, args.output_dir, args.branch,
                           args.max_docs), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
