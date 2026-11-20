"""Sparse term-count and TF-IDF matrices from corpus-derived vocabulary.

The same vocabulary mapping and corpus IDF are reused for every query variant.
"""
from __future__ import annotations

from array import array
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

from scripts.prepare_terms import char_terms, normalize as clean_text, word_terms


def load_vocabulary(path):
    df = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(df, dict) or not df:
        raise ValueError(f"Empty/invalid corpus vocabulary: {path}")
    return {term: i for i, term in enumerate(sorted(df))}


def query_terms(row, branch, vocab, stopwords, ngram_range=(3, 5)):
    """Word and char branches use the same normalization as prepare_terms."""
    if branch == "word":
        if "word_terms" in row:
            tokens = row["word_terms"]
        else:
            tokens = word_terms(clean_text(row["text"]), set(stopwords))
    elif branch == "char_ngram":
        tokens = char_terms(clean_text(row["text"]), ngram_range)
    else:
        raise ValueError(f"Unknown branch {branch}")
    return (vocab[t] for t in tokens if t in vocab)


def to_sparse(rows, vocab, branch, stopwords=(), ngram_range=(3, 5)):
    """Build CSR incrementally without Python lists of COO triples."""
    values, indices, indptr = array("f"), array("i"), array("q", [0])
    count = 0
    for row in rows:
        frequencies = Counter(query_terms(row, branch, vocab, stopwords, ngram_range))
        for j, frequency in sorted(frequencies.items()):
            indices.append(j)
            values.append(float(frequency))
        indptr.append(len(indices))
        count += 1
    return sparse.csr_matrix((
        np.frombuffer(values, dtype=np.float32),
        np.frombuffer(indices, dtype=np.int32),
        np.frombuffer(indptr, dtype=np.int64),
    ), shape=(count, len(vocab)), dtype=np.float32)


def tfidf_queries(counts, idf):
    """Match sklearn's TF-IDF: raw TF, smooth IDF from CORPUS, L2 norm."""
    counts = sparse.csr_matrix(counts, dtype=np.float32)
    idf = np.asarray(idf, dtype=np.float32)
    if counts.shape[1] != idf.shape[0]:
        raise ValueError("IDF and query vocabulary do not match")
    return normalize(counts.multiply(idf).tocsr(), norm="l2")


def fit_corpus(counts):
    counts = sparse.csr_matrix(counts, dtype=np.float32)
    transformer = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
    tfidf = transformer.fit_transform(counts).astype(np.float32)
    return tfidf, transformer.idf_.astype(np.float32)


def load_index(path):
    """Load a built index, preserving document row and vocabulary ordering."""
    path = Path(path)
    meta = json.loads((path / "index_meta.json").read_text(encoding="utf-8"))
    doc_ids = json.loads((path / "doc_ids.json").read_text(encoding="utf-8"))
    vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
    counts = sparse.load_npz(path / "counts.npz").tocsr()
    tfidf = sparse.load_npz(path / "tfidf.npz").tocsr()
    lengths = np.load(path / "doc_lengths.npy")
    idf = np.load(path / "idf.npy")
    if counts.shape != tfidf.shape or counts.shape != (len(doc_ids), len(vocab)):
        raise ValueError("Inconsistent stored matrix/doc/vocab dimensions")
    if idf.shape != (len(vocab),) or lengths.shape != (len(doc_ids),):
        raise ValueError("Mismatched document lengths or idf dimensions")
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("Duplicate document IDs in index")
    return meta, doc_ids, vocab, counts, tfidf, lengths, idf
