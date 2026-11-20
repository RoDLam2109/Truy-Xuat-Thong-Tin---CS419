"""Retrieval models using corpus-derived features.

Document and query matrices must share the stored vocabulary and weighting.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class RankedModel:
    def __init__(self, doc_ids, top_k=100):
        self.doc_ids = list(doc_ids)
        self.top_k = top_k
        # Deterministic tie rule: document ID ascending, regardless of corpus order.
        self.lex_rank = np.empty(len(doc_ids), dtype=np.int32)
        self.lex_rank[np.argsort(np.asarray(doc_ids, dtype=str), kind="stable")] = np.arange(len(doc_ids))

    def _topk(self, columns, values):
        columns = np.asarray(columns, dtype=np.int32)
        values = np.asarray(values)
        valid = np.isfinite(values) & (values > 0)
        columns, values = columns[valid], values[valid]
        k = min(self.top_k, len(values))
        if k == 0:
            return []
        if len(values) > k:
            cutoff = np.partition(values, len(values) - k)[len(values) - k]
            strictly_better = np.flatnonzero(values > cutoff)
            tied = np.flatnonzero(values == cutoff)
            keep_ties = tied[np.argsort(self.lex_rank[columns[tied]])[:k - len(strictly_better)]]
            keep = np.concatenate([strictly_better, keep_ties])
            columns, values = columns[keep], values[keep]
        order = np.lexsort((self.lex_rank[columns], -values))
        return [(int(columns[i]), float(values[i])) for i in order]

    def rank(self, queries, batch_size=16):
        """Return one top-k ranking per query. Queries are never used to refit."""
        raise NotImplementedError


class SparseDotRetriever(RankedModel):
    def __init__(self, corpus_matrix, doc_ids, top_k=100):
        super().__init__(doc_ids, top_k)
        self.corpus = sparse.csr_matrix(corpus_matrix, dtype=np.float32)
        if self.corpus.shape[0] != len(doc_ids):
            raise ValueError("doc_ids misaligned with corpus rows")

    def rank(self, queries, batch_size=16):
        queries = sparse.csr_matrix(queries, dtype=np.float32)
        if queries.shape[1] != self.corpus.shape[1]:
            raise ValueError("Vocabulary mismatch")
        for start in range(0, queries.shape[0], batch_size):
            sims = (queries[start:start + batch_size] @ self.corpus.T).tocsr()
            for i in range(sims.shape[0]):
                lo, hi = sims.indptr[i:i + 2]
                yield self._topk(sims.indices[lo:hi], sims.data[lo:hi])


class BM25Retriever(SparseDotRetriever):
    """Robertson BM25 with non-negative, smoothed IDF.

    score(q,d) = sum_{t in unique(q)} log(1+(N-df+.5)/(df+.5))
                 * tf*(k1+1)/(tf + k1*(1-b+b*dl/avgdl)).
    """
    def __init__(self, counts, doc_lengths, doc_ids, k1=1.2, b=0.75, top_k=100):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1>0 and 0<=b<=1")
        counts = sparse.csr_matrix(counts, dtype=np.float32)
        n_docs = counts.shape[0]
        lengths = np.asarray(doc_lengths, dtype=np.float32).ravel()
        if lengths.size != n_docs:
            raise ValueError("Document lengths misaligned")
        avgdl = float(lengths.mean())
        if avgdl <= 0:
            raise ValueError("No indexed terms")
        df = np.diff(counts.tocsc().indptr).astype(np.float32)
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        weights = counts.copy()
        row_length = np.repeat(lengths, np.diff(weights.indptr))
        k = k1 * (1.0 - b + b * row_length / avgdl)
        tf = weights.data
        weights.data = ((tf * (k1 + 1.0)) / (tf + k) * idf[weights.indices]).astype(np.float32)
        super().__init__(weights, doc_ids, top_k)
        self.k1, self.b, self.avgdl = k1, b, avgdl

    def rank(self, queries, batch_size=16):
        # BM25 uses unique query terms; query TF does not duplicate term weight.
        queries = sparse.csr_matrix(queries, dtype=np.float32).copy()
        queries.data[:] = 1
        yield from super().rank(queries, batch_size)


class LSIRetriever(RankedModel):
    """Truncated SVD of corpus TF-IDF; cosine in latent space."""
    def __init__(self, tfidf, doc_ids, dimensions=64, seed=42, top_k=100):
        super().__init__(doc_ids, top_k)
        tfidf = sparse.csr_matrix(tfidf, dtype=np.float32)
        if dimensions >= min(tfidf.shape):
            raise ValueError("SVD dimensions must be below min(corpus shape)")
        self.svd = TruncatedSVD(n_components=dimensions, n_iter=3, random_state=seed)
        self.docs = normalize(self.svd.fit_transform(tfidf).astype(np.float32), norm="l2")
        self.dimensions, self.seed = dimensions, seed

    def rank(self, queries, batch_size=16):
        queries = sparse.csr_matrix(queries, dtype=np.float32)
        for start in range(0, queries.shape[0], batch_size):
            projected = normalize(self.svd.transform(queries[start:start + batch_size]).astype(np.float32))
            similarities = projected @ self.docs.T
            for scores in similarities:
                yield self._topk(np.arange(len(self.doc_ids)), scores)
