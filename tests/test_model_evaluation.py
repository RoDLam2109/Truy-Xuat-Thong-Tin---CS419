"""Small deterministic retrieval and evaluation tests; no external dataset required."""
import math
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from src.retrieval.models import BM25Retriever, LSIRetriever, SparseDotRetriever
from src.retrieval.evaluation import query_metrics, aggregate, degradation


class ModelTests(unittest.TestCase):
    def setUp(self):
        # Documents: a = 'a a b'; b = 'b'; c = 'c'
        self.counts = csr_matrix([[2, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        self.doc_ids = ["a", "b", "c"]

    def test_vsm_returns_top_positive_and_empty_query(self):
        model = SparseDotRetriever(self.counts, self.doc_ids, top_k=2)
        q = csr_matrix([[1, 0, 0], [0, 0, 0]], dtype=np.float32)
        result = list(model.rank(q))
        self.assertEqual([self.doc_ids[i] for i, _ in result[0]], ["a"])
        self.assertEqual(result[1], [])

    def test_bm25_uses_count_and_nonnegative_idf(self):
        model = BM25Retriever(self.counts, [3, 1, 1], self.doc_ids, k1=1.2, b=0.75)
        rows = list(model.rank(csr_matrix([[1, 0, 0], [0, 1, 0]], dtype=np.float32)))
        self.assertEqual(self.doc_ids[rows[0][0][0]], "a")
        # b is more concise and has the same tf=1 as a for term b.
        self.assertEqual(self.doc_ids[rows[1][0][0]], "b")
        self.assertGreater(rows[0][0][1], 0)

    def test_bm25_ignores_repeated_query_term(self):
        model = BM25Retriever(self.counts, [3, 1, 1], self.doc_ids)
        once = list(model.rank(csr_matrix([[1, 0, 0]], dtype=np.float32)))[0]
        twice = list(model.rank(csr_matrix([[5, 0, 0]], dtype=np.float32)))[0]
        self.assertEqual(once, twice)

    def test_deterministic_tie_by_doc_id(self):
        model = SparseDotRetriever(csr_matrix([[1], [1], [1]]),
                                   ["z", "a", "m"], top_k=2)
        result = list(model.rank(csr_matrix([[1]])))[0]
        self.assertEqual([model.doc_ids[i] for i, _ in result], ["a", "m"])

    def test_lsi_runs_and_does_not_rank_zero_query(self):
        model = LSIRetriever(self.counts, self.doc_ids, dimensions=2, seed=42)
        result = list(model.rank(csr_matrix([[1, 0, 0], [0, 0, 0]])))
        self.assertLessEqual(len(result[0]), 3)
        self.assertEqual(result[1], [])


class MetricTests(unittest.TestCase):
    def test_multiple_relevant_and_empty(self):
        qrels = {"a": 1, "b": 1}
        actual = query_metrics(["x", "b", "a"], qrels)
        expected_dcg = 1 / math.log2(3) + 1 / math.log2(4)
        expected_idcg = 1 + 1 / math.log2(3)
        self.assertAlmostEqual(actual["mrr@10"], 0.5)
        self.assertAlmostEqual(actual["ndcg@10"], expected_dcg / expected_idcg)
        self.assertEqual(actual["recall@100"], 1.0)
        self.assertEqual(query_metrics([], qrels)["recall@100"], 0)

    def test_duplicate_document_is_error(self):
        with self.assertRaises(ValueError):
            query_metrics(["a", "a"], {"a": 1})

    def test_aggregate_and_drop(self):
        clean = aggregate([query_metrics(["a"], {"a": 1}),
                           query_metrics([], {"b": 1})])
        noisy = {"mrr@10": 0, "ndcg@10": 0, "recall@100": 0}
        self.assertEqual(clean["mrr@10"], 0.5)
        self.assertEqual(degradation(clean, noisy)["mrr@10_drop_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
