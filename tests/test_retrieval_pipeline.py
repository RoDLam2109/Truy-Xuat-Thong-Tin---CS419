"""End-to-end behavior, not model score tuning against test labels."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from scripts import prepare_validation as split
from scripts.build_retrieval_index import sha256
from scripts.evaluate_models import evaluate
from src.retrieval.feature_matrix import to_sparse, fit_corpus, tfidf_queries


class ValidationSplitTests(unittest.TestCase):
    def test_grouped_split_is_reproducible_and_preserves_multi_positive(self):
        queries = [{"id": f"q{i}", "text": f"Văn bản pháp luật {i}"}
                   for i in range(12)]
        queries[1]["text"] = "  VĂN bản pháp   luật 0 "
        qrels = [{"query-id": r["id"], "corpus-id": "d1", "score": "1"}
                 for r in queries]
        qrels.append({"query-id": "q0", "corpus-id": "d2", "score": "1"})
        test = [{"id": "t1", "text": "Truy vấn test độc lập"}]
        a, b = split.split_rows(queries, qrels, test, .25, 42)
        self.assertEqual((a, b), split.split_rows(queries, qrels, test, .25, 42))
        self.assertEqual(sum(map(len, (a[0], b[0]))), len(queries))
        self.assertEqual(sum(map(len, (a[1], b[1]))), len(qrels))
        self.assertFalse({split.text_key(x["text"]) for x in a[0]}
                         & {split.text_key(x["text"]) for x in b[0]})
        self.assertEqual(len({r["corpus-id"] for r in qrels
                              if r["query-id"] == "q0"}), 2)
        with self.assertRaisesRegex(ValueError, "overlap"):
            split.split_rows(queries, qrels, [{"id": "q1", "text": "different"}])
        with self.assertRaises(ValueError):
            split.split_rows(queries, qrels, test, 1.0)

    def test_qrels_reader_rejects_duplicate_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "qrels.tsv"
            path.write_text(
                "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td1\t1\n",
                encoding="utf-8")
            with self.assertRaises(ValueError):
                split.read_positive_qrels(path)


class MatrixTests(unittest.TestCase):
    def test_corpus_only_idf_and_repeated_term_frequency(self):
        vocab = {"xe": 0, "tàu": 1}
        docs = [{"id": "d1", "text": "xe xe", "word_terms": ["xe", "xe"]},
                {"id": "d2", "text": "tàu", "word_terms": ["tàu"]}]
        count = to_sparse(docs, vocab, "word")
        self.assertEqual(count.toarray().tolist(), [[2.0, 0.0], [0.0, 1.0]])
        tfidf, idf = fit_corpus(count)
        self.assertEqual(tfidf.shape, (2, 2))
        qtfidf = tfidf_queries(to_sparse([{"text": "xe"}], vocab, "word"), idf)
        self.assertAlmostEqual(float(qtfidf.multiply(qtfidf).sum()), 1)
        self.assertEqual(to_sparse([{"text": "unseen"}], vocab, "word").nnz, 0)

    def test_character_features_preserve_count_and_term_boundaries(self):
        grams = {" xe": 0, "xe ": 1, " tà": 2}
        matrix = to_sparse([{"text": "xe xe"}], grams, "char_ngram")
        self.assertEqual(matrix[0, 0], 2)
        self.assertEqual(matrix[0, 1], 2)
        self.assertEqual(matrix[0, 2], 0)


class EvaluationSmokeTests(unittest.TestCase):
    def test_all_three_models_eight_conditions_and_held_out_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, splits, index, out = [root / n for n in
                                        ("data", "splits", "index", "results")]
            (splits / "val/noise").mkdir(parents=True)
            data.mkdir()
            index.mkdir()
            corpus = data / "corpus.jsonl"
            corpus.write_text('{"id":"d1","title":"","text":"xe"}\n'
                              '{"id":"d2","title":"","text":"tàu"}\n'
                              '{"id":"d3","title":"","text":"máy"}\n', encoding="utf-8")
            vocab = {"xe": 0, "tàu": 1, "máy": 2}
            counts = sparse.csr_matrix(np.eye(3, dtype=np.float32))
            tfidf, idf = fit_corpus(counts)
            sparse.save_npz(index / "counts.npz", counts)
            sparse.save_npz(index / "tfidf.npz", tfidf)
            np.save(index / "doc_lengths.npy", np.ones(3, dtype=np.float32))
            np.save(index / "idf.npy", idf)
            (index / "doc_ids.json").write_text(json.dumps(["d1", "d2", "d3"]))
            (index / "vocab.json").write_text(json.dumps(vocab))
            meta = {"full_corpus": True, "branch": "word",
                    "raw_corpus_sha256": sha256(corpus),
                    "term_config": {"stopwords": [], "ngram_range": [3, 5]}}
            (index / "index_meta.json").write_text(json.dumps(meta))
            (splits / "val/queries_clean.jsonl").write_text(
                '{"id":"q1","text":"xe"}\n', encoding="utf-8")
            (splits / "val/qrels_val.tsv").write_text(
                "query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")
            for condition in ("no_diacritics", "telex_typo", "phonetic", "abbrev",
                              "mixed_low", "mixed_medium", "mixed_high"):
                (splits / "val/noise" / f"queries_{condition}.jsonl").write_text(
                    '{"id":"q1","text":"xe"}\n', encoding="utf-8")
            summaries = evaluate(index, data, splits, "val", out,
                                 models=("vsm", "bm25", "lsi"), svd_dim=1)
            self.assertEqual(len(summaries), 24)
            self.assertAlmostEqual(summaries[0]["ndcg@10"], 1.0)
            with (out / "summary.csv").open(encoding="utf-8") as result:
                self.assertEqual(len(list(csv.DictReader(result))), 24)
            with self.assertRaisesRegex(ValueError, "Final test"):
                evaluate(index, data, splits, "test", root / "oops", models=("vsm",))


if __name__ == "__main__":
    unittest.main()
