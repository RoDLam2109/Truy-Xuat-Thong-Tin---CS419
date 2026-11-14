import json
import tempfile
import unittest
import unicodedata
from pathlib import Path

import numpy as np
from scipy import sparse

from src.data_io import sha256, write_jsonl
from src.terms import TermIndex, VietnameseTerms, build_index

ROOT = Path(__file__).resolve().parents[1]


class TermTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/terms.json").read_text())

    def test_real_segmenter_unicode_punctuation_negation_and_digits(self):
        extract = VietnameseTerms(self.config)
        text = "Tôi mua bảo hiểm y tế, không mất 5 năm."
        tokens = extract(text)
        self.assertIn("bảo_hiểm", tokens)
        self.assertIn("y_tế", tokens)
        self.assertIn("không", tokens)
        self.assertIn("5", tokens)
        self.assertEqual(tokens, extract(unicodedata.normalize("NFD", text)))
        self.assertNotIn(",", tokens)
        self.assertEqual(extract(" ! ? \t"), [])

    def make_data(self, root):
        corpus, queries, cfg = root / "corpus.jsonl", root / "queries.jsonl", root / "config.json"
        write_jsonl(corpus, [{"id": "doc-2", "title": "Bảo hiểm", "text": "bảo hiểm y tế"},
                             {"id": "doc-1", "title": "", "text": "xe máy"}])
        write_jsonl(queries, [{"id": "q2", "text": "bảo hiểm y tế"},
                              {"id": "q1", "text": "zzzzzzzzzzzz"},
                              {"id": "q0", "text": "???"}])
        cfg.write_text(json.dumps(self.config))
        return corpus, queries, cfg

    def test_counts_df_tfidf_ids_reload_and_query_cannot_fit_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, queries, cfg = self.make_data(root)
            hashes = [sha256(p) for p in (corpus, queries, cfg)]
            output = root / "index"
            stats = build_index(corpus, queries, cfg, output)
            index = TermIndex(output)
            self.assertEqual(index.doc_ids, ["doc-2", "doc-1"])
            vocabulary = dict(index.vocabulary)
            self.assertNotIn("zzzzzzzzzzzz", vocabulary)
            counts = sparse.load_npz(output / "counts.npz")
            self.assertEqual(counts[0, vocabulary["bảo_hiểm"]], 2)
            expected_idf = np.log((1 + 2) / (1 + 1)) + 1
            self.assertAlmostEqual(np.load(output / "idf.npy")[vocabulary["bảo_hiểm"]], expected_idf)
            self.assertIn("bảo_hiểm\t1\t1", (output / "term_df.tsv").read_text())
            np.testing.assert_equal(np.load(output / "doc_lengths.npy"), np.asarray(counts.sum(axis=1)).ravel())
            query_counts, query_tfidf = index.transform(["bảo hiểm y tế", "zzzzzzzzzzzz", "???"])
            np.testing.assert_allclose(query_tfidf.toarray(), sparse.load_npz(output / "queries_clean.tfidf.npz").toarray())
            self.assertEqual(query_counts[1:].nnz, 0)
            self.assertEqual(index.vocabulary, vocabulary)
            self.assertEqual(stats["empty_queries"], 2)
            self.assertEqual(json.loads((output / "query_ids.json").read_text()), ["q2", "q1", "q0"])
            self.assertEqual(hashes, [sha256(p) for p in (corpus, queries, cfg)])
            second = root / "second"
            build_index(corpus, queries, cfg, second)
            self.assertEqual(sha256(output / "vocabulary.json"), sha256(second / "vocabulary.json"))
            np.testing.assert_array_equal(counts.toarray(), sparse.load_npz(second / "counts.npz").toarray())
            np.testing.assert_allclose(sparse.load_npz(output / "tfidf.npz").toarray(), sparse.load_npz(second / "tfidf.npz").toarray())

    def test_artifact_modification_and_output_collisions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, queries, cfg = self.make_data(root)
            with self.assertRaisesRegex(ValueError, "chứa file đầu vào"):
                build_index(corpus, queries, cfg, root)
            output = root / "index"
            build_index(corpus, queries, cfg, output)
            with self.assertRaisesRegex(ValueError, "đã tồn tại"):
                build_index(corpus, queries, cfg, output)
            (output / "config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "bị thay đổi"):
                TermIndex(output)


if __name__ == "__main__":
    unittest.main()
