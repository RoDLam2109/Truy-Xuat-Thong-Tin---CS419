import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import prepare_train as train


class PrepareTrainTests(unittest.TestCase):
    def setUp(self):
        self.queries = [
            {"_id": "keep", "text": "Câu hỏi train"},
            {"_id": "keep", "text": "Câu hỏi train"},
            {"_id": "test", "text": "Câu hỏi test"},
            {"_id": "same-text", "text": "  CÂU  HỎI test  "},
            {"_id": "unlabeled", "text": "Không có nhãn train"},
        ]
        self.qrels = [self.rel("keep", "d2"), self.rel("keep", "d1"),
                      self.rel("test", "d1"), self.rel("same-text", "d2")]
        self.test = [{"id": "test", "text": "Câu hỏi test"}]

    @staticmethod
    def rel(qid, doc_id):
        return {"query-id": qid, "corpus-id": doc_id, "score": 1}

    def prepare(self, queries=None, qrels=None):
        return train.prepare_rows(
            self.queries if queries is None else queries,
            self.qrels if qrels is None else qrels,
            self.test, {"d1", "d2"},
        )

    def test_removes_test_leakage_and_keeps_all_positive_labels(self):
        queries, qrels, report = self.prepare()
        self.assertEqual(queries, [{"id": "keep", "text": "Câu hỏi train"}])
        self.assertEqual(qrels, self.qrels[:2])
        self.assertEqual(report["excluded_test_id_overlap"], ["test"])
        self.assertEqual(report["excluded_test_text_overlap"], ["same-text"])
        self.assertEqual(report["duplicate_source_query_rows_removed"], 1)
        self.assertEqual(report["relevant_documents_per_query"], {2: 1})
        self.assertEqual(report["remaining_test_id_overlap"], 0)
        self.assertEqual(report["remaining_test_text_overlap"], 0)

    def test_unicode_decomposed_text_cannot_leak_into_train(self):
        queries = self.queries + [{"_id": "nfd", "text": "Ca\u0302u ho\u0309i test"}]
        _, _, report = self.prepare(queries, self.qrels + [self.rel("nfd", "d1")])
        self.assertIn("nfd", report["excluded_test_text_overlap"])

    def test_conflicting_query_text_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mâu thuẫn"):
            self.prepare(self.queries + [{"_id": "keep", "text": "Khác"}])

    def test_dangling_labels_and_duplicate_pairs_are_rejected(self):
        cases = [
            (self.rel("missing", "d1"), "query không tồn tại"),
            (self.rel("keep", "missing"), "tài liệu không tồn tại"),
            (self.rel("keep", "d1"), "trùng cặp"),
            ({**self.rel("keep", "d1"), "score": 0}, "score = 1"),
        ]
        for row, message in cases:
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, message):
                self.prepare(qrels=self.qrels + [row])

    def test_rejects_empty_test_or_train(self):
        with self.assertRaisesRegex(ValueError, "test rỗng"):
            train.prepare_rows(self.queries, self.qrels, [], {"d1", "d2"})
        with self.assertRaisesRegex(ValueError, "Không còn query train"):
            self.prepare(qrels=[self.rel("test", "d1")])

    def test_file_outputs_are_reproducible_and_preserve_evaluation_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            (raw / "qrels").mkdir(parents=True)
            for path, rows in [
                (raw / "queries.jsonl", self.queries),
                (raw / "qrels/train.jsonl", self.qrels),
                (root / "queries_clean.jsonl", self.test),
                (root / "corpus.jsonl", [{"id": "d1"}, {"id": "d2"}]),
            ]:
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            (root / "qrels_test.tsv").write_bytes(b"original-test-labels\n")
            protected = {name: (root / name).read_bytes() for name in (
                "queries_clean.jsonl", "corpus.jsonl", "qrels_test.tsv"
            )}
            hashes = {name: train.sha256(raw / name) for name in train.SOURCE_HASHES}
            with patch.object(train, "SOURCE_HASHES", hashes):
                train.prepare(root, raw)
                outputs = {name: (root / name).read_bytes() for name in (
                    "queries_train.jsonl", "qrels_train.tsv", "train_manifest.json"
                )}
                train.prepare(root, raw)
            for name, payload in {**protected, **outputs}.items():
                self.assertEqual((root / name).read_bytes(), payload, name)
            self.assertEqual(
                outputs["qrels_train.tsv"].decode(),
                "query-id\tcorpus-id\tscore\nkeep\td2\t1\nkeep\td1\t1\n",
            )

    def test_invalid_source_checksum_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "queries.jsonl").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                train.ensure_sources(root, download=False)


if __name__ == "__main__":
    unittest.main()
