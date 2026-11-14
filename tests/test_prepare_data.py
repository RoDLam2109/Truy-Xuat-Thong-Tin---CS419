import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_data import export_dataset
from src.data_io import read_qrels, read_records


class PrepareDataTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [{"id": "12/2020/NĐ-CP#1", "title": "Điều 1", "text": "Nội dung"}]
        self.queries = [{"query_id": "train", "question": "Câu phát triển?"},
                        {"query_id": "test", "question": "Câu đánh giá?"}]
        self.qrels = {s: [{"query_id": s, "corpus_id": self.corpus[0]["id"], "score": 1}] for s in ("train", "test")}

    def test_export_preserves_ids_and_separates_queries_by_qrels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            stats = export_dataset(self.corpus, self.queries + self.queries[:1], self.qrels, root, {"revision": "example"})
            self.assertEqual(stats["duplicate_identical_query_rows_removed"], 1)
            self.assertEqual([r["id"] for r in read_records(root / "queries_clean.jsonl")], ["test"])
            self.assertEqual([r["id"] for r in read_records(root / "queries_train.jsonl")], ["train"])
            self.assertEqual(list(read_records(root / "corpus.jsonl", corpus=True)), self.corpus)
            self.assertEqual(read_qrels(root / "qrels_test.tsv", {"test"}, {self.corpus[0]["id"]})[0][2], 1)
            self.assertTrue(json.loads((root / "manifest.json").read_text())["files_sha256"])
            with self.assertRaisesRegex(ValueError, "đã tồn tại"):
                export_dataset([], [], self.qrels, root, {})

    def test_conflicting_query_ids_and_split_overlap_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "khác nội dung"):
                export_dataset(self.corpus, self.queries + [{"query_id": "test", "question": "khác"}], self.qrels, Path(directory) / "one", {})
            self.qrels["train"] = self.qrels["test"]
            with self.assertRaisesRegex(ValueError, "trùng query"):
                export_dataset(self.corpus, self.queries, self.qrels, Path(directory) / "two", {})

    def test_dangling_qrels_and_duplicate_documents_do_not_produce_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.qrels["test"][0]["corpus_id"] = "missing"
            with self.assertRaisesRegex(ValueError, "không tồn tại"):
                export_dataset(self.corpus, self.queries, self.qrels, root / "one", {})
            self.assertFalse((root / "one/manifest.json").exists())
            with self.assertRaisesRegex(ValueError, "bị trùng"):
                export_dataset(self.corpus * 2, self.queries, self.qrels, root / "two", {})
            self.assertFalse((root / "two/manifest.json").exists())

    def test_readers_reject_duplicate_ids_and_uncovered_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "queries.jsonl"
            records.write_text('{"id":"a","text":"x"}\n{"id":"a","text":"y"}\n')
            with self.assertRaisesRegex(ValueError, "ID trùng"):
                list(read_records(records))
            qrels = root / "qrels.tsv"
            qrels.write_text("query-id\tcorpus-id\tscore\na\td\t1\n")
            with self.assertRaisesRegex(ValueError, "Mỗi query"):
                read_qrels(qrels, {"a", "b"}, {"d"})


if __name__ == "__main__":
    unittest.main()
