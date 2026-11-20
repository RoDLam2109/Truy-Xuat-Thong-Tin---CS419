import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import check_dataset as check, generate_noise as noise, prepare_train as train


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / "data"
        self.raw = self.data / "raw"
        self.reports = Path(self.temp.name) / "reports"
        (self.raw / "qrels").mkdir(parents=True)
        self.noise_config = check.PROJECT_ROOT / "configs/noise.json"
        corpus = [{"id": "d1", "title": "Điều 1", "text": "Không được với"},
                  {"id": "d2", "title": "Điều 2", "text": "Quy định"}]
        queries = [{"_id": "train", "text": "Không được với?"},
                   {"_id": "test", "text": "Quy định không được?"}]
        test_rels = [{"query-id": "test", "corpus-id": "d1", "score": 1}]
        train_rels = [{"query-id": "train", "corpus-id": doc, "score": 1} for doc in ("d1", "d2")]
        for path, rows in (
            (self.raw / "corpus.jsonl", [{"_id": r["id"], "title": r["title"], "text": r["text"]} for r in corpus]),
            (self.raw / "queries.jsonl", queries),
            (self.raw / "qrels/test.jsonl", test_rels),
            (self.raw / "qrels/train.jsonl", train_rels),
            (self.data / "corpus.jsonl", corpus),
            (self.data / "queries_clean.jsonl", [{"id": "test", "text": queries[1]["text"]}]),
        ):
            noise.write_jsonl(path, rows)
        (self.data / "qrels_test.tsv").write_text("query-id\tcorpus-id\tscore\ntest\td1\t1\n", encoding="utf-8")
        source_names = ("corpus.jsonl", "queries.jsonl", "qrels/train.jsonl", "qrels/test.jsonl")
        self.config = {"dataset": train.DATASET, "revision": train.REVISION,
                       "source_sha256": {name: check.checksum(self.raw / name) for name in source_names},
                       "expected_counts": dict(corpus=2, test_queries=1, test_qrels=1, train_queries=1, train_qrels=2)}
        hashes = {name: self.config["source_sha256"][name] for name in train.SOURCE_HASHES}
        with patch.object(train, "SOURCE_HASHES", hashes):
            train.prepare(self.data, self.raw)
        args = argparse.Namespace(input=self.data / "queries_clean.jsonl", qrels=self.data / "qrels_test.tsv",
                                  config=self.noise_config, output_dir=self.data / "noise", report_dir=self.reports,
                                  seed=42, sample_size=10)
        with contextlib.redirect_stdout(io.StringIO()):
            noise.generate(args)

    def audit(self):
        return check.audit(self.data, self.reports, self.config, self.noise_config)

    def test_complete_source_and_noise_audit_preserves_all_inputs(self):
        before = {p: check.checksum(p) for p in self.data.rglob("*") if p.is_file()}
        result = self.audit()
        self.assertEqual(result["failed_checks"], [])
        self.assertEqual(result["documents_per_query"]["train"], {2: 1})
        self.assertEqual(before, {p: check.checksum(p) for p in before})

    def test_dropping_second_relevant_document_is_detected(self):
        path = self.data / "qrels_train.tsv"
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        result = self.audit()
        self.assertIn("train_qrels_match_source_all_labels_after_test_exclusion", result["failed_checks"])

    def test_corrupted_noise_text_is_detected(self):
        path = self.data / "noise/queries_phonetic.jsonl"
        noise.write_jsonl(path, [{"id": "test", "text": "Changed query"}])
        result = self.audit()
        self.assertIn("phonetic_reproducible", result["failed_checks"])
        self.assertIn("phonetic_checksum", result["failed_checks"])

    def test_tampered_source_is_rejected(self):
        (self.raw / "qrels/test.jsonl").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Checksum nguồn"):
            self.audit()


if __name__ == "__main__":
    unittest.main()
