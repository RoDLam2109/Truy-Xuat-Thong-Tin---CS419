import argparse
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts import generate_noise as noise


class NoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = noise.PROJECT_ROOT / "configs/noise.json"
        cls.config = json.loads(cls.config_path.read_text(encoding="utf-8"))

    def transform(self, text, condition, probability=1):
        row = {"id": "test-query", "text": text}
        _, tokens = noise.analyze_queries([row], self.config)[0]
        return noise.generate_query(row, tokens, condition, probability, 42)

    def test_unicode_and_diacritics(self):
        self.assertEqual(noise.strip_diacritics("ĐƯỢC, Điều 12/NĐ-CP"), "DUOC, Dieu 12/ND-CP")
        self.assertEqual(noise.strip_diacritics("a\u0301"), "a")

    def test_keyboard_errors_follow_explicit_key_rules(self):
        variants = dict(noise.telex_candidates("phạt"))
        self.assertEqual(variants["phat"], "missing_tone_key")
        self.assertEqual(variants["phatj"], "uncomposed_telex_tone_key")
        self.assertEqual(variants["phat5"], "uncomposed_vni_tone_key")
        self.assertIn("đàu", dict(noise.telex_candidates("đầu")))
        self.assertIn("dược", dict(noise.telex_candidates("được")))
        self.assertEqual(noise.telex_candidates("xe"), [])

    def test_spaces_case_and_punctuation_are_preserved(self):
        result, audit = self.transform("\tKhông,  được?\n", "abbrev")
        self.assertTrue(result["text"].startswith(("\tKo,  ", "\tK,  ")))
        self.assertTrue(result["text"].endswith("?\n"))
        self.assertEqual(audit["total_tokens"], 2)
        self.assertEqual(audit["changed_tokens"], 2)

    def test_protected_identifiers_names_and_numeric_units(self):
        text = "hỏi 12/2020/NĐ-CP, 1.000 đồng 5 năm UBND Hà Nội?"
        for condition in ("no_diacritics", "telex_typo", "phonetic", "abbrev", "mixed_high"):
            with self.subTest(condition=condition):
                result, audit = self.transform(text, condition)
                self.assertTrue(result["text"].endswith("12/2020/NĐ-CP, 1.000 đồng 5 năm UBND Hà Nội?"))
                self.assertTrue(all(change["token_index"] == 0 for change in audit["changes"]))

    def test_token_accounting_and_no_forced_changes(self):
        result, audit = self.transform("Không?  5,  !", "abbrev")
        self.assertNotEqual(result["text"], "Không?  5,  !")
        self.assertEqual(audit["total_tokens"], 2)
        self.assertEqual(audit["selected_tokens"], 2)
        self.assertEqual(audit["changed_tokens"], 1)
        self.assertEqual(audit["selected_without_change"], 1)
        clean, audit = self.transform("không được với?", "abbrev", 0)
        self.assertEqual(clean["text"], "không được với?")
        self.assertEqual(audit["changes"], [])

    def test_mixed_levels_keep_shared_changes(self):
        text = "không được trong trường nước " * 30
        changes = []
        for level in ("low", "medium", "high"):
            _, audit = self.transform(text, f"mixed_{level}", self.config["mixed_probabilities"][level])
            changes.append({change["token_index"]: change for change in audit["changes"]})
        self.assertTrue(set(changes[0]) < set(changes[1]) < set(changes[2]))
        for lower, higher in zip(changes, changes[1:]):
            for index, change in lower.items():
                self.assertEqual(change, higher[index])

    def test_partial_diacritics_uses_token_selection_and_keeps_other_accents(self):
        text = "không được chạy trong trường nước " * 12
        result, audit = self.transform(text, "no_diacritics", self.config["single_probability"])
        self.assertGreater(audit["changed_tokens"], 0)
        self.assertLess(audit["changed_tokens"], audit["eligible_tokens"])
        self.assertNotEqual(result["text"], noise.strip_diacritics(text))
        original = text.split()
        actual = result["text"].split()
        changed = {c["token_index"] for c in audit["changes"]}
        for index, (before, after) in enumerate(zip(original, actual)):
            self.assertEqual(after, noise.strip_diacritics(before) if index in changed else before)
        clean, empty_audit = self.transform(text, "no_diacritics", 0)
        self.assertEqual(clean["text"], text)
        self.assertEqual(empty_audit["changed_tokens"], 0)

    def test_collisions_keep_distinct_queries(self):
        clean = [{"id": "1", "text": "có"}, {"id": "2", "text": "cò"}]
        noisy = [{"id": row["id"], "text": "co"} for row in clean]
        groups = noise.collision_groups(clean, noisy)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["distinct_clean_texts"], 2)
        self.assertEqual(groups[0]["qids"], ["1", "2"])

    def test_duplicate_ids_are_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.jsonl"
            path.write_text('{"id":"a","text":"one"}\n{"id":"a","text":"two"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ID trùng"):
                noise.read_queries(path)

    def test_qrels_must_cover_all_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qrels.tsv"
            path.write_text("query-id\tcorpus-id\tscore\na\tdoc\t1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Qrels"):
                noise.validate_qrels(path, [{"id": "b", "text": "test"}])

    def test_pipeline_is_reproducible_and_leaves_inputs_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "queries_clean.jsonl"
            rows = [
                {"id": "a", "text": "Không được chạy xe trong trường?"},
                {"id": "b", "text": "có 12/2020/NĐ-CP và 5 năm"},
                {"id": "c", "text": "cò"},
            ]
            noise.write_jsonl(source, rows)
            original_hash = noise.sha256(source)
            hashes = []
            for run in ("first", "second"):
                output_dir, report_dir = root / run / "data", root / run / "reports"
                args = argparse.Namespace(
                    input=source, config=self.config_path, qrels=None,
                    output_dir=output_dir, report_dir=report_dir, seed=42, sample_size=2,
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    stats = noise.generate(args)
                self.assertEqual(len(stats["conditions"]), 7)
                self.assertEqual(stats["conditions"]["no_diacritics"]["target_token_probability"],
                                 self.config["single_probability"])
                self.assertEqual(len(list(output_dir.glob("queries_*.jsonl"))), 7)
                for condition, counts in stats["conditions"].items():
                    actual = noise.read_queries(output_dir / counts["output_file"])
                    self.assertEqual([row["id"] for row in actual], [row["id"] for row in rows])
                    audits = [json.loads(line) for line in (report_dir / counts["audit_file"]).read_text(encoding="utf-8").splitlines()]
                    for clean, noisy, audit in zip(rows, actual, audits):
                        # Log phải đủ để dựng lại chính xác câu đầu ra.
                        restored = clean["text"]
                        for change in reversed(audit["changes"]):
                            self.assertEqual(clean["text"][change["start"]:change["end"]], change["before"])
                            restored = restored[:change["start"]] + change["after"] + restored[change["end"]:]
                        self.assertEqual(restored, noisy["text"])
                        self.assertEqual(audit["selected_tokens"], audit["changed_tokens"] + audit["selected_without_change"])
                        self.assertEqual(len({change["token_index"] for change in audit["changes"]}), audit["changed_tokens"])
                hashes.append({
                    str(path.relative_to(root / run)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (root / run).rglob("*") if path.is_file()
                })
            self.assertEqual(hashes[0], hashes[1])
            self.assertEqual(noise.sha256(source), original_hash)


if __name__ == "__main__":
    unittest.main()
