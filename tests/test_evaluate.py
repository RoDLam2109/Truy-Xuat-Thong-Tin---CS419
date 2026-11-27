"""Lam's independent tests of metrics boundaries and experiment integration."""
import copy
from contextlib import redirect_stdout
import csv
import json
import io
from importlib.util import find_spec
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse

from scripts.build_retrieval_index import sha256
from scripts.evaluate_models import CONDITIONS
from scripts.report_experiments import report, select_configuration
from scripts.run_experiments import PROTOCOL, run
from src.evaluation.runs import evaluate_rankings, read_qrels, write_json
from src.retrieval.evaluation import degradation, query_metrics
from src.retrieval.feature_matrix import fit_corpus, to_sparse
from scripts.prepare_terms import char_terms


class MetricBoundaryTests(unittest.TestCase):
    def test_cutoffs_ten_and_one_hundred(self):
        ranking = [f"d{i}" for i in range(1, 102)]
        qrels = {"d10": 1, "d11": 1, "d100": 1, "d101": 1}
        scores = query_metrics(ranking, qrels)
        self.assertAlmostEqual(scores["mrr@10"], .1)
        self.assertAlmostEqual(scores["recall@100"], .75)
        ideal = sum(1 / math.log2(i + 1) for i in range(1, 5))
        self.assertAlmostEqual(scores["ndcg@10"], (1 / math.log2(11)) / ideal)

    def test_graded_ndcg_and_all_relevant_denominator(self):
        scores = query_metrics(["low", "high"], {"high": 3, "low": 1, "missing": 2})
        dcg = 1 + 7 / math.log2(3)
        ideal = 7 + 3 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(scores["ndcg@10"], dcg / ideal)
        self.assertAlmostEqual(scores["recall@100"], 2 / 3)

    def test_zero_clean_is_undefined_and_improvement_has_negative_drop(self):
        zero = dict.fromkeys(("mrr@10", "ndcg@10", "recall@100"), 0.)
        one = dict.fromkeys(zero, 1.)
        drops = degradation(zero, one)
        self.assertIsNone(drops["ndcg@10_drop_pct"])
        self.assertEqual(drops["ndcg@10_drop_abs"], -1.)
        self.assertEqual(degradation(dict.fromkeys(zero, .5), one)["ndcg@10_drop_pct"], -100.)


class ExportContractTests(unittest.TestCase):
    def setUp(self):
        self.rankings = [dict(qid="q1", doc_id="a", rank=1, score=2.)]
        self.statuses = [dict(qid="q1", status="ok", retrieved=1),
                         dict(qid="q2", status="empty", retrieved=0)]
        self.qids = ["q1", "q2"]
        self.qrels = {"q1": {"a": 1, "b": 1}, "q2": {"b": 1}}

    def evaluate(self, rankings=None, statuses=None):
        return evaluate_rankings(self.rankings if rankings is None else rankings,
                                 self.statuses if statuses is None else statuses,
                                 self.qids, self.qrels)

    def test_explicit_empty_query_counts_in_macro_mean(self):
        per_query, summary = self.evaluate()
        self.assertEqual(len(per_query), 2)
        self.assertEqual(summary["mrr@10"], .5)
        self.assertEqual(summary["recall@100"], .25)

    def test_missing_and_duplicate_status_rejected(self):
        for statuses in (self.statuses[:1], self.statuses + self.statuses[:1]):
            with self.subTest(statuses=statuses), self.assertRaises(ValueError):
                self.evaluate(statuses=statuses)

    def test_unknown_query_and_duplicate_document_rejected(self):
        for extra in (dict(qid="other", doc_id="a", rank=1, score=2.),
                      dict(qid="q1", doc_id="a", rank=2, score=1.)):
            statuses = copy.deepcopy(self.statuses)
            statuses[0]["retrieved"] = 2
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.evaluate(rankings=self.rankings + [extra], statuses=statuses)

    def test_invalid_score_rank_and_status_rejected(self):
        for field, value in (("score", float("nan")), ("score", float("inf")),
                             ("score", 0), ("rank", 2), ("rank", 1.0)):
            rankings = copy.deepcopy(self.rankings)
            rankings[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.evaluate(rankings=rankings)
        statuses = copy.deepcopy(self.statuses)
        statuses[1]["status"] = "ok"
        with self.assertRaises(ValueError):
            self.evaluate(statuses=statuses)

    def test_score_tie_requires_document_id_order(self):
        ranks = [dict(qid="q1", doc_id="b", rank=1, score=1.),
                 dict(qid="q1", doc_id="a", rank=2, score=1.)]
        statuses = copy.deepcopy(self.statuses)
        statuses[0]["retrieved"] = 2
        with self.assertRaisesRegex(ValueError, "tie order"):
            self.evaluate(rankings=ranks, statuses=statuses)

    def test_qrels_reader_rejects_partial_or_duplicate_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "qrels.tsv"
            for body in ("q1\ta\t1\n", "q1\ta\t1\nq1\ta\t1\nq2\tb\t1\n"):
                path.write_text("query-id\tcorpus-id\tscore\n" + body, encoding="utf-8")
                with self.subTest(body=body), self.assertRaises(ValueError):
                    read_qrels(path, self.qids)


class SelectionTests(unittest.TestCase):
    def test_selection_uses_noisy_mean_instead_of_clean_winner(self):
        rows = []
        for model, clean, noisy in (("vsm", .9, .2), ("bm25", .8, .4)):
            rows.extend(dict(branch="word", model=model, condition=condition,
                             **{"ndcg@10": clean if condition == "clean" else noisy})
                        for condition in CONDITIONS)
        self.assertEqual(select_configuration(rows)[0]["model"], "bm25")
        with self.assertRaises(ValueError):
            select_configuration(rows[:-1])


class ExperimentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.output_capture = redirect_stdout(io.StringIO())
        self.output_capture.__enter__()
        self.addCleanup(self.output_capture.__exit__, None, None, None)

    def fixture(self, root):
        data, splits, indexes = [root / name for name in ("data", "splits", "index")]
        data.mkdir()
        (splits / "val/noise").mkdir(parents=True)
        docs = [dict(id="d1", text="xe", word_terms=["xe"]),
                dict(id="d2", text="tàu", word_terms=["tàu"]),
                dict(id="d3", text="máy", word_terms=["máy"])]
        corpus = data / "corpus.jsonl"
        corpus.write_text("".join(json.dumps(row) + "\n" for row in docs), encoding="utf-8")
        queries = [dict(id="q1", text="xe"), dict(id="q2", text="qqq")]
        for condition in CONDITIONS:
            path = splits / "val" / ("queries_clean.jsonl" if condition == "clean"
                                     else f"noise/queries_{condition}.jsonl")
            path.write_text("".join(json.dumps(row) + "\n" for row in queries), encoding="utf-8")
        (splits / "val/qrels_val.tsv").write_text(
            "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t1\nq2\td3\t1\n", encoding="utf-8")
        for branch in ("word", "char_ngram"):
            directory = indexes / branch
            directory.mkdir(parents=True)
            terms = sorted({term for row in docs for term in
                            (row["word_terms"] if branch == "word" else char_terms(row["text"]))})
            vocab = {term: i for i, term in enumerate(terms)}
            counts = to_sparse(docs, vocab, branch)
            tfidf, idf = fit_corpus(counts)
            sparse.save_npz(directory / "counts.npz", counts)
            sparse.save_npz(directory / "tfidf.npz", tfidf)
            np.save(directory / "idf.npy", idf)
            np.save(directory / "doc_lengths.npy", np.asarray(counts.sum(axis=1)).ravel())
            write_json(directory / "doc_ids.json", [row["id"] for row in docs])
            write_json(directory / "vocab.json", vocab)
            write_json(directory / "index_meta.json", dict(branch=branch, full_corpus=True,
                raw_corpus_sha256=sha256(corpus), term_config=dict(stopwords=[], ngram_range=[3, 5])))
        return data, splits, indexes

    def test_48_runs_exports_empty_queries_resumes_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, splits, indexes = self.fixture(root)
            out = root / "results"
            protocol = {**PROTOCOL, "svd_dim": 1}
            args = (indexes, data, splits, out)
            rows = run(*args, protocol=protocol)
            self.assertEqual(len(rows), 48)
            for row in rows:
                self.assertEqual(row["queries"], 2)
                self.assertGreaterEqual(row["empty_queries"], 1)
                self.assertEqual(row["ndcg@10_drop_abs"], 0)
            hashes = {p: sha256(p) for p in out.rglob("rankings.jsonl")}
            self.assertEqual(run(*args, resume=True, protocol=protocol), rows)
            self.assertEqual(hashes, {p: sha256(p) for p in hashes})
            with self.assertRaises(FileExistsError):
                run(*args, protocol=protocol)
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                run(*args, resume=True, protocol={**protocol, "bm25_k1": 1.5})
            ranking = next(iter(hashes))
            ranking.write_text("corrupted\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                run(*args, resume=True, protocol=protocol)
            with self.assertRaisesRegex(ValueError, "frozen validation"):
                run(indexes, data, splits, root / "test", split="test")

    @unittest.skipUnless(find_spec("matplotlib"),
                         "Install configs/evaluation-requirements.txt for plot rendering")
    def test_complete_report_exports_plots_paired_metrics_and_frozen_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, splits, indexes = self.fixture(root)
            out, report_dir = root / "results", root / "report"
            run(indexes, data, splits, out, protocol={**PROTOCOL, "svd_dim": 1})
            report(out, report_dir, data, splits,
                   Path(__file__).resolve().parents[1] / "configs/noise.json")
            self.assertTrue((report_dir / "mixed_noise.png").is_file())
            selection = json.loads((report_dir / "selection.json").read_text(encoding="utf-8"))
            self.assertEqual(selection["completed_runs"], 48)
            with (report_dir / "paired_drops.csv").open(encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 84)


if __name__ == "__main__":
    unittest.main()
