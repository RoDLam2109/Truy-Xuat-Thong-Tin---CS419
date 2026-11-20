"""Đối chiếu toàn bộ dataset với nguồn GreenNode cố định; không sửa dữ liệu.

--download chỉ tải nguồn gốc còn thiếu vào raw/. Báo cáo và manifest
được tạo sau khi kiểm tra; corpus/query/qrels/noise không được ghi lại.
"""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

try:
    from scripts import generate_noise as noise, prepare_train as train
except ModuleNotFoundError:  # python scripts/check_dataset.py
    import generate_noise as noise
    import prepare_train as train


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (*noise.KINDS, "mixed_low", "mixed_medium", "mixed_high")


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_sources(raw_dir, config, download):
    for name, expected in config["source_sha256"].items():
        path = raw_dir / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Thiếu {path}; chạy lại với --download")
            path.parent.mkdir(parents=True, exist_ok=True)
            url = (f"https://huggingface.co/datasets/{config['dataset']}/resolve/"
                   f"{config['revision']}/{name}")
            # Tên tạm ngẫu nhiên; chỉ thay file nguồn sau khi checksum hợp lệ.
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                temporary = Path(output.name)
                try:
                    with urllib.request.urlopen(url, timeout=60) as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            output.write(block)
                except Exception:
                    output.close()
                    temporary.unlink(missing_ok=True)
                    raise
            if checksum(temporary) != expected:
                temporary.unlink()
                raise ValueError(f"Checksum tải về không khớp: {name}")
            temporary.replace(path)
        if checksum(path) != expected:
            raise ValueError(f"Checksum nguồn không khớp: {path}")


def read_qrels(path):
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames != ["query-id", "corpus-id", "score"]:
            raise ValueError(f"Schema qrels sai: {path}")
        rows = list(reader)
    for row in rows:
        if set(row) != {"query-id", "corpus-id", "score"} or row["score"] != "1":
            raise ValueError(f"Nhãn không hợp lệ: {path}")
        row["score"] = 1
    return rows


def audit(data_dir, report_dir, config, noise_config_path, download=False):
    raw = data_dir / "raw"
    ensure_sources(raw, config, download)
    checks = {}

    def check(name, passed):
        checks[name] = bool(passed)

    source_corpus = [
        {"id": row["_id"], "title": row["title"], "text": row["text"]}
        for row in train.read_jsonl(raw / "corpus.jsonl")
    ]
    corpus = list(train.read_jsonl(data_dir / "corpus.jsonl"))
    doc_ids = {row["id"] for row in corpus}
    check("corpus_matches_source_content_and_order", corpus == source_corpus)
    check("corpus_unique_ids", len(corpus) == len(doc_ids))
    check("corpus_schema_and_content", all(
        set(row) == {"id", "title", "text"}
        and all(isinstance(row[key], str) for key in ("id", "title", "text"))
        and row["id"].strip() and (row["title"] + "\n" + row["text"]).strip()
        for row in corpus
    ))
    source_queries = list(train.read_jsonl(raw / "queries.jsonl"))
    source_test_qrels = list(train.read_jsonl(raw / "qrels/test.jsonl"))
    source_train_qrels = list(train.read_jsonl(raw / "qrels/train.jsonl"))
    # Lấy query test theo ID trong nhãn test nguồn; kiểm tra mọi bản sao cùng nội dung.
    source_by_id = {}
    for row in source_queries:
        qid, text = row["_id"], row["text"]
        if qid in source_by_id and source_by_id[qid] != text:
            raise ValueError(f"Query nguồn mâu thuẫn nội dung: {qid}")
        source_by_id[qid] = text
    expected_test = {row["query-id"]: source_by_id[row["query-id"]] for row in source_test_qrels}
    test = noise.read_queries(data_dir / "queries_clean.jsonl")
    training = noise.read_queries(data_dir / "queries_train.jsonl")
    test_qrels = read_qrels(data_dir / "qrels_test.tsv")
    train_qrels = read_qrels(data_dir / "qrels_train.tsv")
    check("test_queries_match_source_ids_and_text", {r["id"]: r["text"] for r in test} == expected_test)
    check("test_qrels_match_source_all_labels", test_qrels == source_test_qrels)
    expected_train, expected_train_qrels, train_report = train.prepare_rows(
        source_queries, source_train_qrels, test, doc_ids
    )
    check("train_queries_match_source_after_test_exclusion", training == expected_train)
    check("train_qrels_match_source_all_labels_after_test_exclusion", train_qrels == expected_train_qrels)
    test_ids, train_ids = {r["id"] for r in test}, {r["id"] for r in training}
    check("train_test_ids_disjoint", not test_ids & train_ids)
    check("train_test_normalized_text_disjoint", not (
        {train.text_key(r["text"]) for r in test} & {train.text_key(r["text"]) for r in training}
    ))
    for split, queries, qrels in (("test", test, test_qrels), ("train", training, train_qrels)):
        check(f"{split}_query_schema", all(set(r) == {"id", "text"} for r in queries))
        check(f"{split}_qrels_cover_queries", {r["query-id"] for r in qrels} == {r["id"] for r in queries})
        check(f"{split}_qrels_valid_documents", all(r["corpus-id"] in doc_ids for r in qrels))
        check(f"{split}_qrels_unique_pairs", len(qrels) == len({(r["query-id"], r["corpus-id"]) for r in qrels}))
    counts = dict(corpus=len(corpus), test_queries=len(test), test_qrels=len(test_qrels),
                  train_queries=len(training), train_qrels=len(train_qrels))
    for key, expected in config["expected_counts"].items():
        check(f"expected_count_{key}", counts[key] == expected)

    train_manifest = json.loads((data_dir / "train_manifest.json").read_text(encoding="utf-8"))
    check("train_manifest_dataset_revision", (train_manifest["dataset"], train_manifest["revision"]) == (config["dataset"], config["revision"]))
    check("train_manifest_checksums", all(checksum(data_dir / name) == value for name, value in train_manifest["local_sha256"].items()))
    config_noise = json.loads(noise_config_path.read_text(encoding="utf-8"))
    noise.validate_config(config_noise)
    check("noise_seed_and_probabilities", config_noise["seed"] == 42 and config_noise["single_probability"] == .25 and config_noise["mixed_probabilities"] == {"low": .1, "medium": .25, "high": .4})
    stats = json.loads((report_dir / "stats.json").read_text(encoding="utf-8"))
    check("noise_metadata_input", stats["input_sha256"] == checksum(data_dir / "queries_clean.jsonl"))
    check("noise_metadata_qrels", stats["qrels_sha256"] == checksum(data_dir / "qrels_test.tsv"))
    check("noise_metadata_config", stats["config_source_sha256"] == checksum(noise_config_path))
    check("noise_metadata_generator", stats["generator_sha256"] == checksum(Path(noise.__file__)))
    check("noise_config_used", json.loads((report_dir / "config_used.json").read_text(encoding="utf-8")) == config_noise)
    check("exact_seven_noise_files", {p.name for p in (data_dir / "noise").glob("queries_*.jsonl")} == {f"queries_{c}.jsonl" for c in CONDITIONS})
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        args = argparse.Namespace(input=data_dir / "queries_clean.jsonl", qrels=data_dir / "qrels_test.tsv",
                                  config=noise_config_path, output_dir=temp / "noise", report_dir=temp / "reports",
                                  seed=config_noise["seed"], sample_size=10)
        with contextlib.redirect_stdout(io.StringIO()):
            regenerated = noise.generate(args)
        for condition in CONDITIONS:
            path = data_dir / "noise" / f"queries_{condition}.jsonl"
            noisy = noise.read_queries(path)
            check(f"{condition}_ids_and_order", [r["id"] for r in noisy] == [r["id"] for r in test])
            check(f"{condition}_reproducible", path.read_bytes() == (temp / "noise" / path.name).read_bytes())
            check(f"{condition}_checksum", checksum(path) == stats["conditions"][condition]["output_sha256"])
            audit_path = report_dir / f"{condition}.changes.jsonl"
            check(f"{condition}_audit_reproducible", audit_path.read_bytes() == (temp / "reports" / audit_path.name).read_bytes())
            check(f"{condition}_statistics", stats["conditions"][condition] == regenerated["conditions"][condition])
        check("noise_samples_reproducible", (report_dir / "samples.md").read_bytes() == (temp / "reports/samples.md").read_bytes())

    files = [data_dir / name for name in ("corpus.jsonl", "queries_clean.jsonl", "qrels_test.tsv", "queries_train.jsonl", "qrels_train.tsv", "train_manifest.json")]
    files += sorted((data_dir / "noise").glob("queries_*.jsonl"))
    manifest = {"dataset": config["dataset"], "revision": config["revision"],
                "counts": counts, "source_sha256": config["source_sha256"],
                "local_sha256": {p.relative_to(data_dir).as_posix(): checksum(p) for p in files},
                "transformations": ["_id -> id", "test query duplicate IDs removed", "train queries overlapping test removed", "all remaining positive qrels retained", "seven synthetic noise conditions generated with seed 42"]}
    distributions = {split: dict(sorted(Counter(Counter(r["query-id"] for r in qrels).values()).items()))
                     for split, qrels in (("test", test_qrels), ("train", train_qrels))}
    return {"dataset": config["dataset"], "revision": config["revision"], "checks": checks,
            "passed_checks": sum(checks.values()), "total_checks": len(checks),
            "failed_checks": [name for name, passed in checks.items() if not passed], "counts": counts,
            "documents_per_query": distributions, "excluded_train_test_ids": train_report["excluded_test_id_overlap"],
            "corpus_quality": {"empty_text_with_title": sum(not r["text"].strip() and bool(r["title"].strip()) for r in corpus),
                               "duplicate_title_text_extra_rows": len(corpus) - len({(r["title"], r["text"]) for r in corpus}),
                               "replacement_character_ids": [r["id"] for r in corpus if "\ufffd" in r["text"]]},
            "noise": {c: {k: stats["conditions"][c][k] for k in ("changed_queries", "actual_token_noise_rate", "unchanged_queries")} for c in CONDITIONS},
            "semantic_review_status": "Label correctness and noise meaning preservation are not certified by this structural/source audit.",
            "manifest": manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/zalo_legal")
    parser.add_argument("--noise-report-dir", type=Path, default=PROJECT_ROOT / "outputs/noise")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/dataset.json")
    parser.add_argument("--noise-config", type=Path, default=PROJECT_ROOT / "configs/noise.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/greennode_audit.json")
    parser.add_argument("--download", action="store_true", help="Tải nguồn còn thiếu vào raw/ để đối chiếu")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = audit(args.data_dir, args.noise_report_dir, config, args.noise_config, args.download)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not result["failed_checks"]:
        (args.data_dir / "dataset_manifest.json").write_text(json.dumps(result["manifest"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("counts", "passed_checks", "total_checks", "failed_checks")}))
    return bool(result["failed_checks"])


if __name__ == "__main__":
    raise SystemExit(main())
