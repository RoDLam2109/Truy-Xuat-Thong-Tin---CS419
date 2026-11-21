"""Create reproducible train/validation splits from the canonical train set.

Test queries are checked for overlap; canonical inputs are not modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

try:
    from scripts import generate_noise
    from scripts.prepare_train import text_key
except ModuleNotFoundError:
    import generate_noise
    from prepare_train import text_key

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def read_positive_qrels(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["query-id", "corpus-id", "score"]:
            raise ValueError("Invalid qrels header")
        rows = list(reader)
    seen = set()
    for row in rows:
        pair = (row["query-id"], row["corpus-id"])
        if pair in seen or row["score"] != "1":
            raise ValueError(f"Duplicate/non-positive qrel: {pair}")
        seen.add(pair)
    return rows


def split_rows(queries, qrels, test_queries, fraction=0.2, seed=42):
    """Group identical normalized text so it cannot cross train/validation."""
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1")
    query_ids = [row["id"] for row in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate query ID in train")
    test_ids = {row["id"] for row in test_queries}
    test_texts = {text_key(row["text"]) for row in test_queries}
    keys = {row["id"]: text_key(row["text"]) for row in queries}
    if set(query_ids) & test_ids or set(keys.values()) & test_texts:
        raise ValueError("Source train queries overlap with fixed test")
    if {row["query-id"] for row in qrels} != set(query_ids):
        raise ValueError("Every source train query must have positive qrels")
    groups = defaultdict(set)
    for qid, key in keys.items():
        groups[key].add(qid)
    group_keys = sorted(groups)
    if len(group_keys) < 2:
        raise ValueError("Need at least two distinct query groups")
    random.Random(seed).shuffle(group_keys)
    target = max(1, round(len(queries) * fraction))
    validation_ids = set()
    for key in group_keys[:-1]:
        if len(validation_ids) >= target:
            break
        validation_ids.update(groups[key])
    if not validation_ids or len(validation_ids) == len(query_ids):
        raise ValueError("Empty train or validation split")
    train_queries = [r for r in queries if r["id"] not in validation_ids]
    val_queries = [r for r in queries if r["id"] in validation_ids]
    train_qrels = [r for r in qrels if r["query-id"] not in validation_ids]
    val_qrels = [r for r in qrels if r["query-id"] in validation_ids]
    assert not ({keys[r["id"]] for r in train_queries}
                & {keys[r["id"]] for r in val_queries})
    return (train_queries, train_qrels), (val_queries, val_qrels)


def write_qrels(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["query-id", "corpus-id", "score"],
                                delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def create(data_dir: Path, output_dir: Path, noise_config: Path,
           fraction=0.2, seed=42, with_noise=True):
    data_dir, output_dir = data_dir.resolve(), output_dir.resolve()
    if output_dir == data_dir or data_dir in output_dir.parents:
        raise ValueError("Cannot write split into the canonical data directory")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing split: {output_dir}")
    source = {
        "queries_train.jsonl": data_dir / "queries_train.jsonl",
        "qrels_train.tsv": data_dir / "qrels_train.tsv",
        "queries_clean.jsonl": data_dir / "queries_clean.jsonl",
        "qrels_test.tsv": data_dir / "qrels_test.tsv",
    }
    original = {k: sha256(v) for k, v in source.items()}
    queries = generate_noise.read_queries(source["queries_train.jsonl"])
    qrels = read_positive_qrels(source["qrels_train.tsv"])
    test_queries = generate_noise.read_queries(source["queries_clean.jsonl"])
    train, val = split_rows(queries, qrels, test_queries, fraction, seed)
    for dirname, (split_queries, split_qrels) in (("train", train), ("val", val)):
        directory = output_dir / dirname
        directory.mkdir(parents=True, exist_ok=False)
        generate_noise.write_jsonl(directory / "queries_clean.jsonl", split_queries)
        write_qrels(directory / f"qrels_{dirname}.tsv", split_qrels)
    if with_noise:
        generate_noise.generate(SimpleNamespace(
            input=output_dir / "val/queries_clean.jsonl",
            qrels=output_dir / "val/qrels_val.tsv",
            config=noise_config,
            output_dir=output_dir / "val/noise",
            report_dir=output_dir / "val/noise_report",
            seed=seed,
            sample_size=10,
        ))
    manifest = {
        "dataset": "GreenNode Lam v1 (train only)",
        "seed": seed, "validation_fraction": fraction,
        "strategy": "seeded text-group split (Unicode NFC/casefold/collapsed whitespace)",
        "source_sha256": original,
        "train_queries": len(train[0]), "train_qrels": len(train[1]),
        "val_queries": len(val[0]), "val_qrels": len(val[1]),
        "train_multi_positive": dict(Counter(Counter(r["query-id"] for r in train[1]).values())),
        "val_multi_positive": dict(Counter(Counter(r["query-id"] for r in val[1]).values())),
        "test_untouched": True,
        "noise_conditions": 7 if with_noise else 0,
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if original != {k: sha256(v) for k, v in source.items()}:
        raise AssertionError("Canonical source changed during split")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/zalo_legal")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/splits/seed42")
    parser.add_argument("--noise-config", type=Path, default=ROOT / "configs/noise.json")
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-noise", action="store_true")
    args = parser.parse_args()
    result = create(args.data_dir, args.output_dir, args.noise_config,
                    args.fraction, args.seed, not args.no_noise)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
