"""Chuẩn hóa Lam v1 và chọn vocabulary từ corpus, không dùng nhãn test."""
import argparse
from collections import Counter
from contextlib import nullcontext
from functools import partial
from multiprocessing import Pool
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import unicodedata

ROOT = Path(__file__).resolve().parents[1]


def normalize(text):
    """Giữ dấu, số và dấu câu; chỉ NFC, chữ thường và gộp khoảng trắng."""
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def word_terms(text, stopwords):
    from pyvi import ViTokenizer
    # Tách từ trước khi bỏ token chỉ có dấu câu. Giữ các từ phủ định và số.
    return [t for t in ViTokenizer.tokenize(text).split()
            if any(c.isalnum() for c in t) and t not in stopwords]


def char_terms(text, ngram_range=(3, 5)):
    """N-gram có ký tự biên khoảng trắng, không đi xuyên qua từ cách trắng."""
    for token in text.split():
        padded = f" {token} "
        for n in range(ngram_range[0], ngram_range[1] + 1):
            for i in range(len(padded) - n + 1):
                yield padded[i:i + n]


def terms_for(row, branch, vocabulary, ngram_range=(3, 5)):
    """Dùng lại khi tạo ma trận count/TF-IDF: giữ thứ tự và số lần lặp term."""
    if branch not in {"word", "char_ngram"}:
        raise ValueError("branch phải là word hoặc char_ngram")
    candidates = row["word_terms"] if branch == "word" else char_terms(row["text"], ngram_range)
    return [term for term in candidates if term in vocabulary]


def read_rows(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_row(row, corpus, stopwords):
    raw = row["title"] + "\n" + row["text"] if corpus else row["text"]
    text = normalize(raw)
    return {"id": row["id"], "text": text, "word_terms": word_terms(text, stopwords)}


def prepare(input_dir, output_dir, config, workers=1):
    if input_dir.resolve() == output_dir.resolve() or input_dir.resolve() in output_dir.resolve().parents:
        raise ValueError("Output phải nằm ngoài thư mục dữ liệu gốc")
    output_dir.mkdir(parents=True, exist_ok=True)
    stopwords = set(config["stopwords"])
    dfs = {"word": Counter(), "char_ngram": Counter()}
    inputs = [input_dir / "corpus.jsonl", input_dir / "queries_train.jsonl",
              input_dir / "queries_clean.jsonl", *sorted((input_dir / "noise").glob("queries_*.jsonl"))]
    assert len(inputs) == 10, "Cần corpus, train, test gốc và 7 file nhiễu"
    stats = {}
    input_hashes = {str(p.relative_to(input_dir)): sha256(p) for p in inputs}
    with Pool(workers) if workers > 1 else nullcontext() as pool:
        for source in inputs:
            relative = source.relative_to(input_dir)
            target = output_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            count = 0
            seen = set()
            with target.open("w", encoding="utf-8") as out:
                transform = partial(prepare_row, corpus=source.name == "corpus.jsonl", stopwords=stopwords)
                rows = pool.imap(transform, read_rows(source), chunksize=16) if pool else map(transform, read_rows(source))
                for row in rows:
                    if row["id"] in seen:
                        raise ValueError(f"ID trùng trong {source}: {row['id']}")
                    seen.add(row["id"])
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if source.name == "corpus.jsonl":
                        dfs["word"].update(set(row["word_terms"]))
                        dfs["char_ngram"].update(set(char_terms(row["text"], config["ngram_range"])))
                    count += 1
                    if count % 5000 == 0:
                        print(f"{relative}: {count:,} dòng", flush=True)
            stats[str(relative)] = count
            print(f"Hoàn thành {relative}: {count:,} dòng", flush=True)
    n_docs = stats["corpus.jsonl"]
    vocab_stats = {}
    for branch, df in dfs.items():
        selected = {term: freq for term, freq in sorted(df.items())
                    if config["min_df"] <= freq <= config["max_df"] * n_docs}
        write_json(output_dir / f"vocabulary_{branch}.json", selected)
        vocab_stats[branch] = {"before_filter": len(df), "selected": len(selected)}
    for name in ["qrels_train.tsv", "qrels_test.tsv"]:
        shutil.copyfile(input_dir / name, output_dir / name)
        input_hashes[name] = sha256(input_dir / name)
    assert all(sha256(input_dir / name) == digest for name, digest in input_hashes.items()), "Dữ liệu gốc đã thay đổi"
    report = {"dataset": "Lam v1", "normalization": "NFC + lower + collapse whitespace; keep accents, numbers, punctuation",
              "tokenizer": f"pyvi=={version('pyvi')}", "config": config,
              "rows": stats, "vocabulary": vocab_stats, "source_sha256": input_hashes}
    write_json(output_dir / "summary.json", report)
    write_json(output_dir / "config_used.json", config)
    print(json.dumps(vocab_stats, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/zalo_legal")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/processed")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/terms.json")
    parser.add_argument("--workers", type=int, default=4, help="Số tiến trình tách từ; dùng 1 nếu ít RAM")
    args = parser.parse_args()
    prepare(args.input_dir, args.output_dir, json.loads(args.config.read_text(encoding="utf-8")), args.workers)
