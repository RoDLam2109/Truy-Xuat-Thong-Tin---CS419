"""Tách từ và fit vocabulary/TF-IDF trên corpus, không fit trên query."""

from collections import Counter
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
import unicodedata

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from src.data_io import read_records, read_qrels, sha256, write_json, write_jsonl


VERSION = "1.0.0"
# \w không bao gồm dấu câu; loại _ khỏi thành phần để chỉ giữ _ nối các tiếng.
TERM_PATTERN = re.compile(r"[^\W_]+(?:_[^\W_]+)*", re.UNICODE)


def identity(tokens):
    return tokens


def validate_config(config):
    required = {
        "representation", "segmenter", "unicode_normalization", "lowercase",
        "keep_numbers", "stopwords", "min_df", "max_df", "tfidf",
    }
    if set(config) != required:
        raise ValueError(f"Config cần đúng các khóa: {sorted(required)}")
    if (config["representation"], config["segmenter"], config["unicode_normalization"]) != ("word", "pyvi", "NFC"):
        raise ValueError("Phiên bản này hỗ trợ word + pyvi + NFC")
    if any(type(config[key]) is not bool for key in ("lowercase", "keep_numbers")):
        raise ValueError("lowercase và keep_numbers phải là boolean")
    if type(config["min_df"]) is not int or config["min_df"] < 1:
        raise ValueError("min_df phải là số nguyên >= 1")
    maximum = config["max_df"]
    if isinstance(maximum, bool) or not isinstance(maximum, (int, float)) or not 0 < maximum <= 1:
        raise ValueError("max_df là tỷ lệ trong (0, 1]")
    words = config["stopwords"]
    if not isinstance(words, list) or any(not isinstance(t, str) or TERM_PATTERN.fullmatch(t) is None for t in words):
        raise ValueError("stopwords phải là danh sách term, từ nhiều tiếng dùng dấu _")
    expected = {"norm", "use_idf", "smooth_idf", "sublinear_tf"}
    tfidf = config["tfidf"]
    if not isinstance(tfidf, dict) or set(tfidf) != expected or tfidf["norm"] not in ("l1", "l2", None):
        raise ValueError("Cấu hình tfidf không hợp lệ")
    if any(type(tfidf[key]) is not bool for key in expected - {"norm"}):
        raise ValueError("Các tùy chọn TF-IDF phải là boolean")


class VietnameseTerms:
    def __init__(self, config):
        validate_config(config)
        self.config = config
        self.stopwords = {self.normalize(t) for t in config["stopwords"]}

    def normalize(self, text):
        text = unicodedata.normalize("NFC", text)
        if self.config["lowercase"]:
            text = text.lower()
        return " ".join(text.split())

    def __call__(self, text):
        from pyvi import ViTokenizer

        # _ trong văn bản gốc là ranh giới; _ do PyVi sinh ra mới nối tiếng.
        text = self.normalize(text.replace("_", " "))
        if not text:
            return []
        segmented = ViTokenizer.tokenize(text)
        terms = TERM_PATTERN.findall(segmented)
        return [t for t in terms if t not in self.stopwords and
                (self.config["keep_numbers"] or not any(c.isdigit() for c in t))]


def token_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)["tokens"]


def dependency_versions():
    return {name: version(name) for name in ("pyvi", "scikit-learn", "numpy", "scipy", "joblib", "python-crfsuite", "sklearn-crfsuite")}


def paths_disjoint(output_dir, inputs):
    output_dir = Path(output_dir).resolve()
    for path in inputs:
        resolved = Path(path).resolve()
        if resolved == output_dir or output_dir in resolved.parents:
            raise ValueError("Thư mục đầu ra không được chứa file đầu vào")


def build_index(corpus_path, queries_path, config_path, output_dir, qrels_path=None):
    corpus_path, queries_path, config_path, output_dir = map(Path, (corpus_path, queries_path, config_path, output_dir))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    extractor = VietnameseTerms(config)
    inputs = [corpus_path, queries_path, config_path]
    if qrels_path is not None:
        inputs.append(Path(qrels_path))
    paths_disjoint(output_dir, inputs)
    if output_dir.exists():
        raise ValueError("Thư mục output đã tồn tại; chọn đường dẫn mới để không ghi đè run")
    hashes_before = {str(p): sha256(p) for p in inputs}
    # Kiểm tra schema, ID, qrels trước khi tạo artifact.
    doc_ids = [row["id"] for row in read_records(corpus_path, corpus=True)]
    query_ids = [row["id"] for row in read_records(queries_path)]
    if qrels_path is not None:
        read_qrels(qrels_path, set(query_ids), set(doc_ids))
    output_dir.mkdir(parents=True)
    write_json(output_dir / "config.json", config)
    df = Counter()
    document_tokens = 0
    empty_documents = 0

    def documents():
        nonlocal document_tokens, empty_documents
        for index, row in enumerate(read_records(corpus_path, corpus=True), 1):
            # Tách riêng hai trường tránh ghép từ qua ranh giới title/text.
            tokens = extractor(row.get("title", "")) + extractor(row["text"])
            df.update(set(tokens))
            document_tokens += len(tokens)
            empty_documents += not tokens
            if index % 5000 == 0:
                print(f"Đã tách từ {index}/{len(doc_ids)} tài liệu", flush=True)
            yield {"id": row["id"], "tokens": tokens}

    write_jsonl(output_dir / "corpus.terms.jsonl", documents())
    terms = sorted(t for t, count in df.items() if config["min_df"] <= count <= config["max_df"] * len(doc_ids))
    if not terms:
        raise ValueError("Không có term sau lọc; run chưa hoàn thành (không có manifest.json)")
    vocabulary = {term: i for i, term in enumerate(terms)}
    vectorizer = CountVectorizer(analyzer=identity, vocabulary=vocabulary, dtype=np.int64)
    counts = vectorizer.transform(token_rows(output_dir / "corpus.terms.jsonl")).tocsr()
    transformer = TfidfTransformer(**config["tfidf"])
    tfidf = transformer.fit_transform(counts).tocsr()
    sparse.save_npz(output_dir / "counts.npz", counts)
    sparse.save_npz(output_dir / "tfidf.npz", tfidf)
    np.save(output_dir / "doc_lengths.npy", np.asarray(counts.sum(axis=1)).ravel())
    write_json(output_dir / "doc_ids.json", doc_ids)
    write_json(output_dir / "vocabulary.json", vocabulary)
    # Đây là metadata từ corpus cho Vũ; không phải công thức IDF của BM25.
    with (output_dir / "term_df.tsv").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("term\tdf\tkept\n")
        for term in sorted(df):
            stream.write(f"{term}\t{df[term]}\t{int(term in vocabulary)}\n")
    np.save(output_dir / "idf.npy", transformer.idf_ if config["tfidf"]["use_idf"] else np.ones(len(terms)))
    joblib.dump(transformer, output_dir / "tfidf_transformer.joblib")

    # Query sạch chỉ transform, không fit và không thêm term vào vocabulary.
    query_stats = []
    def queries():
        for row in read_records(queries_path):
            tokens = extractor(row["text"])
            kept = sum(t in vocabulary for t in tokens)
            query_stats.append({"id": row["id"], "tokens": len(tokens), "oov_tokens": len(tokens) - kept,
                                "empty_vector": kept == 0})
            yield {"id": row["id"], "tokens": tokens}

    write_jsonl(output_dir / "queries_clean.terms.jsonl", queries())
    query_counts = vectorizer.transform(token_rows(output_dir / "queries_clean.terms.jsonl")).tocsr()
    sparse.save_npz(output_dir / "queries_clean.counts.npz", query_counts)
    sparse.save_npz(output_dir / "queries_clean.tfidf.npz", transformer.transform(query_counts))
    write_json(output_dir / "query_ids.json", query_ids)
    write_jsonl(output_dir / "query_stats.jsonl", query_stats)
    total = sum(row["tokens"] for row in query_stats)
    oov = sum(row["oov_tokens"] for row in query_stats)
    summary = {
        "documents": len(doc_ids), "queries": len(query_ids),
        "document_tokens_before_df_filter": document_tokens,
        "empty_documents_before_df_filter": empty_documents,
        "empty_documents_after_df_filter": int(np.count_nonzero(np.diff(counts.indptr) == 0)),
        "vocabulary_before_df_filter": len(df), "vocabulary_size": len(terms),
        "count_shape": list(counts.shape), "count_nnz": counts.nnz,
        "query_tokens": total, "query_oov_tokens": oov,
        "query_oov_rate": oov / total if total else 0.0,
        "empty_queries": sum(row["empty_vector"] for row in query_stats),
        "document_length_definition": "sum of term counts after stopword and DF filtering; includes repetitions",
    }
    write_json(output_dir / "stats.json", summary)
    if any(sha256(path) != value for path, value in hashes_before.items()):
        raise ValueError("File đầu vào thay đổi trong khi chạy; run chưa hoàn thành")
    source_files = [Path(__file__), Path(__file__).parents[1] / "data_io.py"]
    write_json(output_dir / "manifest.json", {
        "version": VERSION, "fit_source": "corpus_only", "query_condition": "clean",
        "dependencies": dependency_versions(),
        "input_sha256": hashes_before,
        "code_sha256": {p.name: sha256(p) for p in source_files},
        "artifacts_sha256": {p.name: sha256(p) for p in sorted(output_dir.iterdir()) if p.is_file()},
        "stats": summary,
    })
    return summary


class TermIndex:
    """Nạp artifact đáng tin cậy để transform query; không có thao tác fit."""

    def __init__(self, directory):
        self.directory = Path(directory)
        manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["version"] != VERSION:
            raise ValueError("Phiên bản chỉ mục không được hỗ trợ")
        if manifest["dependencies"] != dependency_versions():
            raise ValueError("Dependency khác bản build; dùng môi trường đã ghi trong manifest")
        for name, expected in manifest["artifacts_sha256"].items():
            if sha256(self.directory / name) != expected:
                raise ValueError(f"Artifact bị thay đổi: {name}")
        self.extractor = VietnameseTerms(json.loads((self.directory / "config.json").read_text(encoding="utf-8")))
        self.vocabulary = json.loads((self.directory / "vocabulary.json").read_text(encoding="utf-8"))
        self.doc_ids = json.loads((self.directory / "doc_ids.json").read_text(encoding="utf-8"))
        self.vectorizer = CountVectorizer(analyzer=identity, vocabulary=self.vocabulary, dtype=np.int64)
        # joblib chỉ dùng với artifact tự build / nguồn tin cậy.
        self.tfidf = joblib.load(self.directory / "tfidf_transformer.joblib")

    def transform(self, texts):
        tokens = [self.extractor(text) for text in texts]
        counts = self.vectorizer.transform(tokens).tocsr()
        return counts, self.tfidf.transform(counts).tocsr()
