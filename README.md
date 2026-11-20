# Evaluating the Robustness of Information Retrieval Models under Noisy Vietnamese Queries

## 1. Mục tiêu dự án

Đánh giá mức độ chống chịu của các mô hình truy xuất thông tin khi câu truy vấn tiếng Việt bị nhiễu. Với mỗi mô hình và cách biểu diễn term, lấy kết quả trên query sạch làm điểm chuẩn, chạy các phiên bản query nhiễu, rồi tính **mức suy giảm hiệu năng**.

Dataset đã chốt là **bản Zalo Legal do GreenNode đăng trên Hugging Face**, revision `12d76d4d04b94ceada970fcfbe7fec20bfa97389`. Trong toàn bộ thí nghiệm, chỉ thay đổi câu truy vấn; giữ nguyên corpus, qrels và tập ID truy vấn.

Dự án khảo sát hai yếu tố:

1. **Cách biểu diễn term:** từ tiếng Việt và character n-gram.
2. **Mô hình truy xuất:** VSM với TF-IDF/cosine similarity, BM25 và LSI/SVD.

Các câu hỏi nghiên cứu:

- Mỗi mô hình suy giảm bao nhiêu so với điểm sạch khi query có từng loại lỗi?
- Mức suy giảm thay đổi thế nào khi mức nhiễu hỗn hợp tăng?
- Biểu diễn bằng character n-gram có giảm tác động của lỗi ký tự so với biểu diễn bằng từ tiếng Việt không?
- Mô hình nào cân bằng tốt giữa chất lượng truy xuất và khả năng chống chịu nhiễu?

## 2. Dữ liệu và schema

Dataset chính: **bản Lam v1**, đặt tại `data/zalo_legal/`. Nguồn: [GreenNode/zalo-ai-legal-text-retrieval-vn](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn), revision `12d76d4d04b94ceada970fcfbe7fec20bfa97389`.

Chi tiết nguồn, số lượng và SHA-256: [train_manifest.json](data/zalo_legal/train_manifest.json). Hướng dẫn chia sẻ dữ liệu: [data/README.md](data/README.md).

Corpus, query và qrels đều đối chiếu theo cùng revision GreenNode trong `configs/dataset.json`. Giữ đầy đủ nhãn của mỗi query, kể cả query có nhiều tài liệu liên quan; không ghép query/qrels từ bản phân phối khác khi chưa đối chiếu ID và nội dung. Các file nhiễu là biến thể tổng hợp do dự án tạo, không phải dữ liệu lỗi thực do GreenNode cung cấp.

| Thành phần | Đường dẫn | Số lượng hiện tại |
|---|---|---:|
| Query sạch (test) | `data/zalo_legal/queries_clean.jsonl` | 788 query, 788 ID duy nhất |
| Query train | `data/zalo_legal/queries_train.jsonl` | 2.408 query, không trùng ID/nội dung chuẩn hóa với test |
| Nhãn train | `data/zalo_legal/qrels_train.tsv` | 2.480 quan hệ liên quan |


## 3. Development / evaluation (branch feature/model-evaluation)

The canonical Lam v1 files remain unchanged. From GreenNode's 2,408 train queries,
the deterministic text-group split (seed 42) produces **1,926 development train**
and **482 validation** queries (1,985 and 495 qrels), with seven synthetic
validation noise sets. **Test remains 788 queries/793 qrels** and is not used
for hyperparameter tuning.

The model pipeline reuses PREPROCESSING.md and scripts/prepare_terms.py from main:
two term representations (word and character n-gram) x VSM / BM25 / LSI,
each under clean + 7 noise conditions. Only corpus-derived vocab and IDF are used.

Run in this order (see docs/model_evaluation.md for details):

    python -m uv sync --locked
    python -X utf8 scripts/prepare_train.py --download
    python -X utf8 scripts/prepare_validation.py
    python -X utf8 scripts/prepare_terms.py --workers 4
    python -X utf8 scripts/build_retrieval_index.py --branch word --output-dir outputs/index/word
    python -X utf8 scripts/evaluate_models.py --index-dir outputs/index/word --output-dir outputs/retrieval/val_word

The test must not be used for selecting parameters. All generated data,
indices, validation noise and experiment results live under git-ignored outputs/.
Every index and run is bound to the input SHA-256 in its metadata.
