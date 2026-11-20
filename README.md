# Evaluating the Robustness of Information Retrieval Models under Noisy Vietnamese Queries

Đồ án môn **Truy xuất thông tin (CS419)**: so sánh chất lượng truy xuất và mức suy giảm hiệu năng của VSM, BM25, LSI khi câu truy vấn tiếng Việt bị nhiễu. Thí nghiệm sử dụng hai cách biểu diễn văn bản — **word** và **character n-gram** — trên cùng tập văn bản pháp luật.

**Trạng thái hiện tại:** đã chạy **32/48 cấu hình** trên toàn bộ corpus và tập validation. Chưa đánh giá test cuối cùng.

## Dữ liệu và cách chia tập

Dữ liệu sử dụng: [GreenNode/zalo-ai-legal-text-retrieval-vn](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn), đã chuẩn hóa và kiểm tra trùng lặp train/test. Revision và SHA-256 được lưu trong [configs/dataset.json](configs/dataset.json) và [train_manifest.json](data/zalo_legal/train_manifest.json).

| Thành phần | Queries | Qrels | Ghi chú |
|---|---:|---:|---|
| Corpus | — | — | 61.425 tài liệu, dùng chung cho mọi split |
| Train chuẩn hóa | 2.408 | 2.480 | Đã loại 24 query IDs trùng test |
| Development train | 1.926 | 1.985 | Tách từ train chuẩn hóa, seed 42 |
| Validation | 482 | 495 | Dùng chọn tham số và đánh giá trong quá trình phát triển |
| Test | 788 | 793 | Giữ nguyên để đánh giá cuối cùng |

Train/validation được chia theo nhóm nội dung query chuẩn hóa, tránh một query xuất hiện ở cả hai tập. Mỗi điều kiện validation có cùng 482 query IDs và cùng 495 qrels. Corpus không bị chia nhỏ và không thêm nhiễu.

Bảy điều kiện nhiễu tổng hợp: `no_diacritics`, `telex_typo`, `phonetic`, `abbrev`, `mixed_low`, `mixed_medium`, `mixed_high`. Điều kiện `clean` dùng query chưa được thêm nhiễu.

Xem [cấu trúc dữ liệu và hướng dẫn chuẩn bị](data/README.md).

## Phương pháp

| Thành phần | Thiết lập |
|---|---|
| Word | Tách từ bằng PyVi; giữ dấu, số và từ phủ định |
| Character n-gram | N-gram 3–5 ký tự trong từng token |
| Vocabulary | Xây dựng từ corpus; `min_df=2`, `max_df=0.95` |
| VSM | TF-IDF + cosine similarity |
| BM25 | `k1=1.2`, `b=0.75` |
| LSI | TruncatedSVD, 64 chiều, seed 42 |
| Chỉ số | MRR@10, nDCG@10, Recall@100 và mức suy giảm so với clean |

Vocabulary, IDF và phép chiếu SVD được fit **chỉ từ corpus**; qrels không được dùng để fit các mô hình này. Các tham số trên là baseline cố định, chưa phải cấu hình tối ưu. Chi tiết preprocessing nằm trong [PREPROCESSING.md](PREPROCESSING.md); thiết kế đánh giá và lệnh chạy lại nằm trong [docs/model_evaluation.md](docs/model_evaluation.md).

## Kết quả validation

**nDCG@10**, đánh giá trên 482 queries và toàn bộ 61.425 documents.

| Điều kiện | Word VSM | Word BM25 | Word LSI | Char VSM |
|---|---:|---:|---:|---:|
| Clean | 0.5008 | **0.6626** | 0.2420 | 0.4833 |
| No diacritics | 0.0209 | **0.0297** | 0.0019 | 0.0219 |
| Telex typo | 0.3263 | **0.4725** | 0.1376 | 0.4216 |
| Phonetic | 0.4925 | **0.6511** | 0.2325 | 0.4803 |
| Abbreviation | 0.5004 | **0.6589** | 0.2358 | 0.4785 |
| Mixed low | 0.4198 | **0.5875** | 0.1914 | 0.4528 |
| Mixed medium | 0.3119 | **0.4698** | 0.1204 | 0.3996 |
| Mixed high | 0.2284 | **0.3534** | 0.0803 | 0.3542 |

**Nhận xét:** BM25 word đạt điểm clean cao nhất trong các cấu hình đã chạy (nDCG@10 = 0.6626). Character n-gram cải thiện VSM khi gặp lỗi Telex (0.4216 so với 0.3263) và nhiễu hỗn hợp nặng (0.3542 so với 0.2284), nhưng gần như không xử lý được trường hợp bỏ dấu. MRR@10, Recall@100 và đầy đủ kết quả được ghi trong [docs/validation_results.md](docs/validation_results.md).

**Tiến độ:** word VSM/BM25/LSI **24/24**; character n-gram VSM **8/8**; character n-gram BM25 và LSI **0/16**. Tổng cộng **32/48**. Chưa có số liệu test cuối cùng.

## Chạy lại

Cần **Python 3.11+**, [uv](https://docs.astral.sh/uv/) và bộ dữ liệu nhóm đã chuẩn bị. Clone repository **không kèm corpus và các query JSONL**: tải folder dữ liệu từ [Google Drive của nhóm](https://drive.google.com/drive/folders/1fdZjJ1i7o44f3jNwbZL2okPxKcNfkv7H) vào `data/zalo_legal/` trước khi chạy.

Chạy trong thư mục gốc repository:

```bash
uv sync --locked
uv run python -X utf8 scripts/prepare_validation.py
uv run python -X utf8 scripts/prepare_terms.py --workers 4
uv run python -X utf8 scripts/build_retrieval_index.py --branch word --output-dir outputs/index/word
uv run python -X utf8 scripts/evaluate_models.py --index-dir outputs/index/word --output-dir outputs/retrieval/val_word
```

Để thử nghiệm character n-gram, thay `word` bằng `char_ngram` ở bước xây dựng index và truyền đường dẫn index tương ứng khi đánh giá. Lệnh kiểm tra unit tests:

```bash
uv run python -m unittest discover -s tests -q
```

Các script tạo split, index và kết quả đánh giá không ghi đè thư mục đầu ra đã tồn tại. Muốn chạy lại, chọn thư mục đầu ra mới hoặc chủ động dọn artifact cũ. Tất cả artifact trong `outputs/` được bỏ qua bởi Git; xem [hướng dẫn chạy và kiểm tra dữ liệu](docs/model_evaluation.md).

## Giới hạn và công việc tiếp theo

- Hoàn thành **16 cấu hình validation** còn thiếu: BM25 và LSI trên character n-gram.
- Phân tích lỗi bỏ dấu, đánh giá các cải tiến trên **validation** trước khi sử dụng test.
- Kiểm tra thủ công một số mẫu nhiễu vì phép biến đổi tổng hợp có thể làm thay đổi nghĩa câu hỏi.
- Chỉ báo cáo test sau khi chốt mô hình và tham số. Tập test từng được khảo sát trong giai đoạn trước, vì vậy không mô tả đây là một test set hoàn toàn chưa từng được xem.

Mã nguồn, phương pháp và kết quả hiện có được trình bày ở đây; các số liệu chưa đo không được suy diễn từ smoke test.
