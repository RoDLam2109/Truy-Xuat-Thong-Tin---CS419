# Thiết kế thí nghiệm và hướng dẫn đánh giá

Tài liệu kỹ thuật cho phần retrieval/evaluation. Kết quả thực nghiệm và trạng thái từng mô hình được ghi trong [validation_results.md](validation_results.md).

## 1. Thiết kế dữ liệu

Dữ liệu đầu vào là bộ Zalo Legal được nhóm chuẩn hóa từ GreenNode; revision và SHA-256 nằm trong [configs/dataset.json](../configs/dataset.json). Corpus gồm **61.425 documents**, dùng chung cho train, validation và test.

| Split | Queries | Qrels | Mục đích |
|---|---:|---:|---|
| Train chuẩn hóa | 2.408 | 2.480 | Nguồn để tạo development split |
| Development train | 1.926 | 1.985 | Dành cho các hướng có supervised training trong tương lai |
| Validation | 482 | 495 | So sánh mô hình, chọn tham số |
| Test | 788 | 793 | Đánh giá sau khi chốt cấu hình |

`scripts/prepare_validation.py` tạo train/validation từ **train chuẩn hóa**, không lấy query test. Seed = 42; các query có cùng nội dung sau Unicode NFC, casefold và chuẩn hóa khoảng trắng được xếp chung một split. Tất cả qrels của một query được giữ nguyên, kể cả query có nhiều tài liệu liên quan.

Bảy tập validation noise được sinh từ 482 query sạch của validation; giữ nguyên query IDs, thứ tự và qrels. Test và noise của test không thay đổi.

## 2. Biểu diễn và mô hình

| Thành phần | Cấu hình |
|---|---|
| Word | PyVi 0.1.1; 5 stopword theo `configs/terms.json` |
| Character n-gram | 3–5 ký tự trong từng token, có khoảng trắng biên |
| Vocabulary | DF từ corpus; `min_df=2`, `max_df=0.95` |
| TF-IDF | Raw TF; smoothed IDF; chuẩn hóa L2 |
| VSM | Tích vô hướng giữa các vector TF-IDF đã chuẩn hóa |
| BM25 | Smoothed Robertson IDF; `k1=1.2`, `b=0.75`, query terms duy nhất |
| LSI | TruncatedSVD trên TF-IDF corpus; 64 chiều; random seed 42 |

`scripts/prepare_terms.py` sinh văn bản đã xử lý và hai vocabulary. `scripts/build_retrieval_index.py` tạo ma trận term frequency, TF-IDF, IDF, độ dài document và danh sách document IDs cho mỗi biểu diễn. `scripts/evaluate_models.py` nạp index, biến đổi queries bằng **chính vocabulary/IDF của corpus**, xếp hạng và tính metrics.

Không sử dụng qrels để fit vocabulary, IDF, BM25 statistics hoặc SVD. Development train chưa được dùng để supervised training trong các baseline hiện tại. Các tham số trên được chọn cố định trước thí nghiệm; chưa có hyperparameter search.

## 3. Đánh giá

Mỗi cấu hình chạy trên cùng **482 validation queries** ở 8 điều kiện: clean, no diacritics, Telex typo, phonetic, abbreviation, mixed low/medium/high. Xếp hạng tối đa 100 documents; với những query không có term nào trong vocabulary, ranking rỗng và metrics bằng 0.

- **MRR@10:** nghịch đảo thứ hạng của document liên quan đầu tiên trong top 10, trung bình theo query.
- **nDCG@10:** chất lượng thứ hạng trong top 10, có xét tất cả document relevant đã được gán nhãn.
- **Recall@100:** tỷ lệ documents relevant tìm được trong top 100.
- **Performance drop:** mức chênh lệch tuyệt đối và phần trăm so với cùng mô hình/biểu diễn trên query clean.

Công thức phần trăm suy giảm:

```text
drop (%) = (metric_clean - metric_noisy) / metric_clean × 100
```

Nếu `metric_clean = 0`, phần trăm suy giảm không xác định. Một mô hình có mức suy giảm thấp nhưng điểm clean thấp vẫn có thể kém hơn các baseline khác; vì vậy phải xem cả chất lượng tuyệt đối và mức suy giảm.

## 4. Chạy lại từ repository

Yêu cầu: Python 3.11+, [uv](https://docs.astral.sh/uv/) và dữ liệu đặt đúng thư mục như [data/README.md](../data/README.md). Chạy tại root của repository:

```bash
uv sync --locked
uv run python -X utf8 scripts/prepare_validation.py
uv run python -X utf8 scripts/prepare_terms.py --workers 4

uv run python -X utf8 scripts/build_retrieval_index.py --branch word --output-dir outputs/index/word
uv run python -X utf8 scripts/evaluate_models.py --index-dir outputs/index/word --output-dir outputs/retrieval/val_word

uv run python -X utf8 scripts/build_retrieval_index.py --branch char_ngram --output-dir outputs/index/char_ngram
uv run python -X utf8 scripts/evaluate_models.py --index-dir outputs/index/char_ngram --output-dir outputs/retrieval/val_char
```

**Lưu ý:** lệnh đánh giá char ở trên sẽ chạy cả BM25 và LSI, tức 16 cấu hình **chưa hoàn thành trong báo cáo hiện tại**. Đây là hướng dẫn chạy, không phải xác nhận rằng kết quả đã được thu thập.

Để chỉ chạy một mô hình, thêm `--models vsm`, `--models bm25` hoặc `--models lsi`. Các bước có thể tốn nhiều RAM/CPU trên corpus đầy đủ. Chế độ `--max-docs` khi build index và `--allow-subset-smoke` khi đánh giá chỉ phục vụ smoke test; không dùng số liệu đó làm benchmark.

Kiểm tra:

```bash
uv run python -m unittest discover -s tests -q
```

Script split, index và evaluator từ chối ghi đè thư mục kết quả đã có. Khi chạy lại, dùng đường dẫn đầu ra mới.

## 5. Artifact và kiểm chứng

| Thư mục | Nội dung |
|---|---|
| `outputs/splits/seed42/` | Train/validation sạch, qrels và 7 tập validation noise |
| `outputs/processed/` | Corpus/query tiền xử lý, vocabularies, thông tin nguồn |
| `outputs/index/word/` | Count/TF-IDF index, IDF, độ dài và IDs tài liệu |
| `outputs/index/char_ngram/` | Các artifact tương tự cho character n-gram |
| `outputs/retrieval/val_word/` | 24 cấu hình đã đánh giá, `summary.csv` và `per_query.csv` |
| `outputs/retrieval/val_char_full/` | `progress.json` lưu 8 cấu hình VSM char đã hoàn thành |

`outputs/` được bỏ qua bởi Git. Kết quả tổng hợp đã xác minh được đưa lên GitHub trong [validation_results.md](validation_results.md); SHA-256 giúp đối chiếu dữ liệu và index.

Bộ kiểm tra tự động tại lần chạy gần nhất: **37 tests pass**. Audit nguồn/dataset: **67/67 checks pass**.

## 6. Giới hạn nghiên cứu

- Mới có **32/48** cấu hình validation trên full corpus; 16 cấu hình char BM25/LSI còn thiếu.
- Nhiễu là dữ liệu **synthetic**, một số biến thể có thể đổi nghĩa câu hỏi; cần đọc mẫu trước khi suy luận sâu về robustness.
- Qrels chỉ mô tả những document đã được gán nhãn. Document chưa được gán nhãn vẫn có thể liên quan.
- Tập test đã từng được xem xét trong các thí nghiệm trước. Không sử dụng test làm validation và không khẳng định đây là tập test hoàn toàn chưa từng tiếp xúc.
- Kết quả trước đây trên pipeline từ vựng và tập dữ liệu khác không được so sánh trực tiếp với những số liệu hiện tại.

**Đánh giá test cuối cùng** chỉ thực hiện sau khi chốt cấu hình bằng validation:

```bash
uv run python -X utf8 scripts/evaluate_models.py --split test --confirm-test-use --index-dir outputs/index/word --output-dir outputs/retrieval/test_word_final
```

Lệnh trên chỉ là hướng dẫn; **chưa chạy đánh giá test cuối cùng** trong báo cáo hiện tại.
