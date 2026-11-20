# Tiền xử lý và biểu diễn văn bản

Mã nguồn: [scripts/prepare_terms.py](scripts/prepare_terms.py). Cấu hình: [configs/terms.json](configs/terms.json). Đầu vào là corpus và queries đã chuẩn hóa trong `data/zalo_legal/`.

## Chuẩn hóa văn bản

- Tài liệu: nối `title` và `text`, sau đó chuẩn hóa Unicode NFC, chuyển chữ thường và gộp khoảng trắng.
- Query sạch và query nhiễu được chuẩn hóa tương tự. Giữ nguyên dấu tiếng Việt, chữ số và dấu câu; không tự sửa lỗi gõ hoặc khôi phục dấu.
- ID, thứ tự dòng và qrels không bị thay đổi. Script ghi kết quả vào `outputs/processed/`, không ghi đè dữ liệu nguồn.

## Hai cách biểu diễn

**Kết quả của bước chọn term:** pipeline hiện tạo được hai vocabulary (Word và Character n-gram) bằng cách lọc DF từ corpus. Đây là hai phương án để đưa sang Model Evaluation; chưa thực hiện thí nghiệm chọn một biểu diễn tốt nhất. Xem [README.md](README.md) để biết kết quả của từng mô hình trên hai biểu diễn.

| Biểu diễn | Cách tách term | Vocabulary |
|---|---|---|
| `word` | PyVi 0.1.1; từ ghép nối bằng `_`; loại token chỉ chứa dấu câu | 16.498 term |
| `char_ngram` | N-gram 3–5 ký tự trong từng đơn vị ngăn bởi khoảng trắng; thêm dấu cách ở hai biên | 111.181 term |

Ví dụ với `phạt`, nhánh char tạo các đoạn như ` ph`, `phạ`, `hạt`, `ạt `. N-gram không đi xuyên qua ranh giới giữa hai đơn vị cách trắng.

Nhánh word bỏ 5 stopword: `và`, `là`, `của`, `các`, `những`. Vẫn giữ số và những từ có thể làm thay đổi nghĩa pháp lý như `không`, `chưa`, `được`, `phải`. Nhánh character n-gram không bỏ stopword.

**Chọn vocabulary:** tính document frequency (DF) chỉ trên corpus; giữ term có DF ít nhất 2 và xuất hiện ở không quá 95% tài liệu (`min_df=2`, `max_df=0.95`). DF là số tài liệu chứa term, khác với term frequency (TF) là số lần term xuất hiện. Ngưỡng này là baseline cố định, chưa tối ưu theo validation.

## Artifact đầu ra

```text
outputs/processed/
├── corpus.jsonl
├── queries_train.jsonl
├── queries_clean.jsonl
├── noise/queries_*.jsonl
├── qrels_train.tsv
├── qrels_test.tsv
├── vocabulary_word.json
├── vocabulary_char_ngram.json
├── summary.json
└── config_used.json
```

Các file JSONL xử lý chứa `id`, `text`, `word_terms`. `word_terms` giữ tần suất lặp; chưa bị lọc theo DF. Vocabulary lưu `term → DF`. Khi xây dựng ma trận Count/TF-IDF, `scripts/build_retrieval_index.py` đọc vocabulary và chỉ giữ các term có mặt trong đó; nhánh char được tạo trực tiếp từ `text` để tránh lưu lượng lớn n-gram trong JSONL.

## Số liệu đã xác minh

| Chỉ số | Word | Character n-gram |
|---|---:|---:|
| Term trước lọc DF | 26.143 | 195.945 |
| Term sau lọc DF | **16.498** | **111.181** |

Đã chạy preprocessing trên **61.425 tài liệu**, **2.408 train queries**, **788 test queries** và 7 bộ test noise (788 query/bộ). Đã xây dựng index đầy đủ cho cả hai nhánh. Ma trận count và TF-IDF được tạo ở bước riêng, **chỉ fit từ corpus**; xem [docs/model_evaluation.md](docs/model_evaluation.md).

Chạy riêng bước tiền xử lý:

```bash
uv sync --locked
uv run python -X utf8 scripts/prepare_terms.py --workers 4
```

`--workers 1` phù hợp khi cần hạn chế tài nguyên. Output lớn và được bỏ qua bởi Git. [PyVi](https://github.com/trungtv/pyvi) là thư viện tách từ được sử dụng.
